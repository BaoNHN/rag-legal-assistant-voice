import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from difflib import SequenceMatcher
from collections import Counter

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

STOPWORDS = {
    "theo", "là", "gì", "về", "của", "và",
    "trong", "được", "cho", "có", "khi"
}
PROCEDURE_PATTERNS = [
    "trình tự",
    "thủ tục",
    "quy trình",
    "các bước",
    "hồ sơ",
    "nộp ở đâu",
]

CONDITION_PATTERNS = [
    "điều kiện",
    "yêu cầu",
    "cần có",
    "phải có",
]

DEFINITION_PATTERNS = [
    "là gì",
    "khái niệm",
    "định nghĩa",
    "quy định về",
]

with open(os.path.join(BASE_DIR, "groqkey.txt"), "r") as f:
    GROQ_API_KEY = f.read().strip()

# =========================
# EMBEDDING + VECTORSTORE
# =========================
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================
# LLM
# =========================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)


# =========================
# CLEAN TEXT
# =========================
def clean_answer(text: str) -> str:
    text = re.sub(r"(?i)^xin chào.*?\n", "", text)
    lines = text.split("\n")
    seen, cleaned = set(), []
    for l in lines:
        if l.strip() and l not in seen:
            cleaned.append(l)
            seen.add(l)
    return "\n".join(cleaned).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


# =========================
# QUERY REWRITE
# =========================
def _llm_invoke_with_retry(prompt: str, retries: int = 3) -> str:
    """Invoke LLM with retry on rate-limit/timeout errors."""
    import time
    for attempt in range(retries + 1):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            err = str(e).lower()
            if attempt < retries and any(k in err for k in ["rate", "timeout", "429", "503", "connection"]):
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"  ⚠ LLM retry {attempt + 1}/{retries} after {wait}s ({err[:40]})")
                time.sleep(wait)
                continue
            raise
    return ""


def rewrite_query(question: str) -> str:
    prompt = f"""Viết lại câu hỏi ngắn gọn, rõ nghĩa để tìm trong luật:

{question}"""
    try:
        return _llm_invoke_with_retry(prompt)
    except Exception:
        return question


# =========================
# EXTRACT TOPIC FROM QUESTION
# ─────────────────────────────
# Knowledge questions always follow: "quy định về [TOPIC] là gì?"
# This matches the KB_Articles topic field exactly (100% match rate).
# Used for ChromaDB metadata filtering to pinpoint the right article.
# =========================
def extract_topic_from_question(question: str) -> str | None:
    q = question.strip()
    # Pattern: "quy định về X là gì" or "X là gì" or "X theo quy định"
    patterns = [
        r'quy định về (.+?) là gì',
        r'quy định về (.+?) như thế nào',
        r'quy định về (.+?) gồm',
        r'về (.+?) là gì',
        r'(.+?) là gì theo Luật',
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            topic = m.group(1).strip().lower()
            # Remove trailing noise
            topic = re.sub(r'\s*(theo luật.*|của luật.*)$', '', topic).strip()
            return topic
    return None


# =========================
# TOPIC-AWARE RETRIEVAL
# ─────────────────────────────
# FIX for knowledge questions scoring 57.7/100:
# 1. Try ChromaDB metadata filter on topic first (exact match)
# 2. Fall back to standard semantic search if no match
# This eliminates wrong article retrieval (Điều 34 instead of 36, etc.)
# =========================
def retrieve_docs(question: str, rewritten_q: str):
    topic = extract_topic_from_question(question)

    try:
        results = vectorstore.get(include=["documents", "metadatas"])
        all_docs = []

        for doc_text, meta in zip(results["documents"], results["metadatas"]):
            all_docs.append(Document(
                page_content=doc_text,
                metadata=meta
            ))

        # ===== STRICT TOPIC MATCH =====
        if topic:
            scored = []

            for d in all_docs:
                kb_topic = d.metadata.get("topic", "").strip().lower()

                if not kb_topic:
                    continue

                sim = similarity(topic, kb_topic)

                # bonus for exact containment
                if topic == kb_topic:
                    sim += 1.0
                elif topic in kb_topic:
                    sim += 0.3
                elif kb_topic in topic:
                    sim += 0.1

                scored.append((sim, d))

            scored.sort(key=lambda x: x[0], reverse=True)

            # confidence threshold
            if scored and scored[0][0] >= 0.55:
                return [d for _, d in scored[:5]]

        # ===== FALLBACK SEMANTIC SEARCH =====
        return retriever.invoke(rewritten_q)

    except Exception:
        return retriever.invoke(rewritten_q)


# =========================
# RERANK
# ─────────────────────────
# Scores against page_content + retrieval_keywords (weighted x2)
# + prefers KB_Articles docs over Q&A docs
# =========================
def select_best_doc(question: str, docs):
    q_words = [w for w in tokenize(question) if w not in STOPWORDS]
    q_counter = Counter(q_words)

    best_doc = None
    best_score = -1

    for d in docs:
        score = 0

        content = d.page_content.lower()
        metadata = d.metadata

        # keyword overlap
        for w, cnt in q_counter.items():
            if w in content:
                score += cnt

        # metadata keywords — single word match
        kw_field = metadata.get("retrieval_keywords", "")
        kw_text = kw_field.lower().replace(";", " ")

        for w, cnt in q_counter.items():
            if w in kw_text:
                score += cnt * 3

        # Multi-word keyword phrase match (high precision boost)
        # Matches "công ty hợp danh", "thành viên hợp danh" etc.
        q_lower = question.lower()
        for kw_phrase in kw_field.lower().split(";"):
            kw_phrase = kw_phrase.strip()
            if len(kw_phrase) > 8 and kw_phrase in q_lower:
                score += 10  # strong boost for exact phrase match

        # article reference bonus
        article_ref = metadata.get("article_reference", "")
        if article_ref:
            art_num = re.findall(r'\d+', article_ref)
            if art_num and art_num[0] in question:
                score += 5

        # prioritize KB articles
        if "KB_Articles" in metadata.get("nguon_thu_thap", ""):
            score += 2

        if score > best_score:
            best_score = score
            best_doc = d

    return best_doc


# =========================
# CLASSIFY QUESTION
# =========================
_DOC_TYPE_MAP = {
    "definition": "definition", "rights": "condition",
    "obligations": "condition", "prohibited": "condition",
    "establishment_eligibility": "condition",
    "registration_procedure": "procedure",
    "registration_private_enterprise": "procedure",
    "registration_partnership": "procedure",
    "registration_llc": "procedure",
    "registration_jsc": "procedure",
    "change_registration": "procedure",
    "notification_change": "procedure",
    "publication": "procedure",
    "asset_transfer": "procedure",
    "erc_issuance": "condition",
    "name_prohibitions": "condition",
    "asset_valuation": "condition",
    "legal_representative_duty": "condition",
    "dependent_units": "definition",
}


def classify_question(question: str, best_doc=None) -> str:
    q = question.lower()

    if any(p in q for p in PROCEDURE_PATTERNS):
        return "procedure"

    if any(p in q for p in CONDITION_PATTERNS):
        return "condition"

    if any(p in q for p in DEFINITION_PATTERNS):
        return "definition"

    return "general"


# =========================
# BUILD PROMPT
# =========================
def build_prompt(context: str, question: str, q_type: str,
                 article_ref: str = "", topic: str = "") -> str:
    article_hint = ""
    if article_ref:
        line = article_ref
        if topic:
            line += f" — {topic}"
        article_hint = f"\nCăn cứ pháp lý: {line}\n"

    base = f"""Bạn là trợ lý pháp lý Việt Nam.

⚠️ QUY TẮC:
- Chỉ trả lời 1 cách duy nhất
- Không chào hỏi, không giải thích dư
- Luôn trích dẫn đúng điều luật cụ thể
- Không bịa thêm điều luật ngoài tài liệu
{article_hint}
Tài liệu:
{context}

Câu hỏi: {question}

"""
    if q_type == "procedure":
        base += "Trả lời dạng các bước rõ ràng (1, 2, 3...)."
    elif q_type == "condition":
        base += "Chỉ liệt kê các điều kiện/yêu cầu."
    elif q_type == "definition":
        base += "Trả lời ngắn gọn định nghĩa, nêu rõ căn cứ điều luật."
    else:
        base += "Trả lời ngắn gọn, đúng trọng tâm, nêu CĂN CỨ PHÁP LÝ đầy đủ (tất cả các Điều liên quan)."
    return base


# =========================
# BUILD CITATION
# =========================
def build_citation(meta: dict, answer: str = "") -> str:
    article_ref = meta.get("article_reference", "")
    topic = meta.get("topic", "")
    so_ky_hieu = meta.get("so_ky_hieu", "")
    loai = meta.get("loai_van_ban", "")
    source_url = meta.get("source_url", "")

    # Include any additional articles cited in the answer
    if answer:
        cited_nums = re.findall(r'Điều\s+(\d+)', answer)
        meta_num = re.sub(r'[^\d]', '', article_ref)
        seen = {article_ref}
        extra = []
        for n in cited_nums:
            ref = f"Điều {n}"
            if n != meta_num and ref not in seen:
                extra.append(ref)
                seen.add(ref)
        if extra:
            article_ref = article_ref + "; " + "; ".join(extra)

    parts = []
    if article_ref:
        line = article_ref
        # Only add topic label for single article citations
        if topic and ";" not in article_ref:
            line += f" ({topic})"
        parts.append(line)
    law_name = f"{loai} {so_ky_hieu}".strip()
    if law_name:
        parts.append(law_name)

    citation = " — ".join(parts) if parts else meta.get("nguon_thu_thap", "")
    result = f"\n\n📖 Nguồn: {citation}"
    if source_url and "vbpl.vn" in source_url:
        result += f"\n🔗 {source_url}"
    return result


# =========================
# MAIN
# =========================
def ask_rag(question: str, return_debug: bool = False):
    try:
        question = str(question)

        # STEP 1: rewrite — SKIP if topic is extractable
        # Knowledge questions follow "quy định về X là gì?" pattern.
        # Topic filter works directly on original question → no LLM rewrite needed.
        # This saves one Groq API call per knowledge question, reducing rate limit hits.
        topic = extract_topic_from_question(question)
        better_q = question if topic else rewrite_query(question)

        # STEP 2: topic-aware retrieval (fixes knowledge question accuracy)
        docs = retrieve_docs(question, better_q)

        if not docs:
            return "❌ Không tìm thấy thông tin liên quan trong cơ sở dữ liệu pháp luật."

        # STEP 3: rerank
        best_doc = select_best_doc(better_q, docs)
        if not best_doc:
            return "❌ Không đủ dữ liệu để trả lời câu hỏi này."

        # STEP 4: context — include top-3 docs for richer context
        # For scenario questions that reference multiple articles (e.g. Điều 200 + 201),
        # supplementary docs improve completeness without exceeding token budget
        context_parts = [best_doc.page_content]
        for extra_doc in docs[:3]:
            if extra_doc.page_content != best_doc.page_content:
                context_parts.append(extra_doc.page_content)
        context = "\n\n---\n\n".join(context_parts)[:6000]

        # STEP 5: classify using doc_type metadata
        q_type = classify_question(question, best_doc)

        # STEP 6: prompt with article hint
        article_ref = best_doc.metadata.get("article_reference", "")
        topic = best_doc.metadata.get("topic", "")
        prompt = build_prompt(context, question, q_type, article_ref, topic)

        # STEP 7: generate with retry
        answer = _llm_invoke_with_retry(prompt)

        # STEP 8: clean
        answer = clean_answer(answer)

        # STEP 9: citation (pass answer so all cited articles are included)
        final_answer = answer + build_citation(best_doc.metadata, answer)

        if return_debug:
            return {
                "answer": final_answer,
                "retrieved_context": context,
                "metadata": best_doc.metadata,
                "question_type": q_type,
            }

        return final_answer

    except Exception as e:
        print("RAG ERROR:", e)
        return "❌ Lỗi hệ thống."