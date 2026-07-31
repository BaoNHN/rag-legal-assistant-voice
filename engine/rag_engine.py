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

# Keywords indicating out-of-scope questions (not business law)
OUT_OF_SCOPE_KEYWORDS = [
    "ly hôn", "li hôn", "hôn nhân", "gia đình", "ly thân", "kết hôn",
    "hình sự", "tội phạm", "khởi tố", "bắt giữ", "truy tố", "tù giam",
    "đất đai", "nhà ở", "bất động sản", "quyền sử dụng đất",
    "bảo hiểm xã hội", "bảo hiểm y tế", "tai nạn lao động",
    "thuế thu nhập cá nhân", "thuế giá trị gia tăng", "thuế tiêu thụ",
    "hải quan", "xuất nhập khẩu", "hành chính công",
]

SIMILARITY_THRESHOLD = 1.3  # L2 distance; above this → not relevant enough
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
# Citation footer is appended programmatically by build_citation() after the
# LLM returns — these markers must never appear in the model's own text. If
# the model hallucinates its own fake "source" (seen in practice: invented
# document codes and lecture-material attributions), strip it here so it
# can't collide with or shadow the real citation.
_FAKE_CITATION_LINE = re.compile(
    r"^\s*[-•*]?\s*(📖|📎|Nguồn chính\s*:|Nguồn tham khảo\s*:|Tài liệu\s*:|Nguồn thu thập\s*:)",
    re.IGNORECASE
)


def clean_answer(text: str) -> str:
    text = re.sub(r"(?i)^xin chào.*?\n", "", text)
    lines = text.split("\n")
    seen, cleaned = set(), []
    for l in lines:
        if _FAKE_CITATION_LINE.match(l):
            continue
        if l.strip() and l not in seen:
            cleaned.append(l)
            seen.add(l)
    return "\n".join(cleaned).strip()


# =========================
# CITATION SOURCE WHITELIST
# ─────────────────────────────
# Regex-stripping hallucinated citation lines (above) only catches text the
# model wrote in an obviously footer-shaped way. It can't tell a fabricated
# so_ky_hieu ("TH-LDN-20-2026") from a real one if it ever ends up in a
# Document's metadata (stale import, DB drift, manual edit, etc.). This
# whitelist is the harder guarantee: it's rebuilt from what's *actually*
# indexed in chroma_db right now, stored in chat.db (Const.CITATION_SOURCE,
# ";"-joined), and build_citation() refuses to print any so_ky_hieu that
# isn't in it.
# =========================
CITATION_SOURCE_KEY = "CITATION_SOURCE"


def refresh_citation_sources():
    """Rebuild the Const.CITATION_SOURCE whitelist from chroma_db's current
    so_ky_hieu values. Call this after any change to chroma_db (law import,
    dataset import) so the whitelist never lags behind what's really indexed.
    """
    from database.database import set_const

    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return

    sources = set()
    for m in data["metadatas"]:
        val = (m.get("so_ky_hieu") or "").strip()
        # ";" is the whitelist's own delimiter — a source containing it would
        # corrupt parsing, so it's excluded rather than risk a false split/match.
        if val and ";" not in val:
            sources.add(val)

    set_const(CITATION_SOURCE_KEY, ";".join(sorted(sources)))


def is_known_citation_source(source: str) -> bool:
    """True if `source` is a so_ky_hieu that's actually indexed in chroma_db
    (per the last refresh_citation_sources() run)."""
    from database.database import get_const

    source = (source or "").strip()
    if not source:
        return False

    known_raw = get_const(CITATION_SOURCE_KEY)
    if not known_raw:
        # Whitelist never populated (e.g. very first run before any import
        # triggered a refresh) — fail open instead of blanking every citation.
        return True
    return source in known_raw.split(";")


def list_indexed_sources() -> list:
    """Distinct so_ky_hieu values currently indexed in chroma_db, with chunk
    counts — used by the admin UI to show what can be deleted."""
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return []

    counts = Counter((m.get("so_ky_hieu") or "").strip() for m in data["metadatas"])
    counts.pop("", None)
    return [
        {"so_ky_hieu": k, "chunk_count": v}
        for k, v in sorted(counts.items())
    ]


def delete_source(so_ky_hieu: str) -> int:
    """Delete every chunk in chroma_db whose so_ky_hieu matches, then rebuild
    the CITATION_SOURCE whitelist from what's left indexed — this is what
    keeps the whitelist from lagging a deleted import (see
    refresh_citation_sources above). Returns the number of chunks deleted.
    """
    so_ky_hieu = (so_ky_hieu or "").strip()
    if not so_ky_hieu:
        return 0

    existing = vectorstore.get(where={"so_ky_hieu": so_ky_hieu}, include=[])
    ids = existing.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)

    refresh_citation_sources()
    return len(ids)


# =========================
# MANAGE LAW — DATASET / SCENARIO SOURCES
# ─────────────────────────────
# Import Law tags every chunk with so_ky_hieu (see list_indexed_sources/
# delete_source above), so it needs no import_source tag. Dataset and Scenario
# imports carry their own "import_source" tag (set in import_dataset_engine /
# import_scenario_engine) so the Manage Law page can group and delete them by
# uploaded file without touching unrelated chunks that happen to share a
# so_ky_hieu (Dataset rows are all stamped so_ky_hieu="59/2020/QH14", the same
# code as the real imported law text).
# =========================
UNKNOWN_SOURCE_FILE_LABEL = "(không rõ tệp — nhập trước khi có tính năng theo dõi tệp)"


def backfill_import_source_tags():
    """One-time migration for chunks indexed before import_source/source_file
    existed. Idempotent — only touches chunks that don't have import_source
    yet, so it's cheap and safe to call on every startup."""
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return

    ids   = data.get("ids") or []
    metas = data.get("metadatas") or []

    update_ids, update_metas = [], []
    for doc_id, m in zip(ids, metas):
        if m.get("import_source"):
            continue

        if "segment_index" in m:
            tag = "law"
        elif m.get("doc_type") == "scenario_qa":
            tag = "scenario"
        elif (m.get("so_ky_hieu") or "").strip() == "59/2020/QH14":
            tag = "dataset"
        else:
            # e.g. database/reference_source.py entries — not one of the 3
            # UI-driven import types, leave untagged.
            continue

        new_meta = dict(m)
        new_meta["import_source"] = tag
        if tag == "dataset" and not new_meta.get("source_file"):
            new_meta["source_file"] = UNKNOWN_SOURCE_FILE_LABEL
        update_ids.append(doc_id)
        update_metas.append(new_meta)

    if update_ids:
        vectorstore._collection.update(ids=update_ids, metadatas=update_metas)


def list_dataset_sources() -> list:
    """Distinct uploaded .xlsx filenames among dataset-origin chunks, with
    chunk counts."""
    try:
        data = vectorstore.get(where={"import_source": "dataset"}, include=["metadatas"])
    except Exception:
        return []

    counts = Counter((m.get("source_file") or UNKNOWN_SOURCE_FILE_LABEL).strip() for m in data["metadatas"])
    return [{"name": k, "chunk_count": v} for k, v in sorted(counts.items())]


def delete_dataset_source(source_file: str) -> int:
    """Delete every dataset-origin chunk stamped with this source_file."""
    source_file = (source_file or "").strip()
    if not source_file:
        return 0

    existing = vectorstore.get(
        where={"$and": [{"import_source": "dataset"}, {"source_file": source_file}]},
        include=[],
    )
    ids = existing.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)

    refresh_citation_sources()
    return len(ids)


def list_scenario_sources() -> list:
    """Distinct uploaded .docx filenames among scenario_qa chunks, with chunk
    counts."""
    try:
        data = vectorstore.get(where={"doc_type": "scenario_qa"}, include=["metadatas"])
    except Exception:
        return []

    counts = Counter((m.get("nguon_thu_thap") or UNKNOWN_SOURCE_FILE_LABEL).strip() for m in data["metadatas"])
    return [{"name": k, "chunk_count": v} for k, v in sorted(counts.items())]


def delete_scenario_source(name: str) -> int:
    """Delete every scenario_qa chunk whose source filename (nguon_thu_thap)
    matches."""
    name = (name or "").strip()
    if not name:
        return 0

    existing = vectorstore.get(
        where={"$and": [{"doc_type": "scenario_qa"}, {"nguon_thu_thap": name}]},
        include=[],
    )
    ids = existing.get("ids") or []
    if ids:
        vectorstore.delete(ids=ids)

    refresh_citation_sources()
    return len(ids)


# Populate the whitelist once at startup so it reflects chroma_db even if no
# import happens during this process's lifetime.
refresh_citation_sources()
backfill_import_source_tags()


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
        r'^(.+?) là gì\??$',          # bare "X là gì?" with no prefix
        r'^(.+?) được hiểu là gì\??$',
        r'^(.+?) có nghĩa là gì\??$',
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
# EXTRACT EXPLICIT ARTICLE NUMBER
# ─────────────────────────────
# Questions like "Điều 143" or "Điều 143 quy định gì?" name the article
# directly. Semantic embedding search is unreliable for this (bge-small
# is not tuned for Vietnamese legal numerals), so we match it against the
# article_number metadata exactly instead of relying on vector similarity.
# =========================
def extract_article_number_from_question(question: str) -> str | None:
    m = re.search(r'điều\s+(\d+[a-z]?)\b', question, re.IGNORECASE)
    return m.group(1) if m else None


# =========================
# KEYWORD-PHRASE RECALL
# ─────────────────────────────
# Token-overlap scoring against retrieval_keywords/page_content, independent
# of embedding similarity. Requires a minimum score so a single stray common
# word doesn't drag in an unrelated article — this is a recall booster for
# the semantic search, not a replacement for select_best_doc()'s reranking.
# =========================
def _keyword_recall(question: str, all_docs: list, top_n: int = 5, min_score: int = 4) -> list:
    q_words = {w for w in tokenize(question) if w not in STOPWORDS}
    if not q_words:
        return []

    scored = []
    for d in all_docs:
        kw_field = d.metadata.get("retrieval_keywords", "").lower()
        if not kw_field:
            continue
        content = d.page_content.lower()
        score = 0
        for w in q_words:
            if w in kw_field:
                score += 3
            elif w in content:
                score += 1
        if score >= min_score:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]


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
    article_num = extract_article_number_from_question(question)

    try:
        results = vectorstore.get(include=["documents", "metadatas"])
        all_docs = []

        for doc_text, meta in zip(results["documents"], results["metadatas"]):
            all_docs.append(Document(
                page_content=doc_text,
                metadata=meta
            ))

        # ===== EXACT ARTICLE NUMBER MATCH =====
        # "Điều 143" etc. — filter on article_number metadata exactly instead
        # of trusting embedding similarity to find the right numbered article.
        if article_num:
            exact = [d for d in all_docs if d.metadata.get("article_number", "") == article_num]
            if exact:
                return exact[:5]

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

        # ===== BARE-KEYWORD SUBSTRING MATCH =====
        # Short queries like "Tập đoàn" (no question phrasing, no topic/article
        # match above) — embedding similarity is unreliable for 1-3 word
        # Vietnamese legal terms with an English-tuned model. Scan titles and
        # content directly for the phrase instead.
        q_lower = question.lower().strip()
        if q_lower and len(q_lower.split()) <= 5:
            kw_hits = [
                d for d in all_docs
                if q_lower in d.page_content.lower() or q_lower in d.metadata.get("title", "").lower()
            ]
            if kw_hits:
                return kw_hits[:5]

        # ===== HYBRID: keyword-phrase recall + semantic search =====
        # For longer, naturally-phrased questions, bge-small-en-v1.5 (English-
        # tuned) routinely misses the right article even when its
        # retrieval_keywords field is an near-exact match for the question —
        # e.g. "B 16 tuổi có được đăng ký hộ kinh doanh không?" never surfaced
        # the Điều 20/21 BLDS 2015 doc (keywords: "chưa đủ 18 tuổi", "hộ kinh
        # doanh"...) in the semantic top-5. Score every doc by keyword-token
        # overlap independent of embedding similarity, and merge those hits
        # ahead of the semantic results so select_best_doc() actually gets a
        # chance to consider them.
        kw_candidates = _keyword_recall(question, all_docs)

        results_with_scores = vectorstore.similarity_search_with_score(rewritten_q, k=5)
        # Filter: L2 distance lower = more similar; reject docs above threshold
        semantic_candidates = [doc for doc, score in results_with_scores if score <= SIMILARITY_THRESHOLD]

        merged, seen = [], set()
        for d in kw_candidates + semantic_candidates:
            if d.page_content not in seen:
                merged.append(d)
                seen.add(d.page_content)

        return merged

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

    base = f"""Bạn là trợ lý pháp lý Việt Nam chuyên về Luật Doanh nghiệp.

⚠️ QUY TẮC BẮT BUỘC:
- Câu trả lời TỐI ĐA 200 từ.
- Không chào hỏi, không giải thích dư thừa.
- Không liệt kê toàn bộ văn bản luật — chỉ trả lời đúng nội dung câu hỏi.
- CĂN CỨ PHÁP LÝ phải lấy ĐÚNG từ tài liệu được cung cấp. KHÔNG tự suy diễn hay thay đổi số điều luật.
- Nếu tài liệu có metadata căn cứ pháp lý, phải sử dụng đúng điều luật đó.
- CHỈ được trích số Điều xuất hiện NGUYÊN VĂN trong phần "Tài liệu" bên dưới. Nếu không thấy số Điều liên quan trong tài liệu, hãy nói rõ là tài liệu không đề cập, KHÔNG được đoán hay lấy từ kiến thức chung.
- KHÔNG được tự đặt tên nguồn, mã văn bản, tên tác giả/giảng viên hay bất kỳ trích dẫn nào — phần nguồn tài liệu do hệ thống tự thêm vào sau, bạn không viết phần đó.
{article_hint}
Tài liệu:
{context}

Câu hỏi: {question}

"""
    STRUCT = (
        "\n\nTrả lời theo đúng cấu trúc sau (bắt buộc, mỗi mục một dòng riêng):\n"
        "**Kết luận:** [1 câu trả thẳng câu hỏi]\n"
        "**Phân tích:** [2-3 câu giải thích ngắn]\n"
        "**Lưu ý:** [1 điểm đặc biệt cần nhớ, bỏ dòng này nếu không có]\n"
        "Tổng cộng tối đa 200 từ. Không viết gì ngoài 3 mục trên."
    )
    if q_type == "procedure":
        base += "Trong **Phân tích**, liệt kê các bước theo thứ tự (1, 2, 3...)." + STRUCT
    elif q_type == "condition":
        base += "Trong **Phân tích**, liệt kê ngắn gọn từng điều kiện." + STRUCT
    elif q_type == "definition":
        base += "Trong **Kết luận**, nêu định nghĩa. Trong **Phân tích**, giải thích chi tiết." + STRUCT
    else:
        base += STRUCT
    return base


# =========================
# BUILD CITATION
# =========================
def build_citation(meta: dict, answer: str = "", secondary_docs=None, context: str = "") -> str:
    article_ref = meta.get("article_reference", "")
    topic = meta.get("topic", "")
    so_ky_hieu = meta.get("so_ky_hieu", "")
    loai = meta.get("loai_van_ban", "")
    source_url = meta.get("source_url", "")

    # Refuse to print a document code that isn't actually indexed in chroma_db
    # right now — catches stale/fabricated so_ky_hieu before it reaches the user.
    if so_ky_hieu and not is_known_citation_source(so_ky_hieu):
        so_ky_hieu = ""
        source_url = ""

    # Include any additional articles cited in the answer — but only ones that
    # actually appear in the retrieved context. Otherwise a number the model
    # recalled from general knowledge (or hallucinated) gets credited to a
    # source document that never mentioned it.
    #
    # If a cited number is known (via secondary_docs) to belong to a DIFFERENT
    # law than this primary document, don't lump it into this line's law_name
    # grouping — that mislabels it as if it came from the primary document's
    # law (e.g. "Điều 80; Điều 21 — Nghị định 01/2021/NĐ-CP" when Điều 21 is
    # actually from Bộ luật Dân sự 91/2015/QH13). It's still shown correctly,
    # under its own so_ky_hieu, in the "Nguồn tham khảo" section below.
    other_law_refs = {
        d.metadata.get("article_reference", "")
        for d in (secondary_docs or [])
        if d.metadata.get("so_ky_hieu", "") and d.metadata.get("so_ky_hieu", "") != so_ky_hieu
    }

    if answer:
        cited_nums = re.findall(r'Điều\s+(\d+)', answer)
        meta_num = re.sub(r'[^\d]', '', article_ref)
        seen = {article_ref}
        extra = []
        for n in cited_nums:
            ref = f"Điều {n}"
            if n == meta_num or ref in seen:
                continue
            if ref in other_law_refs:
                continue
            if context and ref not in context:
                continue
            extra.append(ref)
            seen.add(ref)
        if extra:
            article_ref = f"{article_ref}; " + "; ".join(extra) if article_ref else "; ".join(extra)

    parts = []
    if article_ref:
        line = article_ref
        if topic and ";" not in article_ref:
            line += f" ({topic})"
        parts.append(line)

    law_name = f"{loai} {so_ky_hieu}".strip()
    # Always cite the consolidated text (67/VBHN-VPQH 2025) for Enterprise Law
    if ("67/VBHN" in so_ky_hieu or "59/2020" in so_ky_hieu or
            "67/VBHN" in law_name or "59/2020" in law_name):
        law_name = "Văn bản hợp nhất Luật Doanh nghiệp số 67/VBHN-VPQH năm 2025"
    if law_name:
        parts.append(law_name)

    citation = " — ".join(parts) if parts else meta.get("nguon_thu_thap", "")
    result = f"\n\n📖 Nguồn chính: {citation}"
    if source_url and "vbpl.vn" in source_url:
        result += f"\n🔗 {source_url}"

    # Secondary sources (max 3, deduplicated, short format)
    if secondary_docs:
        primary_ref = meta.get("article_reference", "")
        seen_refs = {primary_ref} if primary_ref else set()
        secondary_lines = []

        for doc in secondary_docs[:3]:
            m = doc.metadata
            s_article = m.get("article_reference", "")
            s_topic = m.get("topic", "")
            s_ky_hieu = m.get("so_ky_hieu", "")

            if not s_article or s_article in seen_refs:
                continue
            seen_refs.add(s_article)

            if s_ky_hieu and not is_known_citation_source(s_ky_hieu):
                s_ky_hieu = ""

            if "67/VBHN" in s_ky_hieu or "59/2020" in s_ky_hieu:
                s_law = "VBHN 67/VBHN-VPQH 2025"
            else:
                s_law = s_ky_hieu[:20] if s_ky_hieu else ""

            s_line = s_article
            if s_topic and ";" not in s_article:
                short_topic = s_topic[:25] + "…" if len(s_topic) > 25 else s_topic
                s_line += f" ({short_topic})"
            if s_law:
                s_line += f" — {s_law}"

            if len(s_line) > 100:
                s_line = s_line[:97] + "…"

            secondary_lines.append(f"• {s_line}")

        if secondary_lines:
            result += "\n📎 Nguồn tham khảo:\n" + "\n".join(secondary_lines)

    return result


# =========================
# MAIN
# =========================
def _is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS)


# ── Meta: hỏi về database/hệ thống ──────────────────
_META_DB_PATTERNS = [
    r'(database|cơ sở dữ liệu|hệ thống).*(bao nhiêu|lưu|chứa)',
    r'(đang lưu|lưu giữ|có chứa).*(bao nhiêu)',
    r'bao nhiêu (điều luật|văn bản|tài liệu).*(lưu|database|hệ thống)',
]

def _is_meta_db_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _META_DB_PATTERNS)

def _answer_meta_db() -> str:
    try:
        data   = vectorstore.get(include=["metadatas"])
        metas  = data["metadatas"]
        total  = len(metas)
        refs   = {m.get("article_reference", "") for m in metas if m.get("article_reference")}
        refs.discard("")
        nguons = {m.get("so_ky_hieu", "") for m in metas if m.get("so_ky_hieu")}
        nguons.discard("")
        return (
            f"**Kết luận:** Hệ thống đang lưu {total} đoạn văn bản, "
            f"tương ứng {len(refs)} điều luật khác nhau.\n"
            f"**Phân tích:** Dữ liệu được lấy từ {len(nguons)} nguồn văn bản pháp luật: "
            f"{', '.join(sorted(nguons))}. "
            f"Mỗi điều luật có thể được lưu thành nhiều đoạn để tối ưu tìm kiếm.\n"
            f"**Lưu ý:** Con số này có thể tăng khi giáo viên import thêm văn bản pháp luật mới."
            f"\n\n📖 Nguồn: ChromaDB — RAG Legal Assistant"
        )
    except Exception as e:
        return f"❌ Không thể truy vấn database: {e}"


# ── Meta: hỏi tổng số điều của bộ luật ──────────────
_META_LAW_COUNT_PATTERNS = [
    r'luật doanh nghiệp.*(có|gồm|bao gồm).*(bao nhiêu điều|mấy điều)',
    r'bao nhiêu điều.*(luật doanh nghiệp)',
    r'(59/2020|67/vbhn|luật doanh nghiệp).*(bao nhiêu|mấy).*(điều|chương)',
    r'(bao nhiêu|mấy) (điều|chương).*(luật doanh nghiệp|59/2020|67/vbhn)',
    r'(cấu trúc|cơ cấu|gồm).*(chương|điều).*(luật doanh nghiệp)',
]

def _is_meta_law_count_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _META_LAW_COUNT_PATTERNS)

def _answer_meta_law_count(question: str) -> str:
    prompt = (
        "Bạn là trợ lý pháp lý Việt Nam. Trả lời ngắn gọn câu hỏi sau về cấu trúc Luật Doanh nghiệp.\n"
        "Trả lời theo cấu trúc:\n"
        "**Kết luận:** [1 câu trực tiếp]\n"
        "**Phân tích:** [2-3 câu chi tiết về cấu trúc luật]\n"
        "**Lưu ý:** [điểm cần nhớ]\n\n"
        f"Câu hỏi: {question}"
    )
    try:
        answer = _llm_invoke_with_retry(prompt)
        return answer + "\n\n📖 Nguồn: Kiến thức tổng quát về Luật Doanh nghiệp Việt Nam"
    except Exception as e:
        return f"❌ Lỗi: {e}"


def ask_rag(question: str, return_debug: bool = False):
    try:
        question = str(question)

        # Pre-check: question outside business law scope
        if _is_out_of_scope(question):
            return "⚠️ Câu hỏi này nằm ngoài phạm vi dữ liệu Luật Doanh nghiệp của hệ thống. Vui lòng đặt câu hỏi liên quan đến Luật Doanh nghiệp."

        # Pre-check: meta about law structure (check before db — more specific)
        if _is_meta_law_count_question(question):
            return _answer_meta_law_count(question)

        # Pre-check: meta question about the database
        if _is_meta_db_question(question):
            return _answer_meta_db()

        # STEP 1: rewrite — SKIP if topic is extractable
        # Knowledge questions follow "quy định về X là gì?" pattern.
        # Topic filter works directly on original question → no LLM rewrite needed.
        # This saves one Groq API call per knowledge question, reducing rate limit hits.
        topic = extract_topic_from_question(question)
        better_q = question if topic else rewrite_query(question)

        # STEP 2: topic-aware retrieval (fixes knowledge question accuracy)
        docs = retrieve_docs(question, better_q)

        if not docs:
            return "⚠️ Không tìm thấy thông tin đủ liên quan trong cơ sở dữ liệu. Vui lòng hỏi rõ hơn hoặc kiểm tra câu hỏi có thuộc phạm vi Luật Doanh nghiệp không."

        # STEP 3: rerank
        best_doc = select_best_doc(better_q, docs)
        if not best_doc:
            return "❌ Không đủ dữ liệu để trả lời câu hỏi này."

        # STEP 4: context — include top-3 docs for richer context
        context_parts = [best_doc.page_content]
        extra_docs = []
        for extra_doc in docs[:3]:
            if extra_doc.page_content != best_doc.page_content:
                context_parts.append(extra_doc.page_content)
                extra_docs.append(extra_doc)
        context = "\n\n---\n\n".join(context_parts)[:3000]

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

        # STEP 9: citation (pass answer + extra_docs for secondary sources)
        final_answer = answer + build_citation(best_doc.metadata, answer, extra_docs, context)

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