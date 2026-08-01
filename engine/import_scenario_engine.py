# import_scenario_engine.py
# Background worker: parse a "Bộ tình huống" (scenario/case-study) DOCX and
# index it into ChromaDB.
#
# Expected DOCX shape (see BO_20_TINH_HUONG_THANH_LAP_DOANH_NGHIEP_CHATBOT_2026.docx):
#   Heading 1 "Tình huống NN. <topic>"
#     Normal    "Mã: <case_id>   Độ khó: <difficulty>"
#     Heading 2 "1. Đề bài"
#       Normal  "Tình huống: <scenario>"
#       Normal  "Câu hỏi: <user_question>"
#     Heading 2 "2. Câu hỏi dẫn dắt xác định vấn đề pháp lý"
#       Normal  (one issue-question per paragraph)
#     Heading 2 "3. Đáp án theo phương pháp IRAC"
#       Normal  "I – Issue: ..." / "R – Rule: ..." / "A – Application: ..." / "C – Conclusion: ..."
#     Heading 2 "4. Căn cứ pháp lý"
#       Normal  (one legal reference per paragraph)
#     Heading 2 "5. Dữ liệu hỗ trợ truy xuất chatbot"
#       Normal  "Từ khóa: k1; k2; k3"
#       Normal  "Câu hỏi tương đương: q1 | q2"
#
# Each case becomes its own chunk. Unlike Import Law, no so_ky_hieu/article_reference
# is fabricated: a single case can cite several articles across several laws (see
# TLDN_020 citing both Luật Doanh nghiệp and Nghị định 168/2025/NĐ-CP), so there is
# no single correct "so_ky_hieu" to attach — attaching one would risk exactly the
# fabricated-citation problem engine.rag_engine's CITATION_SOURCE whitelist exists
# to prevent. The full legal_basis list is kept in metadata and in the chunk text
# instead, and citation falls back to nguon_thu_thap (see build_citation).

import os
import re
import threading

from docx import Document as DocxDoc
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from database.database import upsert_import_chat

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # engine/ → root
DB_PATH      = os.path.join(BASE_DIR, "chroma_db")
INSERT_BATCH = 32

# ── Job registry ─────────────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()


def get_scenario_job(job_id: str) -> dict:
    with _jobs_lock:
        return _jobs.get(job_id, {})


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


# ── Parsing ───────────────────────────────────────────────────────────────────
_CASE_HEADING_RE = re.compile(r'^Tình huống\s+\d+\.\s*(.+)$', re.IGNORECASE)
_MA_DOKHO_RE     = re.compile(r'^Mã:\s*(\S+)\s+Độ khó:\s*(.+)$', re.IGNORECASE)
_TINH_HUONG_RE   = re.compile(r'^Tình huống:\s*(.+)$', re.IGNORECASE)
_CAU_HOI_RE      = re.compile(r'^Câu hỏi:\s*(.+)$', re.IGNORECASE)
_IRAC_RE         = re.compile(r'^[IRAC]\s*[-–—]\s*(Issue|Rule|Application|Conclusion)\s*:\s*(.+)$', re.IGNORECASE)
_TU_KHOA_RE      = re.compile(r'^Từ khóa:\s*(.+)$', re.IGNORECASE)
_CAU_HOI_TD_RE   = re.compile(r'^Câu hỏi tương đương:\s*(.+)$', re.IGNORECASE)


def _new_case(topic: str) -> dict:
    return {
        "case_id": "", "topic": topic, "difficulty": "",
        "scenario": "", "user_question": "",
        "issue_questions": [], "issue": "", "rule": "",
        "application": "", "conclusion": "",
        "legal_basis": [], "keywords": [], "alternative_queries": [],
    }


def parse_scenario_docx(path: str) -> list:
    """Returns a list of case dicts, one per "Tình huống NN." block."""
    doc = DocxDoc(path)
    cases = []
    current = None
    section = None  # which numbered Heading-2 subsection we're currently in

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = p.style.name if p.style else ""

        if style == "Heading 1":
            if current:
                cases.append(current)
            m = _CASE_HEADING_RE.match(text)
            current = _new_case(m.group(1).strip()) if m else None
            section = None
            continue

        if current is None:
            continue

        if style == "Heading 2":
            if text.startswith("1."):
                section = "de_bai"
            elif text.startswith("2."):
                section = "issue_questions"
            elif text.startswith("3."):
                section = "irac"
            elif text.startswith("4."):
                section = "legal_basis"
            elif text.startswith("5."):
                section = "retrieval"
            else:
                section = None
            continue

        m = _MA_DOKHO_RE.match(text)
        if m:
            current["case_id"]    = m.group(1).strip()
            current["difficulty"] = m.group(2).strip()
            continue

        if section == "de_bai":
            m = _TINH_HUONG_RE.match(text)
            if m:
                current["scenario"] = m.group(1).strip()
                continue
            m = _CAU_HOI_RE.match(text)
            if m:
                current["user_question"] = m.group(1).strip()
                continue

        elif section == "issue_questions":
            current["issue_questions"].append(text)

        elif section == "irac":
            m = _IRAC_RE.match(text)
            if m:
                label, val = m.group(1).lower(), m.group(2).strip()
                if label == "issue":
                    current["issue"] = val
                elif label == "rule":
                    current["rule"] = val
                elif label == "application":
                    current["application"] = val
                elif label == "conclusion":
                    current["conclusion"] = val

        elif section == "legal_basis":
            current["legal_basis"].append(text)

        elif section == "retrieval":
            m = _TU_KHOA_RE.match(text)
            if m:
                current["keywords"] = [k.strip() for k in m.group(1).split(";") if k.strip()]
                continue
            m = _CAU_HOI_TD_RE.match(text)
            if m:
                current["alternative_queries"] = [q.strip() for q in m.group(1).split("|") if q.strip()]
                continue

    if current:
        cases.append(current)

    # Drop anything that never got a case_id / scenario — not a real case block
    return [c for c in cases if c["case_id"] and c["scenario"]]


def _build_case_doc(case: dict, source_label: str, importer: str) -> Document:
    from engine.rag_engine import detect_entity_type

    lines = [
        f"Tình huống pháp lý: {case['topic']}",
        f"Mô tả: {case['scenario']}",
        f"Câu hỏi: {case['user_question']}",
    ]
    if case["issue_questions"]:
        lines.append("Câu hỏi dẫn dắt: " + " ".join(case["issue_questions"]))
    if case["issue"]:
        lines.append(f"Vấn đề pháp lý (Issue): {case['issue']}")
    if case["rule"]:
        lines.append(f"Quy tắc pháp lý (Rule): {case['rule']}")
    if case["application"]:
        lines.append(f"Áp dụng (Application): {case['application']}")
    if case["conclusion"]:
        lines.append(f"Kết luận (Conclusion): {case['conclusion']}")
    if case["legal_basis"]:
        lines.append("Căn cứ pháp lý: " + "; ".join(case["legal_basis"]))
    if case["keywords"]:
        lines.append("Từ khóa: " + "; ".join(case["keywords"]))
    if case["alternative_queries"]:
        lines.append("Câu hỏi tương đương: " + " | ".join(case["alternative_queries"]))
    content = "\n".join(lines)

    meta = {
        "case_id":            case["case_id"],
        "topic":              case["topic"],
        "difficulty":         case["difficulty"],
        "doc_type":           "scenario_qa",
        "import_source":      "scenario",
        "nguon_thu_thap":     source_label,
        "legal_basis":        "; ".join(case["legal_basis"]),
        "retrieval_keywords": "; ".join(case["keywords"]),
        "char_count":         len(content),
        "importer":           importer,
    }
    entity_type = detect_entity_type(f"{case['topic']} {content}")
    if entity_type:
        meta["entity_type"] = entity_type

    return Document(page_content=content, metadata=meta)


# ── Main background job ───────────────────────────────────────────────────────
def run_import_scenario(job_id: str, file_path: str, student_id: int,
                        original_filename: str = None, importer: str = "admin1"):
    """Parse a scenario DOCX and add one chunk per case to ChromaDB (no wipe,
    skip case_ids already indexed).

    No manual keyword picker here (unlike Law import) — scenario cases are
    test/enrichment data, not the authoritative source, so they never get the
    primary-keyword scoring buff. Instead every unique phrase from each
    case's own "5. Dữ liệu hỗ trợ truy xuất chatbot" → "Từ khóa:" line is
    auto-tagged as a *secondary* keyword (see the end of the try-block below).
    """
    _set_job(job_id, status="running", message="Đang đọc file DOCX…")
    source_label = os.path.splitext(os.path.basename(original_filename or file_path))[0]

    try:
        cases = parse_scenario_docx(file_path)
        if not cases:
            _set_job(job_id, status="failed", message=(
                "❌ Không tìm thấy tình huống nào trong file. "
                "File cần theo đúng cấu trúc 'Tình huống NN. <chủ đề>' với các mục "
                "1. Đề bài / 2. Câu hỏi dẫn dắt / 3. Đáp án IRAC / 4. Căn cứ pháp lý / "
                "5. Dữ liệu hỗ trợ truy xuất chatbot."
            ))
            return

        _set_job(job_id, message=f"Đã đọc {len(cases)} tình huống — đang kiểm tra trùng lặp…")

        docs = [_build_case_doc(c, source_label, importer) for c in cases]

        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs        = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

        existing         = vs.get(include=["metadatas"])
        existing_case_ids = {m.get("case_id", "").strip() for m in existing["metadatas"] if m.get("case_id")}

        new_docs = [d for d in docs if d.metadata["case_id"] not in existing_case_ids]
        skipped  = len(docs) - len(new_docs)

        if new_docs:
            for i in range(0, len(new_docs), INSERT_BATCH):
                chunk = new_docs[i:i + INSERT_BATCH]
                vs.add_documents(chunk)
                _set_job(job_id, message=f"Indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}…")

        result_msg = (
            f"✅ Hoàn tất! Đã thêm {len(new_docs)} tình huống vào ChromaDB "
            f"(bỏ qua {skipped} tình huống đã tồn tại)."
        )

        # Rebuild the citation-source whitelist from what's actually in chroma_db —
        # scenario chunks carry no so_ky_hieu, but a refresh here keeps the whitelist
        # consistent with the rest of the import pipeline (see rag_engine.refresh_citation_sources).
        from engine.rag_engine import refresh_citation_sources
        refresh_citation_sources()

        # Auto-tag this uploaded file with every unique "Từ khóa:" phrase
        # found across its own cases (see database.database.keyword /
        # source_keyword) — keyed by source_label since that's how
        # list_scenario_sources() groups cases from this import type (via
        # nguon_thu_thap). Secondary only — see run_import_scenario docstring.
        from database.database import get_or_create_keyword, set_source_keywords
        unique_phrases = set()
        for c in cases:
            for phrase in c.get("keywords") or []:
                phrase = phrase.strip()
                if phrase:
                    unique_phrases.add(phrase)
        secondary_ids = [get_or_create_keyword(p) for p in unique_phrases]
        set_source_keywords("scenario", source_label, [], secondary_ids)

        _set_job(job_id, status="done", message=result_msg)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status="failed", message=f"❌ Lỗi: {e}")

    finally:
        try:
            msg = get_scenario_job(job_id).get("message", "Xử lý hoàn tất.")
            upsert_import_chat(student_id, msg)
        except Exception:
            pass
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
