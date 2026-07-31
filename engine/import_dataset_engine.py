# import_dataset_engine.py
# Background worker: import an Excel dataset file into ChromaDB.
# Auto-detects sheet format (150 vs 200-updated) from sheet names.

import os
import re
import shutil
import threading
from datetime import datetime
import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH      = os.path.join(BASE_DIR, "chroma_db")
DATASET_DIR  = os.path.join(BASE_DIR, "Dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
INSERT_BATCH = 32

SO_KY_HIEU   = "59/2020/QH14"
LOAI_VAN_BAN = "Luật"

# ── Job registry ─────────────────────────────────────────────────────────────
_jobs: dict = {}
_lock = threading.Lock()


def get_dataset_job(job_id: str) -> dict:
    with _lock:
        return _jobs.get(job_id, {})


def _set(job_id: str, **kwargs):
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(val) -> str:
    s = str(val).strip()
    return "" if s == "nan" else s


def _persist_uploaded_dataset(tmp_path: str, original_filename: str = None) -> str:
    """
    Moves the uploaded dataset .xlsx into DATASET_DIR (Dataset/) so it becomes
    available for RAG evaluation (see evaluate_engine.list_available_datasets)
    instead of being discarded once it's indexed into ChromaDB.
    Never overwrites an existing file — appends a timestamp on name clash.
    Returns the saved filename, or None if the move failed.
    """
    base_name = os.path.basename(original_filename) if original_filename else "imported_dataset.xlsx"
    if not base_name.lower().endswith(".xlsx"):
        base_name += ".xlsx"
    stem, ext = os.path.splitext(base_name)

    dest_name = base_name
    dest_path = os.path.join(DATASET_DIR, dest_name)
    if os.path.exists(dest_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_name = f"{stem}_{ts}{ext}"
        dest_path = os.path.join(DATASET_DIR, dest_name)

    try:
        shutil.move(tmp_path, dest_path)
        return dest_name
    except Exception:
        return None


def _parse_meta(meta_str: str) -> dict:
    result = {}
    if not isinstance(meta_str, str):
        return result
    for part in meta_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result


# ── Sheet processors ──────────────────────────────────────────────────────────
def _build_kb_docs(sheet_df: pd.DataFrame, sheet_name: str, source_file: str) -> list:
    """Process KB_Articles or KB_Articles_Updated sheet."""
    docs = []
    nguon = f"Luật Doanh nghiệp 2020 - {sheet_name} dataset"

    for _, row in sheet_df.iterrows():
        article_ref = _safe(row.get('article_reference', ''))
        topic       = _safe(row.get('topic', ''))
        summary     = _safe(row.get('legal_rule_summary_vi', ''))
        keywords    = _safe(row.get('retrieval_keywords', ''))
        meta_str    = _safe(row.get('suggested_chunk_metadata', ''))
        source_url  = _safe(row.get('source_url', ''))
        note        = _safe(row.get('note', ''))

        if not summary:
            continue

        chunk_meta  = _parse_meta(meta_str)
        chapter     = chunk_meta.get('chapter', '')
        article_num = chunk_meta.get('article', re.sub(r'[^\d]', '', article_ref))
        doc_type    = chunk_meta.get('type', '')

        parts = [f"{article_ref}. {topic}", f"Quy tắc pháp lý: {summary}"]
        if keywords:
            parts.append(f"Từ khóa: {keywords}")
        if note:
            parts.append(f"Ghi chú: {note}")
        content = "\n".join(parts)

        docs.append(Document(
            page_content=content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    nguon,
                "article_reference": article_ref,
                "article_number":    article_num,
                "chapter":           chapter,
                "topic":             topic,
                "doc_type":          doc_type,
                "retrieval_keywords":keywords,
                "source_url":        source_url,
                "char_count":        len(content),
                "import_source":     "dataset",
                "source_file":       source_file,
            }
        ))
    return docs


def _build_update_docs(sheet_df: pd.DataFrame, source_file: str) -> list:
    """Process Legal_Update_2025 sheet.
    Actual columns: update_id, date/effective, legal_source,
                    key_change_vi, impact_on_dataset, implemented_in_sheet,
                    source_url, notes
    """
    docs = []
    for _, row in sheet_df.iterrows():
        article_ref    = _safe(row.get('legal_source',
                            row.get('article_reference', row.get('article', ''))))
        topic          = _safe(row.get('key_change_vi',
                            row.get('topic', row.get('update_topic', ''))))
        impact         = _safe(row.get('impact_on_dataset',
                            row.get('impact', row.get('update_impact', ''))))
        effective_date = _safe(row.get('date/effective',
                            row.get('effective_date', row.get('date', ''))))
        source_url     = _safe(row.get('source_url', ''))
        note           = _safe(row.get('notes', row.get('note', '')))

        if not topic and not impact:
            continue

        parts = []
        if article_ref:
            parts.append(f"Nguon phap ly: {article_ref}")
        if topic:
            parts.append(f"Thay doi phap ly 2025: {topic}")
        if impact:
            parts.append(f"Tac dong: {impact}")
        if effective_date:
            parts.append(f"Hieu luc: {effective_date}")
        if note:
            parts.append(f"Ghi chu: {note}")

        if not parts:
            continue

        content     = "\n".join(parts)
        article_num = re.sub(r'[^\d]', '', article_ref) if article_ref else ''

        docs.append(Document(
            page_content=content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    "Legal_Update_2025",
                "article_reference": article_ref,
                "article_number":    article_num,
                "topic":             topic,
                "doc_type":          "legal_update_2025",
                "char_count":        len(content),
                "import_source":     "dataset",
                "source_file":       source_file,
            }
        ))
    return docs


def _build_qa_docs(sheet_df: pd.DataFrame, sheet_name: str, source_file: str) -> list:
    """Process any Dataset_* Q&A sheet."""
    docs = []
    for _, row in sheet_df.iterrows():
        q_id        = _safe(row.get('id', ''))
        q_type      = _safe(row.get('question_type', ''))
        difficulty  = _safe(row.get('difficulty', ''))
        topic       = _safe(row.get('topic', ''))
        question    = _safe(row.get('question_vi', ''))
        answer      = _safe(row.get('expected_answer_vi', ''))
        article_ref = _safe(row.get('article_reference', ''))
        rule        = _safe(row.get('ground_truth_rule', ''))
        keywords    = _safe(row.get('retrieval_keywords', ''))
        source_url  = _safe(row.get('source_url', ''))

        if not question or not answer:
            continue

        parts = [f"Câu hỏi: {question}", f"Trả lời: {answer}"]
        if rule and rule != answer:
            parts.append(f"Quy tắc pháp lý: {rule}")
        if keywords:
            parts.append(f"Từ khóa: {keywords}")
        content     = "\n".join(parts)
        article_num = re.sub(r'[^\d]', '', article_ref)

        docs.append(Document(
            page_content=content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    f"{sheet_name} Q&A pairs",
                "doc_id":            q_id,
                "question_type":     q_type,
                "difficulty":        difficulty,
                "topic":             topic,
                "article_reference": article_ref,
                "article_number":    article_num,
                "source_url":        source_url,
                "char_count":        len(content),
                "import_source":     "dataset",
                "source_file":       source_file,
            }
        ))
    return docs


# ── Main background task ──────────────────────────────────────────────────────
def run_import_dataset(job_id: str, file_path: str, original_filename: str = None):
    """
    Import a dataset Excel file into ChromaDB.
    Auto-detects 150 (KB_Articles) vs 200-updated (KB_Articles_Updated) format.
    original_filename: the name the user uploaded, used to persist the file
    into BASE_DIR afterwards (see _persist_uploaded_dataset) so it becomes
    selectable in RAG evaluation.
    """
    _set(job_id, status="running", message="Đang đọc file dataset…")
    source_file = os.path.basename(original_filename) if original_filename else os.path.basename(file_path)

    try:
        xl     = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        _set(job_id, message=f"Phát hiện {len(sheets)} sheets: {', '.join(sheets[:6])}…")

        all_docs = []
        report   = []

        # ── KB_Articles_Updated (200-updated format) takes priority over KB_Articles
        if 'KB_Articles_Updated' in sheets:
            _set(job_id, message="Xử lý KB_Articles_Updated…")
            d = _build_kb_docs(xl.parse('KB_Articles_Updated'), 'KB_Articles_Updated', source_file)
            all_docs.extend(d)
            report.append(f"KB_Articles_Updated: {len(d)} tài liệu")
        elif 'KB_Articles' in sheets:
            _set(job_id, message="Xử lý KB_Articles…")
            d = _build_kb_docs(xl.parse('KB_Articles'), 'KB_Articles', source_file)
            all_docs.extend(d)
            report.append(f"KB_Articles: {len(d)} tài liệu")

        # ── Legal_Update_2025 (new in 200-updated)
        if 'Legal_Update_2025' in sheets:
            _set(job_id, message="Xử lý Legal_Update_2025…")
            d = _build_update_docs(xl.parse('Legal_Update_2025'), source_file)
            all_docs.extend(d)
            report.append(f"Legal_Update_2025: {len(d)} tài liệu")

        # ── Q&A Dataset sheet (Dataset_200 > Dataset_150 > any Dataset_*)
        qa_sheet = next(
            (s for s in ['Dataset_200', 'Dataset_150'] if s in sheets),
            next((s for s in sheets if s.startswith('Dataset_')), None)
        )
        if qa_sheet:
            _set(job_id, message=f"Xử lý {qa_sheet}…")
            d = _build_qa_docs(xl.parse(qa_sheet), qa_sheet, source_file)
            all_docs.extend(d)
            report.append(f"{qa_sheet}: {len(d)} cặp Q&A")

        if not all_docs:
            _set(job_id, status="failed",
                 message="❌ Không tìm thấy sheet hợp lệ trong file. "
                         "Cần ít nhất một trong: KB_Articles, KB_Articles_Updated, Dataset_*")
            return

        _set(job_id, message=f"Tổng {len(all_docs)} tài liệu — đang kiểm tra trùng lặp…")

        # ── Dedup + index
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vs        = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

        existing         = vs.get(include=["metadatas"])
        existing_ids     = {m.get("so_ky_hieu", "").strip() for m in existing["metadatas"]}
        existing_doc_ids = {m.get("doc_id", "").strip() for m in existing["metadatas"]}

        new_docs = []
        skipped  = 0
        for d in all_docs:
            doc_id  = d.metadata.get("doc_id", "")
            ky_hieu = d.metadata.get("so_ky_hieu", "")
            nguon   = d.metadata.get("nguon_thu_thap", "")

            if doc_id and doc_id in existing_doc_ids:
                skipped += 1
                continue
            if not doc_id and ky_hieu in existing_ids and "KB_Articles" in nguon:
                skipped += 1
                continue
            new_docs.append(d)

        _set(job_id, message=f"Bỏ qua {skipped} trùng lặp — thêm {len(new_docs)} tài liệu mới…")

        if new_docs:
            for i in range(0, len(new_docs), INSERT_BATCH):
                chunk = new_docs[i:i + INSERT_BATCH]
                vs.add_documents(chunk)
                done = min(i + INSERT_BATCH, len(new_docs))
                _set(job_id, message=f"Indexed {done}/{len(new_docs)}…")

        total_in_db = vs._collection.count()
        summary = "\n".join(report)
        result_msg = (
            f"✅ Hoàn tất!\n{summary}\n"
            f"Tài liệu mới thêm: {len(new_docs)}\n"
            f"Bỏ qua (trùng): {skipped}\n"
            f"Tổng trong DB: {total_in_db}"
        )

        # Refresh the citation-source whitelist so new so_ky_hieu values become
        # citable immediately (see engine.rag_engine.refresh_citation_sources).
        from engine.rag_engine import refresh_citation_sources
        refresh_citation_sources()

        _set(job_id, status="done", message=result_msg)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="failed", message=f"❌ Lỗi: {e}")

    finally:
        if os.path.exists(file_path):
            saved_name = _persist_uploaded_dataset(file_path, original_filename)
            if saved_name:
                _set(job_id, saved_dataset_file=saved_name)
            else:
                # Persist failed — fall back to removing the temp file so uploads_tmp doesn't accumulate.
                try:
                    os.remove(file_path)
                except Exception:
                    pass
