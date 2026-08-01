# import_dataset_engine.py
# Background worker: import an Excel dataset file's CURATED KNOWLEDGE-BASE
# content (KB_Articles / KB_Articles_Updated / Legal_Update_2025) into
# ChromaDB, and register the file as an evaluation fixture.
#
# As of 2026-07-28 the Dataset_*/Demo_* Q&A sheets are PERMANENTLY excluded
# from ChromaDB. A direct inspection of the live vectorstore confirmed a real
# data-leakage risk: content from those sheets was retrievable during real
# question answering — the exact question/answer pair used to test the
# system could be retrieved back as "context" for that same question.
#
# KB_Articles/KB_Articles_Updated/Legal_Update_2025, by contrast, are
# hand-curated legal-rule summaries — not question/answer test pairs — so
# indexing them carries no such leakage risk, and removing them entirely
# (an earlier, overly broad version of this fix) measurably hurt real answer
# quality on several questions that depended on them. They are re-embedded
# here; only the Q&A test sheets stay off ChromaDB, tracked in chat.db
# (see database.database.dataset_file) purely for engine.evaluate_engine's
# Quick/Full Evaluation to read directly from disk.

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
    available for RAG evaluation (see evaluate_engine.list_available_datasets).
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


# so_ky_hieu used to be hardcoded to the module-wide SO_KY_HIEU constant for
# every row of KB_Articles_Updated/Legal_Update_2025, on the assumption the
# whole sheet was about Luật Doanh nghiệp 2020 — but curated rows added after
# the 168/2025/NĐ-CP decree (or referencing 67/VBHN-VPQH/76/2025/QH15) say so
# explicitly in their own article_reference/legal_source text (e.g. "Điều 52
# Nghị định 168/2025/NĐ-CP", "67/VBHN-VPQH 2025"), which the import silently
# ignored. Found live 2026-07-30 (traced from ELU161/ELU170 retrieval bugs):
# 18 chunks across both sheets were citing the wrong law/decree entirely.
def _derive_so_ky_hieu(article_ref: str) -> str:
    ref = (article_ref or "").lower()
    if "168/2025" in ref or "nghị định 168" in ref:
        return "168/2025/NĐ-CP"
    if "67/vbhn-vpqh" in ref:
        return "67/VBHN-VPQH"
    if "76/2025/qh15" in ref:
        return "76/2025/QH15"
    return SO_KY_HIEU


# Range-style rows (curated after the fact to cover several Điều at once, e.g.
# "Điều 112-115 Nghị định 168/2025/NĐ-CP") store their range under an
# "articles=" (plural) key in suggested_chunk_metadata, not "article=" —
# looking up only the singular key silently missed these, falling through to
# stripping every non-digit from article_reference instead, which concatenates
# the article numbers with the decree/year digits into garbage like
# "1121151682025" (found live 2026-07-30 alongside the so_ky_hieu bug above).
# Whole-document references with no "Điều" at all (e.g. "Nghị định
# 168/2025/NĐ-CP", "Luật 76/2025/QH15") correctly get "" here — they don't
# name one specific article.
def _derive_article_number(article_ref: str, chunk_meta: dict) -> str:
    single = chunk_meta.get("article")
    if single:
        return single
    ranged = chunk_meta.get("articles")
    if ranged:
        m = re.search(r"\d+", ranged)
        if m:
            return m.group(0)
    m = re.search(r"Điều\s+(\d+)", article_ref or "", re.IGNORECASE)
    return m.group(1) if m else ""


# ── Sheet processors (KB / curated content only — never Dataset_*/Demo_*) ─────
def _build_kb_docs(sheet_df: pd.DataFrame, sheet_name: str, source_file: str, importer: str) -> list:
    """Process KB_Articles or KB_Articles_Updated sheet — curated legal-rule
    summaries, not test Q&A pairs, so safe to index."""
    from engine.rag_engine import detect_entity_type

    docs = []

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
        # Explicit so_ky_hieu column (2026-07-30, alongside source_url — a
        # different field, the citation URL, not the document identifier) is
        # authoritative when the sheet provides one — falls back to text-
        # derivation only for older-format sheets uploaded before this column
        # existed, so those don't suddenly stop importing.
        so_ky_hieu  = _safe(row.get('so_ky_hieu', '')) or _derive_so_ky_hieu(article_ref)
        article_num = _derive_article_number(article_ref, chunk_meta)
        doc_type    = chunk_meta.get('type', '')
        # nguon_thu_thap ("collection source") is the uploaded file itself —
        # a synthesized "{so_ky_hieu} - {sheet_name} dataset" string used to
        # duplicate so_ky_hieu and couldn't be traced back to which upload
        # produced it.
        nguon       = source_file

        parts = [f"{article_ref}. {topic}", f"Quy tắc pháp lý: {summary}"]
        if keywords:
            parts.append(f"Từ khóa: {keywords}")
        if note:
            parts.append(f"Ghi chú: {note}")
        content = "\n".join(parts)

        meta = {
            "so_ky_hieu":        so_ky_hieu,
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
            "importer":          importer,
        }
        entity_type = detect_entity_type(f"{topic} {content}")
        if entity_type:
            meta["entity_type"] = entity_type

        docs.append(Document(page_content=content, metadata=meta))
    return docs


def _build_update_docs(sheet_df: pd.DataFrame, source_file: str, importer: str) -> list:
    """Process Legal_Update_2025 sheet — curated summaries of 2025 legal
    changes, not test Q&A pairs, so safe to index.
    Actual columns: update_id, date/effective, legal_source,
                    key_change_vi, impact_on_dataset, implemented_in_sheet,
                    source_url, notes, so_ky_hieu (optional, 2026-07-30 —
                    explicit document identifier; falls back to parsing
                    legal_source's text if omitted, see _derive_so_ky_hieu)
    """
    from engine.rag_engine import detect_entity_type

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
        so_ky_hieu  = _safe(row.get('so_ky_hieu', '')) or _derive_so_ky_hieu(article_ref)
        article_num = _derive_article_number(article_ref, {})

        meta = {
            "so_ky_hieu":        so_ky_hieu,
            "loai_van_ban":      LOAI_VAN_BAN,
            "nguon_thu_thap":    source_file,
            "article_reference": article_ref,
            "article_number":    article_num,
            "topic":             topic,
            "doc_type":          "legal_update_2025",
            "char_count":        len(content),
            "import_source":     "dataset",
            "source_file":       source_file,
            "importer":          importer,
        }
        entity_type = detect_entity_type(f"{topic} {content}")
        if entity_type:
            meta["entity_type"] = entity_type

        docs.append(Document(page_content=content, metadata=meta))
    return docs


# ── Main background task ──────────────────────────────────────────────────────
def run_import_dataset(job_id: str, file_path: str, original_filename: str = None, importer: str = "admin1"):
    """
    Imports KB_Articles(_Updated)/Legal_Update_2025 content into ChromaDB
    (curated reference material — safe, see module docstring). Dataset_*/
    Demo_* sheets are NEVER embedded — the file is saved to Dataset/ and
    registered in the dataset_file tracking table so Quick/Full Evaluation
    can read them straight from disk instead.
    """
    _set(job_id, status="running", message="Đang đọc file dataset…")

    try:
        with pd.ExcelFile(file_path) as xl:
            sheets = xl.sheet_names
            kb_sheet_df, kb_sheet_name = None, None
            if 'KB_Articles_Updated' in sheets:
                kb_sheet_df, kb_sheet_name = xl.parse('KB_Articles_Updated'), 'KB_Articles_Updated'
            elif 'KB_Articles' in sheets:
                kb_sheet_df, kb_sheet_name = xl.parse('KB_Articles'), 'KB_Articles'
            update_sheet_df = xl.parse('Legal_Update_2025') if 'Legal_Update_2025' in sheets else None

        demo_sheets    = [s for s in sheets if s.startswith('Demo_')]
        dataset_sheets = [s for s in sheets if s.startswith('Dataset_')]

        if kb_sheet_df is None and update_sheet_df is None and not demo_sheets and not dataset_sheets:
            _set(job_id, status="failed",
                 message="❌ Không tìm thấy sheet hợp lệ trong file. Cần ít nhất một trong: "
                         "KB_Articles, KB_Articles_Updated, Legal_Update_2025, Demo_*, Dataset_*")
            return

        # Persist first so every metadata tag and the tracking-table row agree
        # on the exact on-disk filename (handles the rare name-collision case
        # where _persist_uploaded_dataset appends a timestamp suffix).
        saved_name = _persist_uploaded_dataset(file_path, original_filename)
        if not saved_name:
            _set(job_id, status="failed", message="❌ Lỗi khi lưu file vào thư mục Dataset/.")
            return

        report = []
        kb_docs = []
        if kb_sheet_df is not None:
            _set(job_id, message=f"Xử lý {kb_sheet_name}…")
            d = _build_kb_docs(kb_sheet_df, kb_sheet_name, saved_name, importer)
            kb_docs.extend(d)
            report.append(f"{kb_sheet_name}: {len(d)} tài liệu (đã nạp ChromaDB)")
        if update_sheet_df is not None:
            _set(job_id, message="Xử lý Legal_Update_2025…")
            d = _build_update_docs(update_sheet_df, saved_name, importer)
            kb_docs.extend(d)
            report.append(f"Legal_Update_2025: {len(d)} tài liệu (đã nạp ChromaDB)")
        if demo_sheets:
            report.append(f"Demo (chỉ dùng đánh giá, KHÔNG nạp ChromaDB): {', '.join(demo_sheets)}")
        if dataset_sheets:
            report.append(f"Dataset (chỉ dùng đánh giá, KHÔNG nạp ChromaDB): {', '.join(dataset_sheets)}")

        new_count, skipped = 0, 0
        if kb_docs:
            _set(job_id, message=f"Tổng {len(kb_docs)} tài liệu KB — đang kiểm tra trùng lặp…")

            embedding = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            vs = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

            existing          = vs.get(include=["metadatas", "documents"])
            existing_ids      = {m.get("so_ky_hieu", "").strip() for m in existing["metadatas"]}
            existing_doc_ids  = {m.get("doc_id", "").strip() for m in existing["metadatas"]}
            existing_contents = set(existing["documents"])

            new_docs = []
            for d in kb_docs:
                doc_id  = d.metadata.get("doc_id", "")
                ky_hieu = d.metadata.get("so_ky_hieu", "")
                nguon   = d.metadata.get("nguon_thu_thap", "")

                if doc_id and doc_id in existing_doc_ids:
                    skipped += 1
                    continue
                if not doc_id and ky_hieu in existing_ids and "KB_Articles" in nguon:
                    skipped += 1
                    continue
                # Rows with neither a doc_id nor "KB_Articles" nguon_thu_thap
                # (e.g. Legal_Update_2025) have no natural unique key — fall
                # back to exact content match so re-importing the same file
                # doesn't duplicate them (found live 2026-07-28).
                if not doc_id and d.page_content in existing_contents:
                    skipped += 1
                    continue
                new_docs.append(d)

            new_count = len(new_docs)
            if new_docs:
                for i in range(0, len(new_docs), INSERT_BATCH):
                    vs.add_documents(new_docs[i:i + INSERT_BATCH])
                    _set(job_id, message=f"Indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}…")

            from engine.rag_engine import refresh_citation_sources
            refresh_citation_sources()

            # Auto-tag this file with every unique retrieval_keywords phrase
            # found across its KB rows, as *secondary* keywords (see
            # database.database.keyword/source_keyword) — KB content supports
            # the primary legal sources rather than replacing them, so it
            # never gets the primary-keyword buff (that's reserved for Law
            # imports, see import_law_engine.py).
            from database.database import get_or_create_keyword, set_source_keywords
            unique_phrases = set()
            for d in kb_docs:
                for phrase in (d.metadata.get("retrieval_keywords") or "").split(";"):
                    phrase = phrase.strip()
                    if phrase:
                        unique_phrases.add(phrase)
            if unique_phrases:
                secondary_ids = [get_or_create_keyword(p) for p in unique_phrases]
                set_source_keywords("dataset", saved_name, [], secondary_ids)

        from database.database import register_dataset_file
        register_dataset_file(saved_name, importer)

        result_msg = "✅ Hoàn tất!\n" + "\n".join(report)
        if kb_docs:
            result_msg += f"\nTài liệu KB mới thêm: {new_count}\nBỏ qua (trùng): {skipped}"
        result_msg += "\nSẵn sàng dùng cho Quick/Full Evaluation."

        _set(job_id, status="done", message=result_msg, saved_dataset_file=saved_name)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set(job_id, status="failed", message=f"❌ Lỗi: {e}")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
