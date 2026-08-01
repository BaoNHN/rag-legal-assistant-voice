# build_db_from_dataset_updated.py
# Builds ChromaDB from enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx
#
# Sheet mapping (skip old names with same base — use _Updated suffix):
#   KB_Articles_Updated  -> curated legal articles (replaces KB_Articles)
#   Legal_Update_2025    -> 2025 law amendments + impact notes
#   Dataset_200 (or any Dataset_* sheet) -> Q&A pairs
#   Summary_Updated / Sources_Updated    -> skipped (reference only)
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python database/build_db_from_dataset_updated.py

import os
import re
import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIG
# =========================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH    = os.path.join(BASE_DIR, "enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx")
DB_PATH      = os.path.join(BASE_DIR, "chroma_db")
INSERT_BATCH = 32

SO_KY_HIEU   = "59/2020/QH14"
LOAI_VAN_BAN = "Luật"

# =========================
# STEP 1: LOAD DATASET
# =========================
print(f"Loading: {XLSX_PATH}")

if not os.path.exists(XLSX_PATH):
    print(f"❌ File not found: {XLSX_PATH}")
    print("   Place enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx in the project root.")
    exit(1)

xl = pd.ExcelFile(XLSX_PATH)
print(f"Sheets found: {xl.sheet_names}")

# =========================
# STEP 2: HELPERS
# =========================
def parse_chunk_meta(meta_str: str) -> dict:
    result = {}
    if not isinstance(meta_str, str):
        return result
    for part in meta_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result


def safe(val) -> str:
    s = str(val).strip()
    return "" if s == "nan" else s


# =========================
# STEP 3: KB_Articles_Updated
# =========================
all_docs = []

if 'KB_Articles_Updated' in xl.sheet_names:
    kb = xl.parse('KB_Articles_Updated')
    print(f"\nKB_Articles_Updated: {len(kb)} rows")

    NGUON = "Luật Doanh nghiệp 2020 - KB_Articles_Updated dataset"
    built = 0

    for _, row in kb.iterrows():
        article_ref = safe(row.get('article_reference', ''))
        topic       = safe(row.get('topic', ''))
        summary     = safe(row.get('legal_rule_summary_vi', ''))
        keywords    = safe(row.get('retrieval_keywords', ''))
        meta_str    = safe(row.get('suggested_chunk_metadata', ''))
        source_url  = safe(row.get('source_url', ''))
        note        = safe(row.get('note', ''))

        if not summary:
            continue

        chunk_meta  = parse_chunk_meta(meta_str)
        chapter     = chunk_meta.get('chapter', '')
        article_num = chunk_meta.get('article', re.sub(r'[^\d]', '', article_ref))
        doc_type    = chunk_meta.get('type', '')

        content_parts = [f"{article_ref}. {topic}", f"Quy tắc pháp lý: {summary}"]
        if keywords:
            content_parts.append(f"Từ khóa: {keywords}")
        if note:
            content_parts.append(f"Ghi chú: {note}")
        page_content = "\n".join(content_parts)

        all_docs.append(Document(
            page_content=page_content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    NGUON,
                "article_reference": article_ref,
                "article_number":    article_num,
                "chapter":           chapter,
                "topic":             topic,
                "doc_type":          doc_type,
                "retrieval_keywords":keywords,
                "source_url":        source_url,
                "char_count":        len(page_content),
            }
        ))
        built += 1

    print(f"  -> Built {built} KB_Articles_Updated documents")
else:
    print("[!]  Sheet 'KB_Articles_Updated' not found — skipped")

# =========================
# STEP 4: Legal_Update_2025
# =========================
if 'Legal_Update_2025' in xl.sheet_names:
    lu = xl.parse('Legal_Update_2025')
    print(f"\nLegal_Update_2025: {len(lu)} rows")

    update_built = 0
    for _, row in lu.iterrows():
        # Actual columns: update_id, date/effective, legal_source,
        #                 key_change_vi, impact_on_dataset, implemented_in_sheet,
        #                 source_url, notes
        article_ref    = safe(row.get('legal_source',
                            row.get('article_reference', row.get('article', ''))))
        topic          = safe(row.get('key_change_vi',
                            row.get('topic', row.get('update_topic', ''))))
        impact         = safe(row.get('impact_on_dataset',
                            row.get('impact', row.get('update_impact', ''))))
        effective_date = safe(row.get('date/effective',
                            row.get('effective_date', row.get('date', ''))))
        source_url     = safe(row.get('source_url', ''))
        note           = safe(row.get('notes', row.get('note', '')))

        # Need at least key_change or impact
        if not topic and not impact:
            continue

        content_parts = []
        if article_ref:
            content_parts.append(f"Nguồn pháp lý: {article_ref}")
        if topic:
            content_parts.append(f"Thay đổi pháp lý 2025: {topic}")
        if impact:
            content_parts.append(f"Tác động: {impact}")
        if effective_date:
            content_parts.append(f"Hiệu lực: {effective_date}")
        if note:
            content_parts.append(f"Ghi chú: {note}")

        if not content_parts:
            continue

        page_content = "\n".join(content_parts)
        article_num  = re.sub(r'[^\d]', '', article_ref) if article_ref else ''

        all_docs.append(Document(
            page_content=page_content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    "Legal_Update_2025",
                "article_reference": article_ref,
                "article_number":    article_num,
                "topic":             topic,
                "doc_type":          "legal_update_2025",
                "char_count":        len(page_content),
            }
        ))
        update_built += 1

    print(f"  -> Built {update_built} Legal_Update_2025 documents")
else:
    print("[!]  Sheet 'Legal_Update_2025' not found — skipped")

# =========================
# STEP 5: Q&A Dataset sheet
# =========================
# Accept Dataset_200, Dataset_150, or any Dataset_* sheet
qa_sheet = next(
    (s for s in ['Dataset_200', 'Dataset_150'] if s in xl.sheet_names),
    next((s for s in xl.sheet_names if s.startswith('Dataset_')), None)
)

if qa_sheet:
    ds = xl.parse(qa_sheet)
    print(f"\n{qa_sheet}: {len(ds)} rows")
    qa_built = 0

    for _, row in ds.iterrows():
        q_id        = safe(row.get('id', ''))
        q_type      = safe(row.get('question_type', ''))
        difficulty  = safe(row.get('difficulty', ''))
        topic       = safe(row.get('topic', ''))
        question    = safe(row.get('question_vi', ''))
        answer      = safe(row.get('expected_answer_vi', ''))
        article_ref = safe(row.get('article_reference', ''))
        rule        = safe(row.get('ground_truth_rule', ''))
        keywords    = safe(row.get('retrieval_keywords', ''))
        source_url  = safe(row.get('source_url', ''))

        if not question or not answer:
            continue

        content_parts = [f"Câu hỏi: {question}", f"Trả lời: {answer}"]
        if rule and rule != answer:
            content_parts.append(f"Quy tắc pháp lý: {rule}")
        if keywords:
            content_parts.append(f"Từ khóa: {keywords}")
        page_content = "\n".join(content_parts)
        article_num  = re.sub(r'[^\d]', '', article_ref)

        all_docs.append(Document(
            page_content=page_content,
            metadata={
                "so_ky_hieu":        SO_KY_HIEU,
                "loai_van_ban":      LOAI_VAN_BAN,
                "nguon_thu_thap":    f"{qa_sheet} Q&A pairs",
                "doc_id":            q_id,
                "question_type":     q_type,
                "difficulty":        difficulty,
                "topic":             topic,
                "article_reference": article_ref,
                "article_number":    article_num,
                "source_url":        source_url,
                "char_count":        len(page_content),
            }
        ))
        qa_built += 1

    print(f"  -> Built {qa_built} Q&A documents")
else:
    print("[!]  No Dataset_* sheet found — Q&A pairs skipped")

print(f"\nTotal documents to index: {len(all_docs)}")

# =========================
# STEP 6: DEDUP + INDEX
# =========================
print("\nLoading embedding model (BAAI/bge-m3)…")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vs = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

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

print(f"  Skipping {skipped} duplicates")
print(f"  Inserting {len(new_docs)} new documents")

if not new_docs:
    print("\nNothing new to add — DB already up to date!")
else:
    for i in range(0, len(new_docs), INSERT_BATCH):
        chunk = new_docs[i:i + INSERT_BATCH]
        vs.add_documents(chunk)
        print(f"  Indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}")

print(f"\n[DONE]")
print(f"   New inserted  : {len(new_docs)}")
print(f"   Total in DB   : {vs._collection.count()}")
print(f"\nNext: python app.py")
