# build_db_from_dataset.py
# Builds ChromaDB from enterprise_law_full_rag_chatbot_dataset_150.xlsx
# Uses KB_Articles sheet — 82 curated legal articles with full metadata.
#
# KB_Articles has richer metadata than build_db.py (HuggingFace):
#   - article_reference   : Điều X
#   - topic               : human-readable topic name
#   - legal_rule_summary  : clean Vietnamese legal summary
#   - retrieval_keywords  : comma-separated keywords for better matching
#   - chapter/article/type: structured legal hierarchy
#   - source_url          : official VBPL source
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python database/build_db_from_dataset.py
#   (put enterprise_law_full_rag_chatbot_dataset_150.xlsx in project root)

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
XLSX_PATH    = os.path.join(BASE_DIR, "enterprise_law_full_rag_chatbot_dataset_150.xlsx")
DB_PATH      = os.path.join(BASE_DIR, "chroma_db")
INSERT_BATCH = 32

SO_KY_HIEU    = "59/2020/QH14"
LOAI_VAN_BAN  = "Luật"
NGUON_THU_THAP = "Luật Doanh nghiệp 2020 - KB_Articles dataset"

# =========================
# STEP 1: LOAD DATASET
# =========================
print(f"Loading: {XLSX_PATH}")

if not os.path.exists(XLSX_PATH):
    print(f"❌ File not found: {XLSX_PATH}")
    print("   Please put the .xlsx file in your project root folder.")
    exit(1)

xl  = pd.ExcelFile(XLSX_PATH)
kb  = xl.parse('KB_Articles')
ds  = xl.parse('Dataset_150')

print(f"KB_Articles: {len(kb)} articles")
print(f"Dataset_150: {len(ds)} Q&A pairs")
print()

# =========================
# STEP 2: PARSE METADATA
# =========================
def parse_chunk_meta(meta_str: str) -> dict:
    """
    Parse 'chapter=I; article=4; type=definition' → dict
    """
    result = {}
    if not isinstance(meta_str, str):
        return result
    for part in meta_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result

# =========================
# STEP 3: BUILD DOCUMENTS FROM KB_ARTICLES
# =========================
print("Building documents from KB_Articles...")
docs = []

for _, row in kb.iterrows():
    article_ref  = str(row.get('article_reference', '')).strip()
    topic        = str(row.get('topic', '')).strip()
    summary      = str(row.get('legal_rule_summary_vi', '')).strip()
    keywords     = str(row.get('retrieval_keywords', '')).strip()
    meta_str     = str(row.get('suggested_chunk_metadata', '')).strip()
    source_url   = str(row.get('source_url', '')).strip()
    note         = str(row.get('note', '')).strip()

    if not summary or summary == 'nan':
        continue

    # Parse structured metadata
    chunk_meta = parse_chunk_meta(meta_str)
    chapter    = chunk_meta.get('chapter', '')
    article_num = chunk_meta.get('article', re.sub(r'[^\d]', '', article_ref))
    doc_type   = chunk_meta.get('type', '')

    # Build rich page_content — combine all fields for better retrieval
    # Format mirrors how students ask questions about the law
    content_parts = [
        f"{article_ref}. {topic}",
        f"Quy tắc pháp lý: {summary}",
    ]
    if keywords and keywords != 'nan':
        content_parts.append(f"Từ khóa: {keywords}")
    if note and note != 'nan':
        content_parts.append(f"Ghi chú: {note}")

    page_content = "\n".join(content_parts)

    docs.append(Document(
        page_content=page_content,
        metadata={
            "so_ky_hieu":      SO_KY_HIEU,
            "loai_van_ban":    LOAI_VAN_BAN,
            "nguon_thu_thap":  NGUON_THU_THAP,
            "article_reference": article_ref,
            "article_number":  article_num,
            "chapter":         chapter,
            "topic":           topic,
            "doc_type":        doc_type,
            "retrieval_keywords": keywords,
            "source_url":      source_url,
            "char_count":      len(page_content),
        }
    ))

print(f"  Built {len(docs)} article documents")

# =========================
# STEP 4: ALSO ADD Q&A PAIRS FROM DATASET_150
# ─────────────────────────────────────────────
# Each Q&A pair includes the ground_truth_rule as content.
# This improves retrieval for scenario-based questions which
# are phrased differently from the legal text.
# =========================
print("Building documents from Dataset_150 Q&A pairs...")
qa_docs = []

for _, row in ds.iterrows():
    q_id       = str(row.get('id', '')).strip()
    q_type     = str(row.get('question_type', '')).strip()
    difficulty = str(row.get('difficulty', '')).strip()
    topic      = str(row.get('topic', '')).strip()
    question   = str(row.get('question_vi', '')).strip()
    answer     = str(row.get('expected_answer_vi', '')).strip()
    article_ref = str(row.get('article_reference', '')).strip()
    rule       = str(row.get('ground_truth_rule', '')).strip()
    keywords   = str(row.get('retrieval_keywords', '')).strip()
    source_url = str(row.get('source_url', '')).strip()

    if not question or question == 'nan':
        continue
    if not answer or answer == 'nan':
        continue

    # Build Q&A content — phrased naturally so retrieval works for
    # students asking similar questions
    content_parts = [
        f"Câu hỏi: {question}",
        f"Trả lời: {answer}",
    ]
    if rule and rule != 'nan' and rule != answer:
        content_parts.append(f"Quy tắc pháp lý: {rule}")
    if keywords and keywords != 'nan':
        content_parts.append(f"Từ khóa: {keywords}")

    page_content = "\n".join(content_parts)

    article_num = re.sub(r'[^\d]', '', article_ref)

    qa_docs.append(Document(
        page_content=page_content,
        metadata={
            "so_ky_hieu":        SO_KY_HIEU,
            "loai_van_ban":      LOAI_VAN_BAN,
            "nguon_thu_thap":    "Dataset_150 Q&A pairs",
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

print(f"  Built {len(qa_docs)} Q&A documents")
all_docs = docs + qa_docs
print(f"\nTotal documents to index: {len(all_docs)}")

# =========================
# STEP 5: CHECK FOR DUPLICATES IN EXISTING DB
# =========================
print("\nLoading embedding model (BAAI/bge-small-en-v1.5)...")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print(f"ChromaDB path: {DB_PATH}")
vs = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

# Get existing so_ky_hieu values
existing     = vs.get(include=["metadatas"])
existing_ids = {m.get("so_ky_hieu","").strip() for m in existing["metadatas"]}

# For dataset, also check by doc_id
existing_doc_ids = {m.get("doc_id","").strip() for m in existing["metadatas"]}

new_docs = []
skipped  = 0
for d in all_docs:
    doc_id = d.metadata.get("doc_id", "")
    ky_hieu = d.metadata.get("so_ky_hieu", "")
    nguon   = d.metadata.get("nguon_thu_thap", "")

    # Skip Q&A docs if already indexed (by doc_id)
    if doc_id and doc_id in existing_doc_ids:
        skipped += 1
        continue

    # Skip KB article docs if 59/2020/QH14 from this dataset already exists
    if not doc_id and ky_hieu in existing_ids and "KB_Articles" in nguon:
        skipped += 1
        continue

    new_docs.append(d)

print(f"  Skipping {skipped} duplicates already in DB")
print(f"  Inserting {len(new_docs)} new documents")

# =========================
# STEP 6: INDEX INTO CHROMADB
# =========================
if not new_docs:
    print("\nNothing new to add — DB already up to date!")
else:
    for i in range(0, len(new_docs), INSERT_BATCH):
        chunk = new_docs[i:i + INSERT_BATCH]
        vs.add_documents(chunk)
        print(f"  Indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}")

print(f"\n✅ DONE!")
print(f"   KB_Articles docs : {len(docs)}")
print(f"   Q&A docs         : {len(qa_docs)}")
print(f"   New inserted      : {len(new_docs)}")
print(f"   Total in DB       : {vs._collection.count()}")
print(f"\nNext: python app.py")