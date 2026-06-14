# build_db.py  --  metadata-only Chroma DB (real schema from dataset card)
#
# 17 metadata columns: id, title, so_ky_hieu, ngay_ban_hanh, loai_van_ban,
# ngay_co_hieu_luc, ngay_het_hieu_luc, nguon_thu_thap, ngay_dang_cong_bao,
# nganh, linh_vuc, co_quan_ban_hanh, chuc_danh, nguoi_ky, pham_vi,
# thong_tin_ap_dung, tinh_trang_hieu_luc
#
# Filters:
#   1. nganh  matches a business-related sector
#   2. tinh_trang_hieu_luc != "Hết hiệu lực toàn bộ"  (skip expired laws)
#
# Citations: so_ky_hieu + nguon_thu_thap (no synthetic URLs -- safer
# against link rot, and is how Vietnamese legal docs are normally cited).

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]    = "False"
os.environ["HF_HUB_TIMEOUT"]      = "60"
os.environ["HF_HUB_MAX_RETRIES"]  = "10"

from datasets import load_dataset
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# CONFIG
# =========================
MAX_DOCS     = 5000
INSERT_BATCH = 64
SKIP_EXPIRED = True

REPO_ID = "th1nhng0/vietnamese-legal-documents"

BUSINESS_SECTORS = (
    "Tài chính",
    "Công Thương",
    "Kế hoạch và Đầu tư",
    "Kế hoạch - Đầu tư",
    "Lao động - Thương binh và Xã hội",
    "Ngân hàng",
    "Công nghiệp",
    "Bảo Hiểm",
    "Thuỷ sản",
    "Bưu chính - Viễn thông",
    "Thương mại",
)

EXPIRED_STATUSES = (
    "Hết hiệu lực toàn bộ",
    "Hết hiệu lực",
)


def matches_business_sector(nganh: str) -> bool:
    if not nganh:
        return False
    n = nganh.lower()
    return any(s.lower() in n for s in BUSINESS_SECTORS)


def is_expired(status: str) -> bool:
    if not status:
        return False
    s = status.lower()
    return any(e.lower() in s for e in EXPIRED_STATUSES)


# Fields concatenated into the embedded text. Order matters -- title first.
TEXT_FIELDS = [
    "title",
    "loai_van_ban",        # Type: Quyết định / Nghị định / Thông tư
    "so_ky_hieu",          # Official number (the durable citation key)
    "nganh",               # Sector
    "linh_vuc",            # Legal field (often null but useful when present)
    "co_quan_ban_hanh",    # Issuing authority
    "chuc_danh",           # Signatory title
    "nguoi_ky",            # Signatory name
    "pham_vi",             # Geographical scope
    "thong_tin_ap_dung",   # Implementation note
    "ngay_ban_hanh",       # Issuance date
    "ngay_co_hieu_luc",    # Effective date
    "nguon_thu_thap",      # Collection source (e.g. Công báo) -- helps citation
    "tinh_trang_hieu_luc", # Effect status
]

VN_LABEL = {
    "title":               "Tiêu đề",
    "loai_van_ban":        "Loại văn bản",
    "so_ky_hieu":          "Số ký hiệu",
    "nganh":               "Ngành",
    "linh_vuc":            "Lĩnh vực",
    "co_quan_ban_hanh":    "Cơ quan ban hành",
    "chuc_danh":           "Chức danh",
    "nguoi_ky":            "Người ký",
    "pham_vi":             "Phạm vi",
    "thong_tin_ap_dung":   "Thông tin áp dụng",
    "ngay_ban_hanh":       "Ngày ban hành",
    "ngay_co_hieu_luc":    "Ngày có hiệu lực",
    "nguon_thu_thap":      "Nguồn thu thập",
    "tinh_trang_hieu_luc": "Tình trạng hiệu lực",
}


def build_text(row: dict) -> str:
    parts = []
    for f in TEXT_FIELDS:
        v = row.get(f)
        if v is None:
            continue
        v = str(v).strip()
        if not v:
            continue
        parts.append(f"{VN_LABEL.get(f, f)}: {v}")
    return "\n".join(parts)


# =========================
# 1. LOAD METADATA
# =========================
print("Loading metadata...")
meta_ds = load_dataset(REPO_ID, "metadata", split="data")
print(f"  metadata rows: {len(meta_ds)}")
print(f"  metadata fields: {list(meta_ds[0].keys())}")


# =========================
# 2. FILTER + BUILD DOCS
# =========================
# Documents with these so_ky_hieu are managed by dedicated scripts.
# Skip here to avoid overwriting with incomplete HuggingFace metadata.
#
# Priority order (run in this sequence):
#   1. build_db_from_dataset.py  → 59/2020/QH14 curated KB + Q&A
#   2. build_db_from_txt.py      → 59/2020/QH14 full OCR (overwrites #1 if run)
#   3. build_db.py (this file)   → all other laws from HuggingFace
#
# To add more PDFs managed locally, add their so_ky_hieu here:
PDF_MANAGED = {
    "59/2020/QH14",   # Luật Doanh nghiệp 2020 — managed by build_db_from_dataset.py / build_db_from_txt.py
}

print("Filtering...")
docs = []
n_business = n_expired = n_pdf_managed = 0

for row in meta_ds:
    if len(docs) >= MAX_DOCS:
        break

    if not matches_business_sector(row.get("nganh") or ""):
        continue
    n_business += 1

    if SKIP_EXPIRED and is_expired(row.get("tinh_trang_hieu_luc") or ""):
        n_expired += 1
        continue

    # Skip docs managed by build_db_from_txt.py (have full OCR text)
    ky_hieu = (row.get("so_ky_hieu") or "").strip()
    if ky_hieu in PDF_MANAGED:
        n_pdf_managed += 1
        continue

    text = build_text(row)
    if len(text) < 30:
        continue

    docs.append(Document(
        page_content=text,
        metadata={
            "id":                int(row.get("id") or 0),
            "title":             row.get("title", "") or "",
            "so_ky_hieu":        ky_hieu,
            "loai_van_ban":      row.get("loai_van_ban", "") or "",
            "nganh":             row.get("nganh", "") or "",
            "linh_vuc":          row.get("linh_vuc", "") or "",
            "co_quan_ban_hanh":  row.get("co_quan_ban_hanh", "") or "",
            "ngay_ban_hanh":     row.get("ngay_ban_hanh", "") or "",
            "nguon_thu_thap":    row.get("nguon_thu_thap", "") or "",
            "tinh_trang":        row.get("tinh_trang_hieu_luc", "") or "",
        },
    ))

print(f"  business rows scanned : {n_business}")
print(f"  expired rows skipped  : {n_expired}")
print(f"  PDF-managed skipped   : {n_pdf_managed} (has full OCR text)")
print(f"  kept documents        : {len(docs)}")


# =========================
# 3. EMBED + WRITE CHROMA
# ─────────────────────────
# Never deletes existing DB.
# Uses so_ky_hieu as unique key to skip duplicates.
# Safe to run multiple times.
# =========================
print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # database/ → root
DB_PATH  = os.path.join(BASE_DIR, "chroma_db")
print(f"ChromaDB path: {DB_PATH}")

# Load existing DB or create new one
vs = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

# Build a composite key (so_ky_hieu + title) to detect duplicates.
# More precise than so_ky_hieu alone — allows partial re-imports
# without skipping documents that weren't indexed yet.
print("Checking existing documents in ChromaDB...")
existing     = vs.get(include=["metadatas"])
existing_keys = set()
for meta in existing["metadatas"]:
    ky_hieu = meta.get("so_ky_hieu", "").strip()
    title   = meta.get("title", "").strip()
    if ky_hieu:
        existing_keys.add(f"{ky_hieu}::{title}")

print(f"  Already in DB  : {len(existing_keys)} unique documents")

# Filter out docs already in DB using composite key
new_docs = []
skipped  = 0
for d in docs:
    ky_hieu = d.metadata.get("so_ky_hieu", "").strip()
    title   = d.metadata.get("title", "").strip()
    key     = f"{ky_hieu}::{title}"
    if key in existing_keys:
        skipped += 1
    else:
        new_docs.append(d)

print(f"  Skipping       : {skipped} duplicates")
print(f"  New to insert  : {len(new_docs)} documents")

if not new_docs:
    print("\nNothing new to add — DB is already up to date!")
else:
    print(f"\nInserting {len(new_docs)} new documents...")
    for i in range(0, len(new_docs), INSERT_BATCH):
        chunk = new_docs[i:i + INSERT_BATCH]
        vs.add_documents(chunk)
        print(f"  indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}")
    print(f"\nDONE: {len(new_docs)} new documents added.")

print(f"Total documents in DB: {vs._collection.count()}")