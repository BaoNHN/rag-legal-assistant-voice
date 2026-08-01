# rebuild_chroma_bge_m3.py
# One-time migration: BAAI/bge-small-en-v1.5 (English-tuned) → BAAI/bge-m3
# (multilingual) embedding, run 2026-07-25.
#
# Rebuilds chroma_db entirely from ITS OWN current content — not from the
# original PDFs/xlsx/docx — because every chunk already holds the exact text
# that was indexed; re-deriving from source files would risk drift (OCR
# re-run differences, renamed uploads) for zero benefit. The only content
# change made here is sub-splitting law articles over 3000 chars (see
# engine.import_law_engine._split_long_segment) — dataset/scenario chunks
# and normal-length law chunks pass through with page_content untouched,
# only re-embedded.
#
# USAGE (two steps, two separate processes — see note below):
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python database/_export_old_chunks.py     # step 1: dump current DB to JSON
#   python database/rebuild_chroma_bge_m3.py  # step 2: delete + rebuild from that JSON
#
# Split into two processes deliberately: chromadb caches its persistent
# client per-process, so opening chroma_db to READ it and then shutil.rmtree
# -ing that same directory IN THE SAME PROCESS fails on Windows with
# "PermissionError: chroma.sqlite3 is being used by another process" — the
# read connection is never released until the process exits.
#
# SAFETY: back up chroma_db/ before running this — it deletes and rebuilds
# the collection in place. (Already done once as chroma_db_backup_20260725_pre_bge_m3/
# for the 2026-07-25 migration; re-run that backup step for any future rerun.)

import json
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)

# Running "python database\rebuild_chroma_bge_m3.py" makes Python
# auto-prepend _SCRIPT_DIR (.../database) to sys.path. Since that folder
# itself contains database.py, that entry makes "import database" resolve
# to the single file database/database.py instead of the database/ package
# rooted at BASE_DIR — breaking "from database.database import ..." inside
# engine.import_law_engine with "'database' is not a package". Drop it.
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EXPORT_PATH = os.path.join(_SCRIPT_DIR, "_old_chunks_export.json")
INSERT_BATCH = 32

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from engine.import_law_engine import _split_long_segment


def main():
    print("== Step 1: load exported chunks (produced by _export_old_chunks.py) ==")
    if not os.path.exists(EXPORT_PATH):
        raise SystemExit(
            f"Missing {EXPORT_PATH} — run 'python database/_export_old_chunks.py' first."
        )
    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        old_docs = json.load(f)  # list of [text, meta]
    print(f"  {len(old_docs)} chunks loaded.")

    print("\n== Step 2: re-chunk over-long law articles, pass everything else through ==")
    new_docs = []
    split_count = 0
    for text, meta in old_docs:
        is_real_article = meta.get("import_source") == "law" and meta.get("article_number")
        if is_real_article and len(text) > 3000:
            pieces = _split_long_segment(text)
            if len(pieces) > 1:
                split_count += 1
            for i, piece in enumerate(pieces):
                new_meta = dict(meta)
                new_meta["char_count"] = len(piece)
                if len(pieces) > 1:
                    new_meta["segment_index"] = f"{meta.get('segment_index', 0)}.{i}"
                new_docs.append(Document(page_content=piece, metadata=new_meta))
        else:
            new_docs.append(Document(page_content=text, metadata=dict(meta)))

    print(f"  {split_count} over-long articles split into smaller pieces.")
    print(f"  {len(old_docs)} chunks -> {len(new_docs)} chunks after re-chunking.")

    print("\n== Step 3: delete old collection (already backed up separately) ==")
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("\n== Step 4: embed + insert with BAAI/bge-m3 ==")
    new_embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = None
    for i in range(0, len(new_docs), INSERT_BATCH):
        chunk = new_docs[i:i + INSERT_BATCH]
        if vs is None:
            vs = Chroma.from_documents(chunk, new_embedding, persist_directory=DB_PATH)
        else:
            vs.add_documents(chunk)
        print(f"  Indexed {min(i + INSERT_BATCH, len(new_docs))}/{len(new_docs)}")

    print("\n== Step 5: refresh citation-source whitelist ==")
    from engine.rag_engine import refresh_citation_sources
    refresh_citation_sources()

    os.remove(EXPORT_PATH)

    print(f"\nDONE. New chroma_db at {DB_PATH} — {len(new_docs)} chunks, embedding=BAAI/bge-m3.")


if __name__ == "__main__":
    main()
