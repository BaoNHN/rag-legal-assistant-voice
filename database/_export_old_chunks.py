# _export_old_chunks.py
# Step 1/2 of the bge-m3 migration (see rebuild_chroma_bge_m3.py). Runs as
# its OWN process so the sqlite connection it opens against chroma_db is
# fully released on exit — chromadb caches its persistent client per
# process, so reading and deleting chroma_db in the same process fails with
# "PermissionError: file is being used by another process" on Windows.
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "chroma_db")
EXPORT_PATH = os.path.join(_SCRIPT_DIR, "_old_chunks_export.json")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

old_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
old_vs = Chroma(persist_directory=DB_PATH, embedding_function=old_embedding)
data = old_vs.get(include=["documents", "metadatas"])

pairs = list(zip(data["documents"], data["metadatas"]))
with open(EXPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False)

print(f"Exported {len(pairs)} chunks -> {EXPORT_PATH}")
