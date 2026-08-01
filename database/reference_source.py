# reference_source.py
# Registry of standalone reference articles kept indexed in chroma_db —
# content that plugs a specific gap in the corpus (e.g. an eligibility
# condition the bulk-imported law text doesn't cover) rather than a full
# document import. Use Import Law / Import Dataset (engine/import_law_engine.py,
# engine/import_dataset_engine.py) for bulk document imports; use this file
# only for small, hand-picked reference articles like these.
#
# To add a new reference article: append an entry to ARTICLES below and
# re-run this script. Do NOT create a new one-off script for this — this
# file is the single, durable place for "Nguồn tham khảo" additions so they
# stay easy to find, re-run, and dedupe against each other.
#
# USAGE:
#   conda activate rag_env
#   cd D:\hoc\project\rag-legal-assistant-master
#   python -m database.reference_source

import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "chroma_db")

# Each entry is one standalone article. (so_ky_hieu, article_number) together
# must be unique — that pair is what dedup checks against on re-run.
ARTICLES = [
    {
        "so_ky_hieu": "91/2015/QH13",
        "loai_van_ban": "Bộ luật",
        "nguon_thu_thap": (
            "Bộ luật Dân sự năm 2015 (Luật số 91/2015/QH13) "
            "ban hành bởi Cổng thông tin điện tử Chính phủ"
        ),
        "article_number": "20",
        "article_reference": "Điều 20",
        "topic": "Năng lực hành vi dân sự của người thành niên",
        "doc_type": "civil_capacity_condition",
        "retrieval_keywords": (
            "năng lực hành vi dân sự; người thành niên; đủ 18 tuổi; "
            "điều kiện đăng ký hộ kinh doanh; điều kiện thành lập doanh nghiệp"
        ),
        "source_url": "",
        "content": (
            "Điều 20. Năng lực hành vi dân sự của người thành niên\n"
            "Quy tắc pháp lý: Người thành niên là người từ đủ mười tám tuổi trở lên. "
            "Người thành niên có năng lực hành vi dân sự đầy đủ, trừ trường hợp mất năng lực "
            "hành vi dân sự, có khó khăn trong nhận thức, làm chủ hành vi hoặc bị hạn chế năng "
            "lực hành vi dân sự theo quy định tại Điều 22, 23, 24 Bộ luật Dân sự 2015.\n"
            "Từ khóa: năng lực hành vi dân sự; người thành niên; đủ 18 tuổi; điều kiện đăng ký "
            "hộ kinh doanh; điều kiện thành lập doanh nghiệp\n"
            "Ghi chú: Căn cứ để xác định điều kiện về tuổi khi đăng ký hộ kinh doanh, thành lập "
            "doanh nghiệp, hoặc ký kết giao dịch dân sự."
        ),
    },
    {
        "so_ky_hieu": "91/2015/QH13",
        "loai_van_ban": "Bộ luật",
        "nguon_thu_thap": (
            "Bộ luật Dân sự năm 2015 (Luật số 91/2015/QH13) "
            "ban hành bởi Cổng thông tin điện tử Chính phủ"
        ),
        "article_number": "21",
        "article_reference": "Điều 21",
        "topic": "Năng lực hành vi dân sự của người chưa thành niên",
        "doc_type": "civil_capacity_condition",
        "retrieval_keywords": (
            "năng lực hành vi dân sự; người chưa thành niên; chưa đủ 18 tuổi; "
            "từ đủ 15 tuổi đến chưa đủ 18 tuổi; hộ kinh doanh; đăng ký hộ kinh doanh; "
            "người đại diện theo pháp luật đồng ý"
        ),
        "source_url": "",
        "content": (
            "Điều 21. Năng lực hành vi dân sự của người chưa thành niên\n"
            "Quy tắc pháp lý: Người chưa thành niên là người chưa đủ mười tám tuổi. Giao dịch "
            "dân sự của người chưa đủ sáu tuổi do người đại diện theo pháp luật của người đó "
            "xác lập, thực hiện. Người từ đủ sáu tuổi đến chưa đủ mười lăm tuổi khi xác lập, "
            "thực hiện giao dịch dân sự phải được người đại diện theo pháp luật đồng ý, trừ giao "
            "dịch dân sự phục vụ nhu cầu sinh hoạt hàng ngày phù hợp với lứa tuổi. Người từ đủ "
            "mười lăm tuổi đến chưa đủ mười tám tuổi tự mình xác lập, thực hiện giao dịch dân sự, "
            "trừ giao dịch dân sự liên quan đến bất động sản, động sản phải đăng ký và giao dịch "
            "dân sự khác theo quy định của luật phải được người đại diện theo pháp luật đồng ý.\n"
            "Từ khóa: năng lực hành vi dân sự; người chưa thành niên; chưa đủ 18 tuổi; từ đủ 15 "
            "tuổi đến chưa đủ 18 tuổi; hộ kinh doanh; đăng ký hộ kinh doanh; người đại diện theo "
            "pháp luật đồng ý\n"
            "Ghi chú: Áp dụng khi xác định người chưa đủ 18 tuổi có được tự mình đăng ký hộ kinh "
            "doanh, ký hợp đồng hay xác lập giao dịch dân sự khác hay không."
        ),
    },
    # NOTE: an entry citing Nghị định 01/2021/NĐ-CP Điều 80 (hộ kinh doanh
    # eligibility) used to live here, patching a gap in the corpus. NĐ
    # 01/2021 was replaced by NĐ 168/2025/NĐ-CP (Điều 124 khoản 2) effective
    # 2026-07-25 — removed 2026-07-27 since the gap it patched is now covered
    # by the full 168/2025/NĐ-CP bulk import (Điều 82 "Quyền thành lập hộ
    # kinh doanh" + Điều 83 "Nghĩa vụ đăng ký hộ kinh doanh" via Import Law),
    # so a hand-picked standalone entry would only duplicate/risk drifting
    # from the real indexed text. Do not re-add an 01/2021-based entry.

    # ── Thêm entry mới ở đây (append) — không tạo file one-off khác. ──
]


def main():
    docs = [
        Document(
            page_content=a["content"],
            metadata={
                "so_ky_hieu": a["so_ky_hieu"],
                "loai_van_ban": a["loai_van_ban"],
                "nguon_thu_thap": a["nguon_thu_thap"],
                "article_reference": a["article_reference"],
                "article_number": a["article_number"],
                "topic": a["topic"],
                "doc_type": a["doc_type"],
                "retrieval_keywords": a["retrieval_keywords"],
                "source_url": a["source_url"],
                "char_count": len(a["content"]),
            },
        )
        for a in ARTICLES
    ]

    print("Loading embedding model (BAAI/bge-m3)...")
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vs = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    existing = vs.get(include=["metadatas"])
    existing_refs = {
        (m.get("so_ky_hieu", ""), m.get("article_number", ""))
        for m in existing["metadatas"]
    }

    new_docs = [
        d for d in docs
        if (d.metadata["so_ky_hieu"], d.metadata["article_number"]) not in existing_refs
    ]
    skipped = len(docs) - len(new_docs)

    if new_docs:
        vs.add_documents(new_docs)
    print(f"Inserted {len(new_docs)} article(s), skipped {skipped} already-indexed.")
    print(f"Total in DB: {vs._collection.count()}")

    from engine.rag_engine import refresh_citation_sources
    refresh_citation_sources()
    print("Citation source whitelist refreshed.")


if __name__ == "__main__":
    main()
