import os
import re
import unicodedata
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

# Keywords indicating out-of-scope questions (not business law). As of
# 2026-07-28 this is a FALLBACK ONLY — the live source of truth is
# database.database.get_active_out_of_scope_keywords() (keyword.status=2,
# admin-editable in the "Từ khóa" tab), loaded once per ask_rag() call and
# passed into _is_out_of_scope(). This constant is used only if that DB read
# fails, so out-of-scope blocking fails safe (still blocks something) rather
# than fails open (blocks nothing) — do not delete even though it's no
# longer the primary path.
OUT_OF_SCOPE_KEYWORDS = [
    "ly hôn", "li hôn", "hôn nhân", "gia đình", "ly thân", "kết hôn",
    "hình sự", "tội phạm", "khởi tố", "bắt giữ", "truy tố", "tù giam",
    "đất đai", "nhà ở", "bất động sản", "quyền sử dụng đất",
    "bảo hiểm xã hội", "bảo hiểm y tế", "tai nạn lao động",
    "thuế thu nhập cá nhân", "thuế giá trị gia tăng", "thuế tiêu thụ",
    "hải quan", "xuất nhập khẩu", "hành chính công",
]

# An OUT_OF_SCOPE_KEYWORDS hit is overridden when the question also carries
# one of these — it's very likely a genuine Enterprise Law question that
# only references the other domain as supporting context (a disqualifying
# condition, a funding source, a contributed asset), not a question actually
# about that domain. See _is_out_of_scope().
_BUSINESS_CONTEXT_SIGNALS = [
    "công ty", "doanh nghiệp", "hộ kinh doanh", "cổ đông", "cổ phần",
    "thành viên", "vốn điều lệ", "góp vốn", "thành lập", "đăng ký kinh doanh",
    "chủ sở hữu",
]

SIMILARITY_THRESHOLD = 1.3  # L2 distance; above this → not relevant enough
PROCEDURE_PATTERNS = [
    "trình tự",
    "thủ tục",
    "quy trình",
    "các bước",
    "hồ sơ",
    "nộp ở đâu",
    "thành lập",
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

from engine.groq_keys import current_key, rotate_key, get_keys, is_rate_limit_error

# =========================
# EMBEDDING + VECTORSTORE
# =========================
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================
# LLM
# =========================
LLM_MODEL = "llama-3.1-8b-instant"

llm = ChatGroq(
    api_key=current_key(),
    model=LLM_MODEL,
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


def validate_answer_citations(answer: str, context: str) -> bool:
    """Last-resort gate before an answer is shown to the user: every "Điều N"
    the model wrote in its own prose must actually appear in the retrieved
    context. build_citation() never derives the citation *footer* from the
    answer text at all (it's built strictly from best_doc/extra_docs
    metadata) — this catches a different failure: a wrong/invented article
    number left sitting in the answer BODY text, which build_citation()
    never touches, still reading as trustworthy to the user even with no
    citation attached to it."""
    answer_articles = set(re.findall(r'Điều\s+(\d+[a-z]?)', answer or "", flags=re.IGNORECASE))
    if not answer_articles:
        return True

    context_lower = (context or "").lower()
    return all(f"điều {n}" in context_lower for n in answer_articles)


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

    metas = data["metadatas"]
    counts = Counter((m.get("so_ky_hieu") or "").strip() for m in metas)
    counts.pop("", None)
    importers = _first_importer_by_key(metas, "so_ky_hieu")
    return [
        {"so_ky_hieu": k, "chunk_count": v, "importer": importers.get(k, DEFAULT_IMPORTER)}
        for k, v in sorted(counts.items())
    ]


def get_law_source_info(so_ky_hieu: str) -> dict:
    """Basic metadata for one law source (so_ky_hieu, loai_van_ban,
    nguon_thu_thap, chunk_count, importer) for the Manage Law 'Xem thông tin'
    modal — deliberately excludes page_content/chunk text."""
    so_ky_hieu = (so_ky_hieu or "").strip()
    if not so_ky_hieu:
        return {}

    try:
        data = vectorstore.get(where={"so_ky_hieu": so_ky_hieu}, include=["metadatas"])
    except Exception:
        return {}

    metas = data.get("metadatas") or []
    if not metas:
        return {}

    first = metas[0]
    return {
        "so_ky_hieu": so_ky_hieu,
        "loai_van_ban": first.get("loai_van_ban", ""),
        "nguon_thu_thap": first.get("nguon_thu_thap", ""),
        "chunk_count": len(metas),
        "importer": first.get("importer") or DEFAULT_IMPORTER,
    }


def get_law_source_articles(so_ky_hieu: str) -> list:
    """Every distinct article_number that actually exists in ChromaDB for one
    law source, sorted numerically where it parses as an int (non-numeric
    suffixes like "20a" last, alphabetically) — used to constrain the "Tăng
    ưu tiên theo Điều" Điều-number input to real values instead of any
    arbitrary number (2026-07-29 UX request)."""
    so_ky_hieu = (so_ky_hieu or "").strip()
    if not so_ky_hieu:
        return []
    try:
        data = vectorstore.get(where={"so_ky_hieu": so_ky_hieu}, include=["metadatas"])
    except Exception:
        return []
    articles = {
        str(m.get("article_number") or m.get("article"))
        for m in (data.get("metadatas") or [])
        if m.get("article_number") or m.get("article")
    }

    def sort_key(article):
        return (0, int(article)) if article.isdigit() else (1, article)

    return sorted(articles, key=sort_key)


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

# Chunks imported before the "importer" field existed carry no such tag —
# backfill_importer_tags() below stamps them with this placeholder rather
# than leaving the admin UI's "Người nhập" column blank for old data.
DEFAULT_IMPORTER = "admin1"


def _first_importer_by_key(metas: list, key: str, default_group: str = "") -> dict:
    """First non-empty `importer` value seen per distinct `key` value —
    used to show one importer per grouped row in the admin source tables.
    `default_group` mirrors the same fallback label the row-grouping Counter
    uses (e.g. UNKNOWN_SOURCE_FILE_LABEL) so groups line up."""
    result = {}
    for m in metas:
        group = (m.get(key) or "").strip() or default_group
        if group and group not in result:
            result[group] = m.get("importer") or DEFAULT_IMPORTER
    return result


def backfill_importer_tags():
    """One-time migration for chunks indexed before the importer field
    existed. Idempotent — only touches chunks missing it, so it's cheap and
    safe to call on every startup."""
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:
        return

    ids   = data.get("ids") or []
    metas = data.get("metadatas") or []

    update_ids, update_metas = [], []
    for doc_id, m in zip(ids, metas):
        if m.get("importer"):
            continue
        new_meta = dict(m)
        new_meta["importer"] = DEFAULT_IMPORTER
        update_ids.append(doc_id)
        update_metas.append(new_meta)

    if update_ids:
        vectorstore._collection.update(ids=update_ids, metadatas=update_metas)


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


def backfill_entity_type_tags():
    """One-time migration for chunks indexed before import pipelines started
    stamping entity_type at import time (see detect_entity_type() calls in
    import_law_engine.py / import_dataset_engine.py / import_scenario_engine.py).
    Computes it from the same text fields infer_doc_entity()'s runtime
    fallback already scans (page_content/topic/retrieval_keywords/title), so
    future retrieval hits the cheap explicit-field path instead of
    re-inferring from raw text on every question — matters once the corpus
    grows to tens of thousands of chunks. Idempotent — only touches chunks
    missing entity_type, safe to call on every startup."""
    try:
        data = vectorstore.get(include=["metadatas", "documents"])
    except Exception:
        return

    ids   = data.get("ids") or []
    metas = data.get("metadatas") or []
    docs  = data.get("documents") or []

    update_ids, update_metas = [], []
    for doc_id, m, content in zip(ids, metas, docs):
        if m.get("entity_type") or m.get("doc_type") in _ENTITY_AGNOSTIC_DOC_TYPES:
            continue
        text = " ".join([
            str(content or ""),
            str(m.get("topic") or ""),
            str(m.get("retrieval_keywords") or ""),
            str(m.get("title") or ""),
        ])
        entity_type = detect_entity_type(text)
        if not entity_type:
            continue
        new_meta = dict(m)
        new_meta["entity_type"] = entity_type
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

    metas = data["metadatas"]
    counts = Counter((m.get("source_file") or UNKNOWN_SOURCE_FILE_LABEL).strip() for m in metas)
    importers = _first_importer_by_key(metas, "source_file", UNKNOWN_SOURCE_FILE_LABEL)
    return [
        {"name": k, "chunk_count": v, "importer": importers.get(k, DEFAULT_IMPORTER)}
        for k, v in sorted(counts.items())
    ]


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

    metas = data["metadatas"]
    counts = Counter((m.get("nguon_thu_thap") or UNKNOWN_SOURCE_FILE_LABEL).strip() for m in metas)
    importers = _first_importer_by_key(metas, "nguon_thu_thap", UNKNOWN_SOURCE_FILE_LABEL)
    return [
        {"name": k, "chunk_count": v, "importer": importers.get(k, DEFAULT_IMPORTER)}
        for k, v in sorted(counts.items())
    ]


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
backfill_importer_tags()
# backfill_entity_type_tags() runs at the true bottom of this module (after
# ask_rag) — it needs _ENTITY_AGNOSTIC_DOC_TYPES/detect_entity_type, both
# defined further down the file, not yet bound at this point in module load.


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


# =========================
# ENTITY-TYPE DETECTION
# ─────────────────────────────
# The old flow let "TNHH một thành viên" questions get answered from "TNHH
# hai thành viên trở lên" documents (and vice versa) because they share most
# of their vocabulary — nothing hard-filtered on business entity type before
# reranking. This block detects the entity type implied by the question and
# by each candidate document, so filter_compatible_docs() below can drop
# documents belonging to a different entity type before they ever reach the
# context or the reranker.
#
# Real indexed metadata (see import_law_engine.py / import_dataset_engine.py)
# has no "entity_type" field — inference always falls back to scanning
# page_content/topic/retrieval_keywords/title. A doc matching BOTH the
# one-member and multi-member phrasing (a general provision that applies to
# either) intentionally returns None rather than guessing — None means "not
# entity-specific", so filter_compatible_docs() keeps it for any question.
# =========================
def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.lower().split())


_TONE_MARKS = "̣̀́̃̉"  # huyền, sắc, ngã, hỏi, nặng


def _strip_tone(text: str) -> str:
    """Drop Vietnamese tone marks (sắc/huyền/hỏi/ngã/nặng) while keeping the
    base vowel distinct (ă/â/ê/ô/ơ/ư/đ untouched) — makes the hardcoded
    intent/entity-type phrase lists below tolerant of a missing/wrong tone
    mark (e.g. "lâp" vs "lập"), a common typo that otherwise silently drops
    a question out of its intended bucket with no error: "thành lâp..." failed
    to match "thành lập" in _INTENT_PROCEDURE_PHRASES, intent fell back to
    "general", the procedure-intent retrieval boost in _score_doc() never
    fired, and an unrelated article (Điều 203, "chuyển đổi... thành công ty
    TNHH một thành viên") won on raw keyword overlap instead of Điều 21/74."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))


def _phrase_in(phrase: str, text: str) -> bool:
    """Tone-mark-typo-tolerant, case-insensitive substring check, for the
    phrase lists below. Lowercasing here (not left to each call site) matters
    because _strip_tone alone does NOT lowercase — a capitalized word at the
    very start of a sentence ("Hộ kinh doanh phát hiện...") silently failed
    to match its lowercase phrase-list entry ("hộ kinh doanh") with no error,
    just a phrase check that always came back False for any question
    starting that way (2026-07-29 review, ELU186: this is why a "hộ kinh
    doanh" question's household-business boost never fired — some existing
    callers pre-lowercase their own `question` before calling this, e.g.
    detect_query_constraints's normalize_text(), but relying on every
    call site to remember that is exactly how this slipped through)."""
    return _strip_tone(phrase.lower()) in _strip_tone(text.lower())


_ENTITY_ONE_MEMBER_PHRASES = [
    "tnhh một thành viên", "tnhh 1 thành viên",
    "trách nhiệm hữu hạn một thành viên", "trách nhiệm hữu hạn 1 thành viên",
]
_ENTITY_MULTI_MEMBER_PHRASES = [
    "tnhh hai thành viên", "tnhh 2 thành viên", "hai thành viên trở lên",
    "trách nhiệm hữu hạn hai thành viên",
]
_ENTITY_PARTNERSHIP_PHRASES = ["công ty hợp danh", "thành viên hợp danh"]
_ENTITY_JSC_PHRASES = ["công ty cổ phần", "cổ đông", "cổ phần"]
_ENTITY_PRIVATE_PHRASES = ["doanh nghiệp tư nhân", "chủ doanh nghiệp tư nhân"]


def _detect_entity_type(text: str) -> str | None:
    t = normalize_text(text)
    has_one = any(_phrase_in(p, t) for p in _ENTITY_ONE_MEMBER_PHRASES)
    has_multi = any(_phrase_in(p, t) for p in _ENTITY_MULTI_MEMBER_PHRASES)
    if has_one and not has_multi:
        return "llc_one_member"
    if has_multi and not has_one:
        return "llc_multi_member"
    if any(_phrase_in(p, t) for p in _ENTITY_PARTNERSHIP_PHRASES):
        return "partnership"
    if any(_phrase_in(p, t) for p in _ENTITY_JSC_PHRASES):
        return "joint_stock_company"
    if any(_phrase_in(p, t) for p in _ENTITY_PRIVATE_PHRASES):
        return "private_enterprise"
    return None


# Public alias — import_law_engine.py / import_dataset_engine.py /
# import_scenario_engine.py call this at import time to stamp entity_type
# into chunk metadata upfront (see backfill_entity_type_tags() docstring
# above for why this matters at scale). Internal call sites in this module
# keep using the private name.
detect_entity_type = _detect_entity_type


_INTENT_PROCEDURE_PHRASES = [
    "thủ tục", "hồ sơ", "đăng ký thành lập", "cách thành lập", "trình tự",
    "nộp hồ sơ", "thành lập",
]
_INTENT_DEFINITION_PHRASES = ["là gì", "khái niệm", "định nghĩa"]
_INTENT_SCENARIO_PHRASES = [
    # No hardcoded honorific+letter combos ("anh a"/"chị b"/"ông c"/"bà d")
    # — that enumeration only ever covered 4 pseudonyms and missed every
    # other name a scenario question might use ("anh Tèo", "bà Mị", a bare
    # "R"...). Relying only on question-ending phrases below is name-agnostic
    # and catches the same questions regardless of what the hypothetical
    # party is called.
    "tình huống", "có được không", "đúng hay sai",
    "có còn là", "có phải là", "có còn được coi là", "có còn được xem là",
    # Yes/no question endings — deliberately excludes open wh-question
    # endings ("như thế nào", "thế nào", "ra sao", "là gì", "bao lâu"...) —
    # those legitimately belong to procedure/definition intent (e.g.
    # "Thành lập ... như thế nào?" must stay "procedure"), and excludes bare
    # "không?"/"chưa?" alone — too broad, would flip roughly half of all
    # yes/no-phrased questions in the corpus to scenario intent.
    "được không", "phải không", "đúng không", "có đúng không", "hay không",
    "được chưa", "rồi chưa",
    # "X cần xử lý thế nào?" / "...giải quyết ra sao?" surface-match the same
    # open wh-ending as a genuine procedure question ("Thành lập ... như thế
    # nào?"), so bare ending-shape alone can't tell them apart — but "xử lý"/
    # "giải quyết" (handle/resolve A PROBLEM) only ever appears in this
    # corpus on scenario_case rows about one specific party's specific
    # violation (ELS016, ELS028, ELU164 — verified against the full 200-row
    # dataset, 2026-07-29), never on a genuine "how do I register" question.
    # Without this, "thành lập" elsewhere in the same sentence (scene-setting
    # for the scenario, e.g. "...khi thành lập công ty cổ phần nhưng không
    # thanh toán đủ...") won hard as "procedure" intent, which pulls in every
    # _ESTABLISHMENT_DOC_TYPES doc via retrieve_docs' procedure-intent
    # augmentation and outranks the actually-correct article via _score_doc's
    # +25 procedure boost (ELS028 cited Điều 22 "Hồ sơ đăng ký" instead of the
    # correct Điều 113 "Thanh toán cổ phần..." this way — 2026-07-29 review).
    "xử lý thế nào", "xử lý ra sao", "giải quyết thế nào", "giải quyết ra sao",
    # "Còn lại có đủ điều kiện ... không?" — an eligibility/headcount check on
    # a specific hypothetical, not a "what are the conditions" definition
    # question (those instead end "là gì?", already caught by
    # _INTENT_DEFINITION_PHRASES). Also verified 100% scenario_case in the
    # full dataset (ELS026, ELS036).
    "có đủ điều kiện",
]


def detect_query_constraints(question: str) -> dict:
    q = normalize_text(question)
    intent = "general"
    # Scenario checked BEFORE procedure — bare "thành lập" (establish) is in
    # _INTENT_PROCEDURE_PHRASES and matches almost any hypothetical about
    # forming a company ("R muốn thành lập... Có còn là...không?"), even when
    # the actual question is a yes/no eligibility/classification check, not
    # a "how do I register" procedure question. That let "thành lập" win the
    # old procedure-first order and drag in _ESTABLISHMENT_DOC_TYPES articles
    # (hồ sơ/trình tự/cấp GCN) over the real definitional article (Điều 74)
    # (2026-07-28 eval review, ELS022/ELS040 both misfired this way). A
    # scenario phrase like "có được không"/"có còn là" is a far more specific
    # signal than a bare "thành lập" mention, so it should win when both are
    # present — a genuine procedure question ("thành lập ... như thế nào?")
    # still falls through correctly since it matches no scenario phrase.
    if any(_phrase_in(p, q) for p in _INTENT_SCENARIO_PHRASES):
        intent = "scenario"
    elif any(_phrase_in(p, q) for p in _INTENT_DEFINITION_PHRASES):
        intent = "definition"
    elif any(_phrase_in(p, q) for p in _INTENT_PROCEDURE_PHRASES):
        intent = "procedure"
    return {"entity_type": _detect_entity_type(q), "intent": intent}


# Doc types that state a rule applicable to ALL business entity types (who
# may establish a business, civil capacity/age conditions, general
# registration procedure...) rather than one specific structure. Left to
# text inference, an incidental keyword — e.g. Điều 17 (quyền thành lập,
# entity-agnostic) mentions "mua cổ phần" only as one of several rights it
# lists, which _ENTITY_JSC_PHRASES matches on "cổ phần" alone — would wrongly
# scope a general provision to one entity type and get it excluded by
# filter_compatible_docs() for every OTHER question (e.g. Điều 17 never
# surfacing for a TNHH một thành viên question — 2026-07-25 eval review).
_ENTITY_AGNOSTIC_DOC_TYPES = {
    "establishment_eligibility", "civil_capacity_condition",
    "household_business_eligibility_condition",
    # Beneficial-owner declaration criteria apply across every entity type in
    # the same article (JSC shareholders, LLC/partnership members, single-
    # member LLC owners all listed side by side) — a law-import chunk here
    # had an incorrectly auto-tagged entity_type="llc_multi_member" that
    # wrongly excluded it from JSC questions like ELU161 (2026-07-30 fix).
    "beneficial_owner_declaration",
    # Same failure mode, different article: "Chuyển đổi doanh nghiệp tư nhân
    # thành công ty TNHH, công ty cổ phần, công ty hợp danh" (Điều 205) is
    # addressed to a DNTN owner considering ANY of three target types —
    # entity-type auto-detection picked up "hợp danh" (the last type named
    # in the title) and tagged it entity_type="partnership" specifically,
    # so filter_compatible_docs() excluded it entirely from a question about
    # converting to TNHH (found live 2026-07-31, ELS050: the article scored
    # highest of all candidates — 109 vs the next at 53 — but never got the
    # chance because the entity filter dropped it from 19 candidates down to
    # 3 before scoring ever ran, and the LLM latched onto an unrelated
    # scenario_qa chunk sharing generic "doanh nghiệp tư nhân" vocabulary
    # instead, producing a completely off-topic answer). Checking the whole
    # convert_* family after finding this one (2026-07-31) confirmed it's
    # systemic, not a one-off: every "chuyển đổi X thành Y" article is
    # tagged with only the TARGET entity type (convert_llc_to_jsc →
    # entity_type="joint_stock_company", convert_jsc_to_single_llc →
    # "llc_one_member", convert_jsc_to_multi_llc → "llc_multi_member"), so
    # each one gets excluded whenever a question is framed from the SOURCE
    # entity's side — which is how these questions are naturally phrased
    # ("Công ty TNHH muốn chuyển thành công ty cổ phần...", ELS048: same
    # symptom as Điều 205, correct article scored highest of all candidates
    # but got filtered out before scoring ran, and Điều 204 — the reverse
    # direction — won instead).
    "convert_private_enterprise", "convert_llc_to_jsc",
    "convert_jsc_to_single_llc", "convert_jsc_to_multi_llc",
}


def infer_doc_entity(doc) -> str | None:
    """Entity type of a retrieved Document. Checks an explicit entity_type/
    doc_type metadata field first (forward-compatible with future imports
    that tag it), then falls back to text inference — no currently indexed
    chunk actually carries that field yet."""
    metadata = doc.metadata or {}
    if metadata.get("doc_type") in _ENTITY_AGNOSTIC_DOC_TYPES:
        return None
    explicit = _detect_entity_type(metadata.get("entity_type") or metadata.get("doc_type") or "")
    if explicit:
        return explicit
    text = " ".join([
        str(doc.page_content or ""),
        str(metadata.get("topic") or ""),
        str(metadata.get("retrieval_keywords") or ""),
        str(metadata.get("title") or ""),
    ])
    return _detect_entity_type(text)


def filter_compatible_docs(question: str, docs: list) -> list:
    """Drop documents whose entity type conflicts with the one the question
    asks about. Docs with no detectable entity type (general provisions) are
    always kept — see module docstring above for why that's intentional."""
    expected_entity = detect_query_constraints(question)["entity_type"]
    if not expected_entity:
        return docs

    compatible = []
    for doc in docs:
        doc_entity = infer_doc_entity(doc)
        if doc_entity and doc_entity != expected_entity:
            continue
        compatible.append(doc)
    return compatible


# =========================
# QUERY REWRITE
# =========================
def _llm_invoke_with_retry(prompt: str, retries: int = 3) -> str:
    """Invoke LLM with retry on rate-limit/timeout errors.

    On a rate-limit error, rotates to the next Groq API key (see
    groqkey.txt / groq_keys.py) before falling back to a timed backoff — a
    429 on one key says nothing about whether a sibling key is also
    limited, so trying the next key costs no wall-clock time and is
    strictly better than sleeping out the same key's window."""
    import time
    global llm
    rotations_left = max(len(get_keys()) - 1, 0)
    attempt = 0
    while True:
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            if is_rate_limit_error(e) and rotations_left > 0:
                rotations_left -= 1
                llm = ChatGroq(api_key=rotate_key(), model=LLM_MODEL, temperature=0)
                print(f"  ⚠ Groq rate limit — rotating API key ({rotations_left} left)")
                continue
            err = str(e).lower()
            if attempt < retries and any(k in err for k in ["rate", "timeout", "429", "503", "connection"]):
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                attempt += 1
                print(f"  ⚠ LLM retry {attempt}/{retries} after {wait}s ({err[:40]})")
                time.sleep(wait)
                continue
            raise


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
def extract_article_number_from_question(question: str) -> list[str]:
    """Every distinct "Điều N" number mentioned, in the order first seen.
    Was re.search (first match only) until 2026-07-31 — a comparison
    question naming two articles ("Phân biệt Điều 119 và Điều 120...") only
    ever captured the first, and retrieve_docs()'s exact-match short-circuit
    then returned ONLY Điều 119 chunks, so Điều 120 never entered the
    candidate pool at all. Every generated answer then mentioned "Điều 120"
    (the question demands a comparison) with nothing in context to back it,
    so validate_answer_citations() rejected all 3 retry attempts and the
    question fell through to the "couldn't find a matching answer" fallback
    (ELU189, found live 2026-07-31)."""
    seen = []
    for m in re.finditer(r'điều\s+(\d+[a-z]?)\b', question, re.IGNORECASE):
        n = m.group(1)
        if n not in seen:
            seen.append(n)
    return seen


# =========================
# KEYWORD-PHRASE RECALL
# ─────────────────────────────
# Token-overlap scoring against retrieval_keywords/page_content, independent
# of embedding similarity. Requires a minimum score so a single stray common
# word doesn't drag in an unrelated article — this is a recall booster for
# the semantic search, not a replacement for select_best_doc()'s reranking.
# =========================
def _keyword_recall(question: str, all_docs: list, top_n: int = 12, min_score: int = 4) -> list:
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


def _source_keyword_candidates(question: str, all_docs: list, source_keywords_map: dict, top_n: int = 5) -> list:
    """When the question matches a keyword an admin/teacher tagged onto a
    source (see database.database.keyword/source_keyword), pull that
    source's best-matching chunks into the retrieval candidate pool — same
    rationale as the PROCEDURE-INTENT AUGMENTATION below: a tagged source's
    chunks might never surface in the semantic/keyword top-5 on their own
    (e.g. a newly-imported law with no curated retrieval_keywords), so the
    _score_doc() keyword boost would never get a chance to matter — tagging
    a source would otherwise only be a tie-breaker among docs retrieval
    already happened to find, not an actual lever.

    Capped to top_n chunks per matched source (ranked by raw token overlap
    with the question) — deliberately NOT the whole source. A tagged source
    can have hundreds of chunks (e.g. 59/2020/QH14 has 310); dumping all of
    them in would blow up the candidate pool and reintroduce the same
    noise-drowns-signal failure mode the boost-magnitude fix in _score_doc()
    was about (measured live 2026-07-28, ELU186 — see that comment for the
    concrete before/after scores)."""
    if not source_keywords_map:
        return []

    q_words = {w for w in tokenize(question) if w not in STOPWORDS}
    if not q_words:
        return []

    # Which (source_type, source_key) pairs does this question's text
    # actually match? Check this first so we don't bother scoring the whole
    # corpus when no tag applies. Checks all three kind buckets (2026-07-30
    # fix) — this used to check only primary|secondary, which meant it never
    # fired for ANY priority tag: document-level "law" sources retired their
    # primary/secondary buckets entirely in favor of priority-only back on
    # 2026-07-29, and every article-level ("law_article") tag added since
    # then was saved under priority kind too (see _score_doc's merged
    # article-scoring comment) — so this whole augmentation channel had been
    # silently inert for the entire priority mechanism, article-level tags
    # included, ever since. Article-level tags in particular had NO
    # dedicated recall-boosting path at all (the loop below only ever built
    # "law"/"dataset"/"scenario" keys via _source_type_key(), never
    # "law_article"), so a tagged Điều only ever entered the candidate pool
    # by luck via keyword_recall/semantic search ranking — found live
    # 2026-07-30 while diagnosing ELS020/ELU158, both genuine recall gaps
    # despite having matching article-level tags.
    matched_keys = set()
    for source_type, by_key in source_keywords_map.items():
        for source_key, kw_map in by_key.items():
            names = kw_map["primary"] | kw_map["secondary"] | kw_map["priority"]
            if any(_phrase_in(name, question) for name in names):
                matched_keys.add((source_type, source_key))
    if not matched_keys:
        return []

    scored_by_source: dict = {}
    for d in all_docs:
        keys = [_source_type_key(d.metadata)]
        so_ky_hieu  = d.metadata.get("so_ky_hieu", "")
        article_num = d.metadata.get("article_number") or d.metadata.get("article")
        if so_ky_hieu and article_num:
            keys.append(("law_article", f"{so_ky_hieu}#{article_num}"))
        matched = [k for k in keys if k in matched_keys]
        if not matched:
            continue
        content = d.page_content.lower()
        title = (d.metadata.get("title") or "").lower()
        score = sum(1 for w in q_words if w in content or w in title)
        for key in matched:
            scored_by_source.setdefault(key, []).append((score, d))

    result, seen = [], set()
    for scored in scored_by_source.values():
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, d in scored[:top_n]:
            if id(d) not in seen:
                seen.add(id(d))
                result.append(d)
    return result


# =========================
# TOPIC-AWARE RETRIEVAL
# ─────────────────────────────
# FIX for knowledge questions scoring 57.7/100:
# 1. Try ChromaDB metadata filter on topic first (exact match)
# 2. Fall back to standard semantic search if no match
# This eliminates wrong article retrieval (Điều 34 instead of 36, etc.)
# =========================
def retrieve_docs(question: str, rewritten_q: str, source_keywords_map: dict = None):
    topic = extract_topic_from_question(question)
    article_nums = extract_article_number_from_question(question)

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
        # Capped per-number (not one flat slice across all of them) so a
        # comparison question naming two articles doesn't let one number's
        # chunks crowd the other entirely out of the returned pool.
        if article_nums:
            exact = []
            for num in article_nums:
                matches = [d for d in all_docs if d.metadata.get("article_number", "") == num]
                exact.extend(matches[:3])
            if exact:
                return exact

        # ===== STRICT TOPIC MATCH =====
        # For a bare generic noun ("doanh nghiệp") the SequenceMatcher ratio
        # against ANY longer topic that happens to contain it as a substring
        # ("Tên doanh nghiệp", "Doanh nghiệp xã hội"...) is inherently high —
        # that's a property of string containment, not evidence the topic is
        # actually about defining the term. Short-circuiting straight to
        # scored[:5] on that alone previously made "Doanh nghiệp là gì?" cite
        # Điều 37/Điều 10 instead of the real definition in Điều 4 (2026-07-28
        # eval review) — semantic search alone ranked Điều 4 far closer
        # (0.60 vs 0.81+ L2 distance) but never got a chance to run. Only a
        # near-exact topic match (topic == kb_topic, sim ≥ ~1.5 — containment
        # bonuses alone can reach at most ~1.3) is trusted enough to
        # short-circuit; anything looser is merged with the keyword/semantic
        # candidates below instead of replacing them, so the reranker can
        # still weigh in.
        topic_candidates = []
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

            if scored and scored[0][0] >= 1.5:
                return [d for _, d in scored[:5]]
            if scored and scored[0][0] >= 0.55:
                topic_candidates = [d for _, d in scored[:5]]

        # ===== BARE-KEYWORD SUBSTRING MATCH =====
        # Short queries like "Tập đoàn" (no question phrasing, no topic/article
        # match above) — embedding similarity is unreliable for 1-3 word
        # Vietnamese legal terms with an English-tuned model. Scan titles and
        # content directly for the phrase instead.
        #
        # Gated on `not topic` — this is meant only for genuinely bare noun
        # queries with no question phrasing (as the comment above always
        # said), but the code never actually checked that, so a phrased
        # query like "Doanh nghiệp là gì?" (4 words, so still ≤5) fell
        # through to this raw substring scan too. Many Q&A-style chunks in
        # this corpus are templated as "...quy định về <X> là gì?", and
        # whenever X itself happens to end in "doanh nghiệp" (e.g. "cấp giấy
        # chứng nhận đăng ký doanh nghiệp"), the chunk's own embedded
        # question coincidentally contains "doanh nghiệp là gì?" as a
        # substring — matching completely unrelated articles (Điều 27/28/212
        # about certificates, not the Điều 4 definition) on pure string luck
        # (2026-07-28 eval review, same report as the topic-match fix above).
        q_lower = question.lower().strip()
        if not topic and q_lower and len(q_lower.split()) <= 5:
            kw_hits = [
                d for d in all_docs
                if q_lower in d.page_content.lower() or q_lower in d.metadata.get("title", "").lower()
            ]
            if kw_hits:
                return kw_hits[:5]

        # ===== HYBRID: keyword-phrase recall + semantic search =====
        # For longer, naturally-phrased questions, semantic search alone can
        # still miss the right article even when its retrieval_keywords field
        # is a near-exact match for the question (originally observed with
        # bge-small-en-v1.5, an English-tuned model — kept as a safety net
        # after switching to the multilingual bge-m3, since keyword-token
        # overlap is a cheap independent signal regardless of embedding
        # quality) — e.g. "B 16 tuổi có được đăng ký hộ kinh doanh không?"
        # never surfaced the Điều 20/21 BLDS 2015 doc (keywords: "chưa đủ 18
        # tuổi", "hộ kinh doanh"...) in the semantic top-5. Score every doc by
        # keyword-token
        # overlap independent of embedding similarity, and merge those hits
        # ahead of the semantic results so select_best_doc() actually gets a
        # chance to consider them.
        kw_candidates = _keyword_recall(question, all_docs)

        results_with_scores = vectorstore.similarity_search_with_score(rewritten_q, k=5)
        # rewrite_query() is itself an LLM call with no guaranteed
        # reproducibility — a differently-worded paraphrase on a different
        # run can shift which candidates the ONE semantic search below even
        # sees, with no fallback anchor on the original phrasing (found live
        # 2026-07-31, ELS050: with one rewrite, Điều 205 — the correct
        # article — won by a wide margin, 109 vs 53; with a differently-
        # worded rewrite from a separate run, Điều 205 didn't even make the
        # candidate pool, and an unrelated scenario_qa chunk sharing generic
        # "doanh nghiệp tư nhân" vocabulary won a context slot instead,
        # visibly dragging the generated answer off-topic). Also searching
        # the untouched original question gives semantic search a second,
        # rewrite-independent anchor — cheap (one extra vector search, no
        # extra LLM call) since `question` is already in hand.
        if rewritten_q != question:
            results_with_scores += vectorstore.similarity_search_with_score(question, k=5)
        # Filter: L2 distance lower = more similar; reject docs above threshold
        semantic_candidates = [doc for doc, score in results_with_scores if score <= SIMILARITY_THRESHOLD]
        # Keyed by page_content (this function's own dedup key, see `seen`
        # below) rather than object identity — similarity_search_with_score()
        # returns its own separate Document instances from vectorstore.get()'s,
        # so identity wouldn't survive the merge. min(): a doc found via both
        # searches keeps its closer of the two distances.
        distance_by_content: dict = {}
        for doc, score in results_with_scores:
            prev = distance_by_content.get(doc.page_content)
            distance_by_content[doc.page_content] = score if prev is None else min(prev, score)

        merged, seen = [], set()
        for d in topic_candidates + kw_candidates + semantic_candidates:
            if d.page_content not in seen:
                merged.append(d)
                seen.add(d.page_content)

        # Stamp the embedding distance onto every candidate that had one,
        # regardless of which path (topic/keyword/semantic) actually pulled
        # it into the pool — _score_doc reads this to keep a strong semantic
        # match from losing to a chunk that only entered via keyword-tag
        # recall and never showed real embedding proximity (2026-07-30, see
        # that function's own comment: ELD037's correct Điều 17 ranked #2 by
        # raw bge-m3 similarity — distance 0.75 vs. threshold 1.3 — yet lost
        # to Điều 52, which never appeared in the semantic top-5 at all,
        # purely on keyword-tag stacking).
        for d in merged:
            dist = distance_by_content.get(d.page_content)
            if dist is not None:
                d.metadata["_embedding_distance"] = dist

        # ===== SOURCE-KEYWORD AUGMENTATION =====
        # See _source_keyword_candidates() docstring — guarantees a tagged
        # source's best-matching chunks (capped, not the whole source) enter
        # the candidate pool even if semantic/keyword search above missed them.
        for d in _source_keyword_candidates(question, all_docs, source_keywords_map):
            if d.page_content not in seen:
                merged.append(d)
                seen.add(d.page_content)

        # ===== PROCEDURE-INTENT AUGMENTATION =====
        # "Thành lập công ty X như thế nào?" shares almost all of its
        # vocabulary ("công ty", "tnhh", "thành viên"...) with ownership/
        # definition articles, so those articles' keyword-overlap score
        # crowds out the actual hồ sơ/trình tự/cấp-GCN articles (doc_type
        # registration_llc / registration_procedure / erc_issuance etc.)
        # from ever reaching the top-5 kw/semantic candidates above. Pull
        # every establishment-shaped doc_type in directly so
        # select_best_doc()'s procedure-intent boost (see _score_doc) has
        # something to actually rank. filter_compatible_docs() downstream
        # still drops entity-type mismatches (e.g. registration_jsc docs
        # for a TNHH one-member question).
        if detect_query_constraints(question)["intent"] == "procedure":
            for d in all_docs:
                doc_type = d.metadata.get("doc_type", "")
                if doc_type in _ESTABLISHMENT_DOC_TYPES and d.page_content not in seen:
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
_OFFICIAL_SOURCE_MARKERS = [
    "cổng thông tin", "chính phủ", "công báo", "vbpl.vn", "chinhphu.vn", "văn bản hợp nhất",
]

# chroma_db carries two parallel copies of Enterprise Law — "59/2020/QH14"
# (the original 2020 law-import) and "67/VBHN-VPQH" (the 2025 consolidated
# text). Article-level tags (see below) were added per-copy as they came up,
# so whichever copy happened to score lower at admin-tagging time silently
# missed out on tags the other copy got (2026-07-31, user report: "VBHN 67
# lép vế" — same Điều 127, VBHN copy scored 65 vs the 59/2020 copy's 83 on
# ELS033, purely from tag coverage, not content). Mirrored at lookup time
# (see _score_doc) so a tag added to either copy of "Điều N" now helps both
# automatically — EXCEPT where the two copies are actually different
# articles under the same number: this isn't just a handful of individually-
# amended articles (a first attempt hand-listed 6 of them from the chủ sở
# hữu hưởng lợi additions), it's whole-section renumbering — e.g.
# so_ky_hieu=67/VBHN-VPQH's "Điều 54" is "Cơ cấu tổ chức quản lý công ty"
# while so_ky_hieu=59/2020/QH14's "Điều 54" is "Xử lý phần vốn góp trong một
# số trường hợp đặc biệt", found live 2026-07-31 debugging ELS020. A
# hand-curated exclusion list can't be trusted to be exhaustive, so this
# compares every article's actual heading text between the two copies
# instead (user's suggestion) and excludes any pair below a similarity
# threshold — calibrated at 0.6 against known safe pairs (e.g. "Công ty
# TNHH một thành viên" vs "Công ty trách nhiệm hữu hạn một thành viên",
# 0.78 — same article, just spelled out) and known-divergent ones (e.g.
# Điều 54 above, 0.20; Điều 119 "Sổ đăng ký cổ đông" vs "Nghĩa vụ của cổ
# đông", 0.58).
_ENTERPRISE_LAW_SIBLING = {
    "59/2020/QH14": "67/VBHN-VPQH",
    "67/VBHN-VPQH": "59/2020/QH14",
}
_ENTERPRISE_LAW_HEADING_SIMILARITY_THRESHOLD = 0.6

# Heading comparison alone can't catch this class of divergence: the 2025
# amendments added whole new khoản to some articles (chủ sở hữu hưởng lợi
# provisions) without changing the article's own title, so 59/2020/QH14 and
# 67/VBHN-VPQH still read as the same heading ("Nghĩa vụ của doanh nghiệp",
# etc.) despite VBHN 67 having content the original law never had at all
# (confirmed live 2026-07-30, Điều 4/8/23/25/216/217). A hand-picked list of
# "which articles got amended" isn't trustworthy either — checked instead
# for the one phrase that's the actual marker of this entire amendment
# package (per its own delegation clause, VBHN 67 Điều 217 khoản 6: "Chính
# phủ quy định chi tiết... về chủ sở hữu hưởng lợi"): if it appears in one
# copy's text for an article but not the other's, that's content divergence
# regardless of whether the article was on today's list of known cases.
# (Doesn't need to separately handle Điều 23/25/216/217 specifically — they
# have zero chunks under so_ky_hieu=59/2020/QH14 at all, so there's no
# competing chunk on that side for mirroring to wrongly boost in the first
# place; this check only matters where both copies actually have content.)
_ENTERPRISE_LAW_CONTENT_DIVERGENCE_MARKERS = ["chủ sở hữu hưởng lợi"]
_enterprise_law_divergent_articles_cache: set | None = None


def _get_enterprise_law_divergent_articles() -> set:
    """Article numbers where the two Enterprise Law copies' headings don't
    match closely enough to trust tag-mirroring between them. Computed once
    (chroma_db is static within a process run) and cached — this queries the
    whole collection, so it must not run per _score_doc() call."""
    global _enterprise_law_divergent_articles_cache
    if _enterprise_law_divergent_articles_cache is not None:
        return _enterprise_law_divergent_articles_cache

    headings: dict[tuple[str, str], str] = {}
    full_text: dict[tuple[str, str], list] = {}
    try:
        # An article can have several dataset chunks (one per khoản, each
        # with its own "topic" covering just that khoản), and chromadb's
        # return order isn't meaningful, so comparing whichever chunk
        # happened to come back first risked comparing two unrelated khoản
        # summaries instead of the two articles' actual headings (caught
        # live 2026-07-31: an early version flagged Điều 4 as divergent
        # because it picked up a "Khoản 35 Điều 4" dataset chunk's own topic
        # string, not Điều 4's real "Giải thích từ ngữ" heading). Fixed by
        # preferring, per (so_ky_hieu, article): the import_source="law" raw
        # full-text chunk if one exists, else the one dataset chunk whose
        # article_reference is the bare article ("Điều N", not "Khoản X
        # Điều N") — a whole-article summary, not a per-khoản one. Necessary
        # because 59/2020/QH14 turns out to have ZERO law-import chunks at
        # all (found live 2026-07-31) — every article on that side is
        # dataset-curated, so requiring import_source="law" on both sides
        # silently compared nothing and reported 0 divergent articles,
        # exactly the "unverified defaults to safe" failure mode this
        # function exists to avoid.
        results = vectorstore.get(
            where={"so_ky_hieu": {"$in": list(_ENTERPRISE_LAW_SIBLING.keys())}},
            include=["metadatas", "documents"],
        )
        whole_article_dataset_chunks: dict[tuple[str, str], str] = {}
        for doc_text, meta in zip(results["documents"], results["metadatas"]):
            so_ky_hieu  = meta.get("so_ky_hieu", "")
            article_num = meta.get("article_number", "")
            key = (so_ky_hieu, article_num)
            if not article_num:
                continue

            # Every chunk for this (so_ky_hieu, article) — not just the
            # whole-article one below — feeds the content-marker check, so a
            # marker sitting in just one khoản-level dataset chunk still
            # counts.
            full_text.setdefault(key, []).append(doc_text or "")

            first_line = (doc_text or "").split("\n", 1)[0]
            heading = first_line.split(".", 1)[-1].strip() if "." in first_line else first_line

            if meta.get("import_source") == "law":
                headings[key] = heading  # authoritative, always wins
                continue

            article_ref = (meta.get("article_reference") or "").strip().lower()
            if article_ref and not article_ref.startswith("khoản") and key not in whole_article_dataset_chunks:
                whole_article_dataset_chunks[key] = heading

        for key, heading in whole_article_dataset_chunks.items():
            headings.setdefault(key, heading)
    except Exception:
        _enterprise_law_divergent_articles_cache = set()
        return _enterprise_law_divergent_articles_cache

    article_nums = {k[1] for k in headings} | {k[1] for k in full_text}
    divergent = set()
    for article_num in article_nums:
        h1 = headings.get(("59/2020/QH14", article_num))
        h2 = headings.get(("67/VBHN-VPQH", article_num))
        if h1 and h2 and similarity(h1, h2) < _ENTERPRISE_LAW_HEADING_SIMILARITY_THRESHOLD:
            divergent.add(article_num)
            continue

        t1 = " ".join(full_text.get(("59/2020/QH14", article_num), [])).lower()
        t2 = " ".join(full_text.get(("67/VBHN-VPQH", article_num), [])).lower()
        if not t1 or not t2:
            continue  # one side has no chunk at all — nothing to mirror to
        for marker in _ENTERPRISE_LAW_CONTENT_DIVERGENCE_MARKERS:
            if (marker in t1) != (marker in t2):
                divergent.add(article_num)
                break

    _enterprise_law_divergent_articles_cache = divergent
    return divergent


def _source_type_key(metadata: dict) -> tuple:
    """Maps a chunk's metadata to the (source_type, source_key) pair used by
    database.database.source_keyword — see init_db()'s comment on that table
    for the exact field mapping per import type. Returns (None, "") for
    chunks from neither of the 3 UI-driven import pipelines (e.g. the
    standalone database/reference_source.py entries)."""
    import_source = metadata.get("import_source", "")
    if import_source == "law":
        return "law", (metadata.get("so_ky_hieu") or "").strip()
    if import_source == "dataset":
        return "dataset", (metadata.get("source_file") or "").strip()
    if import_source == "scenario":
        return "scenario", (metadata.get("nguon_thu_thap") or "").strip()
    return None, ""


def _score_doc(question: str, d, q_counter: Counter, intent: str, source_keywords_map: dict = None) -> int:
    score = 0

    content = d.page_content.lower()
    metadata = d.metadata

    # Embedding-similarity bonus — bge-m3 can correctly rank a chunk #2 out
    # of hundreds by raw semantic distance, then have every keyword-overlap
    # signal below still lose it to a chunk that only entered the candidate
    # pool via keyword-tag recall and never showed real semantic proximity
    # (found 2026-07-30, ELD037: correct Điều 17 at L2 distance 0.75 — well
    # inside SIMILARITY_THRESHOLD's 1.3 cutoff — lost to Điều 52, which
    # wasn't even in the raw top-5 semantic results). retrieve_docs() stamps
    # "_embedding_distance" (lower = closer) onto every candidate that
    # appeared in the raw semantic search, regardless of which path actually
    # pulled it into the pool; docs that never showed up there (topic/
    # keyword/tag-only hits) get no bonus here, not a penalty — additive-only,
    # same philosophy as the keyword-priority tags below. Scaled so a very
    # close match (~0.75) is worth about two priority-tag matches (~30),
    # tapering to ~0 at the threshold itself.
    embedding_distance = metadata.get("_embedding_distance")
    if embedding_distance is not None:
        score += max(0, round((SIMILARITY_THRESHOLD - embedding_distance) * 20))

    # keyword overlap
    for w, cnt in q_counter.items():
        if w in content:
            score += cnt

    # metadata keywords — single word match. Curated dataset entries
    # (import_source="dataset") hand-pick short, distinctive keyword phrases,
    # so a single-word hit there is a meaningful signal. Law-import chunks
    # (import_source="law") instead get retrieval_keywords auto-derived from
    # their own article heading (see import_law_engine.py) — a whole
    # sentence-topic, not curated, so common domain words it shares with
    # dozens of other articles in the same chapter ("công ty", "cổ đông"...)
    # would otherwise rack up the same 3x credit as a genuinely specific
    # curated match. Downweighting law-import matches to the same 1x as raw
    # content overlap fixed a regression where a topically-adjacent-but-wrong
    # official article (e.g. Điều 149 sharing "cổ đông" with the question)
    # outscored the actually-correct curated chunk (Điều 111) on a scenario
    # question — see 2026-07-25 eval-score-drop investigation.
    kw_field = metadata.get("retrieval_keywords", "")
    kw_text = kw_field.lower().replace(";", " ")
    kw_word_weight = 1 if metadata.get("import_source") == "law" else 3

    for w, cnt in q_counter.items():
        if w in kw_text:
            score += cnt * kw_word_weight

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

    # Source-tier priority: official law text > standardized KB > raw Q&A
    # dataset. Markers/thresholds taken from what's actually indexed
    # today (see engine.rag_engine module — law chunks are tagged
    # nguon_thu_thap="Cổng thông tin chính phủ"). Kept small (a tiebreaker,
    # not a dominant force) — it used to be +20, which combined with the
    # kw_word_weight above let a merely topically-adjacent official article
    # systematically outrank the actually-correct curated chunk on scenario
    # questions with repeated common terms (see kw_word_weight comment).
    # KB_Articles tier (+2) must stay below the official-source tier (+6) —
    # a separate "kb_articles_updated" +10 stacked on top of this used to
    # push KB chunks to +12, ABOVE official law text, contradicting this
    # comment's own stated order (found 2026-07-30, user review); removed.
    nguon = metadata.get("nguon_thu_thap", "")
    nguon_lower = nguon.lower()
    if any(marker in nguon_lower for marker in _OFFICIAL_SOURCE_MARKERS):
        score += 6
    elif "kb_articles" in nguon_lower:
        score += 2
    # "dataset_200" scoring removed 2026-07-30: the 2026-07-28 dataset-leak
    # fix took every Dataset_*/Demo_* Q&A sheet out of ChromaDB entirely
    # (only KB_Articles*/Legal_Update_2025 stay indexed) — no chunk in
    # chroma_db has carried "dataset_200" in nguon_thu_thap since, so this
    # branch was unreachable dead code (verified live, 0/543 chunks).

    # Admin/teacher-tagged keyword boost (see database.database.keyword /
    # source_keyword tables) — this is the generic, extensible replacement
    # for hardcoding a topic check per source: any law/dataset/scenario
    # source tagged with a primary keyword that appears in the question gets
    # a boost, secondary keyword a smaller one. Newly imported sources get
    # this for free the moment they're tagged, without touching this
    # function again. Purely additive: source_keywords_map only has entries
    # for (source_type, source_key) pairs an admin/teacher has actually
    # tagged, so untagged sources (including everything indexed before this
    # feature existed) are unaffected.
    #
    # Deliberately kept small (8/4, not 25/10): this boost is applied
    # uniformly to every chunk of the tagged source, including generic raw
    # law-import text. A so_ky_hieu can have BOTH plain law-import chunks AND
    # a hand-curated dataset chunk built specifically for a given question
    # (dataset chunks score higher per-word via kw_word_weight=3 vs 1 for
    # law-import, see above) — a flat +25 was large enough to let a generic
    # tagged chunk outscore the correct curated answer by a couple of points
    # (measured live, 2026-07-28, ELU186: curated chunk scored 84, raw law
    # chunks scored 61 untagged vs 86 tagged at +25, flipping the winner to
    # the wrong specific Điều within the right so_ky_hieu). At +8/+4 the same
    # raw chunks land at 69, still below the curated chunk's 84.
    source_type, source_key = _source_type_key(metadata)
    kw_map = (source_keywords_map or {}).get(source_type, {}).get(source_key) if source_type else None
    if kw_map:
        if any(_phrase_in(name, question) for name in kw_map["primary"]):
            score += 8
        elif any(_phrase_in(name, question) for name in kw_map["secondary"]):
            score += 4

        # "priority" keywords boost THIS source directly and ADDITIVELY —
        # every matched keyword's full value counts (+15 each), not just the
        # first match — so an admin can close a gap of any size by tagging
        # more genuinely-matching phrases instead of needing one huge number
        # (single tier, no hard/soft — 2026-07-29 user request). This is the
        # sole document-level lever now — a "penalty" kind (unconditional
        # per-source deprioritization) existed briefly the same day and was
        # removed: it needed to be BOTH applied (59/2020/QH14's Điều 26
        # competing against 168/2025/NĐ-CP's Điều 76/77 on ELU177/178) AND
        # not applied (Điều 26 is the correct answer on ELS066/ELU170/
        # ELU169 — same article, different questions) — no single per-source
        # value can satisfy both, since it can't discriminate between
        # sibling articles of the same document. Boosting the correct
        # article directly instead never touches a competing source's own
        # score, so it can't cause that conflict. See
        # database.database.PRIORITY_SCORE.
        for name in kw_map.get("priority", ()):
            if _phrase_in(name, question):
                score += 15

    # Per-ARTICLE tagging — everything above tags a whole document (source_key
    # = so_ky_hieu), so it can never discriminate between two articles of the
    # SAME document (e.g. 168/2025/NĐ-CP's Điều 76/77 vs Điều 118 — a
    # document-wide priority tag lifts all three equally, confirmed live
    # 2026-07-29: Điều 118 still outscored the actually-correct Điều 77 by 5
    # points on ELU178 even after every document-level tag tried). Reuses
    # get_source_keywords_map()/set_source_keywords() completely unchanged —
    # they're already generic over any (source_type, source_key) pair — by
    # using a distinct source_type="law_article" whose source_key encodes the
    # article too ("<so_ky_hieu>#<article_number>"), per user request
    # 2026-07-29 ("làm sao để... per-article discrimination").
    so_ky_hieu  = metadata.get("so_ky_hieu", "")
    article_num = metadata.get("article_number") or metadata.get("article")
    if so_ky_hieu and article_num:
        article_kw_maps_by_source = (source_keywords_map or {}).get("law_article", {})
        lookup_keys = [f"{so_ky_hieu}#{article_num}"]
        sibling = _ENTERPRISE_LAW_SIBLING.get(so_ky_hieu)
        if sibling and article_num not in _get_enterprise_law_divergent_articles():
            lookup_keys.append(f"{sibling}#{article_num}")

        # Article-level tagging is a single +15-per-match, stacking tier
        # now (2026-07-29, user request: the old "Chấm điểm nguồn theo
        # Điều" +8-flat tier and "Tăng ưu tiên theo Điều" +15-stacking
        # tier were two separate UI sections/mechanisms doing
        # conceptually the same job — discriminate one Điều from its
        # siblings — so they're merged into one "Tăng ưu tiên theo Điều"
        # section backed by the same underlying "Chấm điểm nguồn"
        # keyword pool). Kind (primary/secondary/priority) no longer
        # matters at this scope — every name tagged to this Điều, from
        # whichever kind it was saved under (including tags made before
        # this merge), scores the same way. Merged across both Enterprise
        # Law copies (see _ENTERPRISE_LAW_SIBLING) so a tag added to either
        # copy of the same Điều helps both, instead of only whichever copy
        # happened to be tagged directly.
        article_names = set()
        for key in lookup_keys:
            article_kw_map = article_kw_maps_by_source.get(key)
            if article_kw_map:
                article_names |= (
                    article_kw_map.get("primary", set())
                    | article_kw_map.get("secondary", set())
                    | article_kw_map.get("priority", set())
                )
        for name in article_names:
            if _phrase_in(name, question):
                score += 15

    # Scenario/case-study docs are only appropriate for scenario-style
    # questions ("Anh A... có được không?") — for a general/procedure/
    # definition question, a scenario doc answering a *different*
    # specific fact pattern shouldn't outrank the general-purpose law text.
    is_scenario_doc = metadata.get("doc_type") == "scenario_qa" or \
        (metadata.get("question_type") or "").lower() == "scenario_case"
    if is_scenario_doc:
        score += 8 if intent == "scenario" else -15

    # A base entity-definition article (Điều 74 "Công ty TNHH một thành
    # viên", Điều 111 "Công ty cổ phần"...) shares near-identical
    # retrieval_keywords with its own sibling articles about that same
    # entity's rights/obligations/capital rules (Điều 76 quyền chủ sở hữu,
    # Điều 77 nghĩa vụ chủ sở hữu...), so raw keyword-overlap scoring alone
    # leaves them within 1-2 points of each other — not just for literal
    # "là gì?" questions but also for scenario questions that hinge on
    # whether a fact pattern still qualifies as that entity type ("mỗi
    # người sở hữu 50%, có còn là công ty TNHH một thành viên không?" —
    # answered by the definition article, not a rights/obligations sibling;
    # see 2026-07-29 review, ELS022). detect_query_constraints() already
    # classifies that kind of question as "scenario" rather than
    # "definition" (2026-07-28 fix, see its own docstring) specifically to
    # keep it out of unrelated procedure articles — so this boost has to
    # fire on both intents, not "definition" alone, to actually reach it.
    if intent in ("definition", "scenario") and metadata.get("doc_type", "") in _DEFINITION_DOC_TYPES:
        score += 12

    # "Thành lập công ty X như thế nào?" questions need the establishment
    # articles (hồ sơ, trình tự đăng ký, cấp GCN — see
    # _ESTABLISHMENT_DOC_TYPES below) — but those articles' retrieval_keywords
    # barely overlap with the generic "công ty/tnhh/thành viên" vocabulary
    # shared by dozens of ownership/definition articles. A small boost isn't
    # enough to close that gap, so this has to be large enough to reliably
    # land them in the top context slots.
    if intent == "procedure" and metadata.get("doc_type", "") in _ESTABLISHMENT_DOC_TYPES:
        score += 25

    # "Hộ kinh doanh" (household business) questions need Nghị định
    # 168/2025/NĐ-CP's dedicated hộ kinh doanh chapter (Điều 81-124)
    # specifically — Luật Doanh nghiệp (so_ky_hieu 59/2020/QH14 /
    # 67/VBHN-VPQH) covers company registration, a DIFFERENT legal
    # instrument that happens to share heavy vocabulary overlap ("hồ sơ",
    # "giấy chứng nhận", "cơ quan đăng ký kinh doanh") with hộ kinh doanh
    # procedures. This used to be a hardcoded phrase-check boost here; moved
    # to the admin-managed keyword table instead (2026-07-30, user request)
    # — "hộ kinh doanh" is now a priority keyword tagged (source_type=
    # "law_article") onto every Điều 81-124 chunk, the same mechanism as any
    # other admin-tagged topic. See database.database.get_source_keywords_map.
    #
    # The doc_type penalty below stays in code, though: company registration
    # (_ESTABLISHMENT_DOC_TYPES) and household-business registration are
    # different legal regimes — a hộ kinh doanh question is NEVER correctly
    # answered by one of these, so penalize directly instead of just
    # boosting the competition and hoping it's enough (measured live
    # 2026-07-29/30, ELU186: even with every priority tag on Điều 115 plus
    # this section's keyword-table boost, the correct article still lost to
    # "Điều 26, Luật Doanh nghiệp" without this penalty — company-
    # registration KB rows stack too many other additive bonuses: procedure
    # +25, dataset 3x keyword weight, exact-phrase +10). This is
    # deliberately NOT expressed as a keyword-table priority tag: the
    # keyword system is additive-boost-only by design (see
    # database.database's "penalty kind... removed" note) — a per-source
    # penalty was tried there once (2026-07-29) and reverted the same day
    # because it couldn't discriminate sibling articles that are sometimes
    # right, sometimes wrong. This rule doesn't have that failure mode: it
    # keys off doc_type (a whole document-family split, not a specific
    # so_ky_hieu/article), and hộ kinh doanh vs. company registration is a
    # structural, always-true exclusion, not a per-question judgment call.
    if _phrase_in("hộ kinh doanh", question) and metadata.get("doc_type", "") in _ESTABLISHMENT_DOC_TYPES:
        score -= 40

    # scenario_qa illustrative examples (the "BO_20_TINH_HUONG..." batch)
    # carry no so_ky_hieu/article_reference at all — structurally incapable
    # of ever producing a valid "Căn cứ pháp lý" line (build_legal_basis_line
    # returns "" for an empty article_ref). bge-m3 can rank one of these
    # essentially tied with, or ahead of, the actual correct law article when
    # the scenario happens to closely paraphrase the question (found
    # 2026-07-30, ELD040: an unrelated "Ông Khoa" nominee-ownership example
    # scored 78, edging out the correct Điều 18 chunk at 70) — winning would
    # mean the answer cites nothing at all, a worse failure mode than citing
    # a wrong-but-real Điều. Dampened as a multiplier (not a flat penalty)
    # so it still surfaces as *context* for the LLM's own reasoning when nothing
    # else is close, just doesn't win best_doc/citation purely on topical
    # overlap with zero citation to show for it.
    if metadata.get("doc_type") == "scenario_qa":
        score = round(score * 0.5)

    return score


def select_best_doc(question: str, docs, source_keywords_map: dict = None):
    q_words = [w for w in tokenize(question) if w not in STOPWORDS]
    q_counter = Counter(q_words)
    intent = detect_query_constraints(question)["intent"]

    best_doc = None
    best_score = -1

    for d in docs:
        score = _score_doc(question, d, q_counter, intent, source_keywords_map)
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

# Doc types specifically about STARTING a business — hồ sơ, trình tự đăng
# ký, cơ quan tiếp nhận, kết quả (GCN). Narrower than _DOC_TYPE_MAP's
# "procedure"/"condition" buckets above, which also catch doc types that
# have nothing to do with establishment (publication, name_prohibitions,
# asset_valuation, change_registration...) and would otherwise crowd a
# "thành lập X" question's context with irrelevant ongoing-compliance
# articles instead of the ones the question is actually asking for.
_ESTABLISHMENT_DOC_TYPES = {
    "registration_llc", "registration_partnership", "registration_jsc",
    "registration_private_enterprise", "registration_procedure",
    "erc_issuance", "erc_contents", "establishment_eligibility",
}

# Curated KB doc_types whose article defines what an entity type IS (as
# opposed to a sibling article about that same entity's rights, obligations,
# capital rules, org structure...) — verified against real content
# (2026-07-29) one by one, not guessed from the name alone: "household_business"
# was excluded here despite the name because its actual chunk content turned
# out to be about online registration procedure, not a definition.
_DEFINITION_DOC_TYPES = {
    "definition", "single_member_llc", "multi_member_llc",
    "jsc_definition", "partnership_definition", "private_enterprise",
    "social_enterprise",
}


def classify_question(question: str, best_doc=None) -> str:
    q = question.lower()

    if any(_phrase_in(p, q) for p in PROCEDURE_PATTERNS):
        return "procedure"

    if any(_phrase_in(p, q) for p in CONDITION_PATTERNS):
        return "condition"

    if any(_phrase_in(p, q) for p in DEFINITION_PATTERNS):
        return "definition"

    return "general"


# =========================
# BUILD PROMPT
# =========================

# Two response tones (see build_prompt's `voice` param): "formal" (default)
# keeps the existing strict legal/scientific wording untouched; "casual"
# layers on plain-language instructions without relaxing any of the
# citation/hallucination QUY TẮC BẮT BUỘC below — only wording style changes.
_VOICE_INSTRUCTIONS = {
    # Casual previously only said "dùng từ ngữ thông dụng" with no concrete
    # instruction to actually address the user informally — an 8B model
    # under this prompt's many strict formatting rules defaults back to
    # neutral/formal register with nothing distinctive to hold onto, so
    # casual and formal output came out reading nearly identically (user
    # report, 2026-07-29). Naming the exact pronouns ("mình"/"bạn") and
    # concrete casual connective phrases gives it something concrete to
    # actually produce instead of a vague style adjective.
    "casual": (
        "\n🗣️ VĂN PHONG: Trả lời bằng giọng thân thiện, tự nhiên, dễ hiểu, như đang giải thích cho "
        "người không rành luật. Xưng \"mình\", gọi người hỏi là \"bạn\". Dùng câu ngắn và cách diễn "
        "đạt đời thường (VD: \"nói đơn giản là...\", \"trường hợp này thì...\", \"hiểu đơn giản là...\"). "
        "Nếu bắt buộc phải dùng thuật ngữ pháp lý, giải thích ngắn gọn trong ngoặc đơn ngay sau đó. "
        "Vẫn phải giữ đúng cấu trúc Kết luận/Phân tích/Lưu ý và đúng số Điều luật như quy định — chỉ "
        "đổi cách diễn đạt, KHÔNG được bớt hay đổi nội dung pháp lý, KHÔNG được giảm độ chính xác.\n"
    ),
    "formal": (
        "\n🗣️ VĂN PHONG: Trả lời bằng văn phong nghiêm túc, chuyên nghiệp, phù hợp tư vấn pháp luật "
        "hoặc văn bản hành chính. Câu văn đầy đủ, trang trọng, dùng thuật ngữ pháp lý chính xác, "
        "KHÔNG xưng \"mình\"/gọi \"bạn\", hạn chế từ ngữ hội thoại. Ưu tiên cấu trúc kiểu \"Theo quy "
        "định tại...\", \"Được hiểu là...\", \"Có trách nhiệm...\", \"Trong phạm vi...\".\n"
    ),
}


def build_prompt(context: str, question: str, q_type: str,
                 article_ref: str = "", topic: str = "", voice: str = "formal") -> str:
    article_hint = ""
    if article_ref:
        line = article_ref
        if topic:
            line += f" — {topic}"
        article_hint = f"\nCăn cứ pháp lý: {line}\n"

    voice_hint = _VOICE_INSTRUCTIONS.get(voice, "")

    base = f"""Bạn là trợ lý pháp lý Việt Nam chuyên về Luật Doanh nghiệp.
{voice_hint}
⚠️ QUY TẮC BẮT BUỘC:
- Câu trả lời TỐI ĐA 200 từ.
- Không chào hỏi, không giải thích dư thừa.
- Không liệt kê toàn bộ văn bản luật — chỉ trả lời đúng nội dung câu hỏi.
- CĂN CỨ PHÁP LÝ phải lấy ĐÚNG từ tài liệu được cung cấp. KHÔNG tự suy diễn hay thay đổi số điều luật.
- Nếu tài liệu có metadata căn cứ pháp lý, phải sử dụng đúng điều luật đó.
- CHỈ được trích số Điều xuất hiện NGUYÊN VĂN trong phần "Tài liệu" bên dưới. Nếu không thấy số Điều liên quan trong tài liệu, hãy nói rõ là tài liệu không đề cập, KHÔNG được đoán hay lấy từ kiến thức chung.
- KHÔNG được tự đặt tên nguồn, mã văn bản, tên tác giả/giảng viên hay bất kỳ trích dẫn nào — phần nguồn tài liệu và phần "Căn cứ pháp lý" do hệ thống tự thêm vào sau, bạn không viết 2 phần đó.
- KHÔNG tự thêm tình tiết, số liệu hay điều kiện không có trong câu hỏi của người dùng.
- KHÔNG chuyển đổi loại hình doanh nghiệp nêu trong câu hỏi (vd: từ "một thành viên" sang "hai thành viên trở lên", hoặc ngược lại) khi trả lời.
- Nếu câu hỏi về công ty TNHH MỘT thành viên, KHÔNG được liệt kê "danh sách thành viên" trong hồ sơ đăng ký dù tài liệu có nhắc tới — mục này chỉ áp dụng cho công ty TNHH hai thành viên trở lên.
- KHÔNG dùng nội dung của một tình huống/case cụ thể để trả lời một câu hỏi mang tính khái quát, trừ khi người dùng thực sự hỏi về tình huống đó.
- Với bất kỳ nội dung nào trong tài liệu không áp dụng cho câu hỏi (sai loại hình doanh nghiệp, sai đối tượng...), hãy BỎ HẲN nội dung đó khỏi câu trả lời — KHÔNG liệt kê rồi ghi chú kiểu "(không áp dụng)", vì làm vậy khiến câu trả lời mất trọng tâm.
- Nếu Tài liệu bên dưới KHÔNG đủ để trả lời đúng câu hỏi, phải trả lời: "Không đủ dữ liệu để kết luận." thay vì suy diễn.
- Khi câu hỏi/tình huống nêu một con số (%, ngày, năm...) và Tài liệu nêu một ngưỡng để so sánh, PHẢI so sánh đúng hai số đó trước khi kết luận (vd: 26% ĐÃ vượt ngưỡng "từ 25% trở lên" — KHÔNG được kết luận "chưa đạt 25%").
- Nếu Tài liệu nói một nội dung "có" xuất hiện/áp dụng (kể cả khi kèm điều kiện như "nếu có"), KHÔNG suy ra ngược thành "không phải" hay "không bắt buộc" — điều kiện ("nếu có") chỉ giới hạn phạm vi áp dụng, không phủ định nội dung.
{article_hint}
Tài liệu:
{context}

Câu hỏi: {question}

"""
    STRUCT = (
        "\n\nTrả lời theo đúng cấu trúc sau (bắt buộc, mỗi mục một dòng riêng):\n"
        "**Kết luận:** [1 câu trả thẳng câu hỏi]\n"
        "**Phân tích:** [2-3 câu giải thích ngắn]\n"
        "**Lưu ý:** [1 điểm MỚI, KHÔNG lặp lại bất kỳ ý nào đã nêu ở Kết luận/Phân tích — "
        "ví dụ: so sánh với loại hình doanh nghiệp khác, trường hợp ngoại lệ, hoặc điểm dễ nhầm lẫn. "
        "Bỏ hẳn dòng này nếu không có điểm mới nào để thêm]\n"
        "Tổng cộng tối đa 200 từ. Không viết gì ngoài các mục trên."
    )
    # "general" is the default classify_question() falls back to for any
    # scenario_case question that isn't procedure/condition/definition-shaped
    # — which in practice is most of them. Without a completeness nudge here,
    # the model answers only the immediate question and drops any exception/
    # condition sitting in the retrieved context that it wasn't directly
    # asked about, even when that exception is exactly what the correct
    # answer hinges on (found live 2026-07-31, ELS020/ELS035/ELS050/ELU188:
    # all scored "Kết luận đúng/chưa sai nhưng thiếu điều kiện" — the
    # judge's most common reason string across this eval's low scores).
    if q_type not in ("procedure", "condition", "definition"):
        base += (
            "Nếu Tài liệu có nêu ngoại lệ, điều kiện áp dụng, hoặc thẩm quyền/trách nhiệm khác nhau "
            "tùy trường hợp (vd: \"trừ trường hợp...\", \"nếu được chấp thuận...\", thẩm quyền thuộc "
            "về chủ thể khác tùy giá trị/loại giao dịch), PHẢI nêu ngoại lệ/điều kiện đó trong "
            "**Kết luận** hoặc **Phân tích** — không được bỏ qua chỉ vì câu hỏi không hỏi trực tiếp."
        )

    if q_type == "procedure":
        base += (
            "Trong **Phân tích**, liệt kê các bước theo thứ tự (1, 2, 3...). "
            "Nếu câu hỏi về THÀNH LẬP doanh nghiệp và tài liệu có đề cập, "
            "cố gắng bao quát: điều kiện thành lập, hồ sơ cần nộp, trình tự/cơ quan tiếp nhận, "
            "và kết quả (Giấy chứng nhận đăng ký doanh nghiệp)."
        ) + STRUCT
    elif q_type == "condition":
        base += "Trong **Phân tích**, liệt kê ngắn gọn từng điều kiện." + STRUCT
    elif q_type == "definition":
        # "quy định về X là gì?" (one of DEFINITION_PATTERNS) — when X itself
        # is the retrieved article's own heading (e.g. "Điều 4. Giải thích từ
        # ngữ" for "quy định về giải thích từ ngữ là gì?"), a bare "nêu định
        # nghĩa" instruction let the model answer at the meta level — restate
        # what the section is about ("giải thích từ ngữ là quy định về việc
        # giải thích các từ ngữ") — instead of stating the actual rule text
        # sitting right there in the "Quy tắc pháp lý" line of the retrieved
        # document. The real content still showed up correctly, just in
        # Phân tích instead of Kết luận (found live 2026-07-31, ELK001).
        base += (
            "Trong **Kết luận**, nêu THẲNG nội dung quy định/định nghĩa lấy từ "
            "\"Quy tắc pháp lý\" trong Tài liệu — KHÔNG mô tả tài liệu/điều luật "
            "đang nói về chủ đề gì (SAI, ví dụ: \"giải thích từ ngữ là quy định về "
            "việc giải thích các từ ngữ\"); phải nêu chính nội dung quy định đó. "
            "Trong **Phân tích**, giải thích chi tiết thêm."
        ) + STRUCT
    else:
        base += STRUCT

    # A tone instruction placed once, near the top, gets crowded out by the
    # long QUY TẮC BẮT BUỘC list and the rigid Kết luận/Phân tích/Lưu ý
    # labels that follow it — confirmed live (2026-07-29): "casual" produced
    # zero uses of "mình"/"bạn", reading identically to "formal". Repeating
    # a short, concrete reminder at the very end (the last thing the model
    # reads before generating) is what actually changes small-model output.
    if voice == "casual":
        base += (
            "\n\n(Nhắc lại về văn phong: xưng \"mình\", gọi người hỏi là \"bạn\", câu văn tự nhiên, "
            "tránh lối viết như văn bản luật. Bắt buộc: phần **Lưu ý** phải mở đầu bằng cụm trực tiếp "
            "nói với người hỏi, ví dụ \"Bạn lưu ý là...\" hoặc \"Mình lưu ý bạn là...\".)"
        )

    # Same fix as the casual-voice reminder above, same reason: a rule
    # buried in the QUY TẮC BẮT BUỘC list (read once, long before the model
    # gets to the question) wasn't enough on its own — tested live
    # 2026-07-31 on the exact case this names (26% vs "từ 25% trở lên"),
    # llama-3.1-8b-instant still answered "chưa đạt 25%" for 26% with that
    # earlier version in place. Repeated here, genuinely last (after STRUCT
    # and the casual-voice block), where it's the last thing read before
    # generation starts. Widened from "\d+\s*%" to any digit run
    # (2026-07-31, ELS062) — the same failure mode hits plain-count caps
    # too: asked whether 60 investors fit a "tối đa 50 thành viên" cap, the
    # model correctly quoted the 50-member limit and then still concluded
    # the 60-person group qualified.
    if re.search(r'\d', question):
        base += (
            "\n\n(Nhắc lại: câu hỏi có nêu số liệu (%, số người, số ngày...). Trước khi viết Kết luận, "
            "tự so sánh con số đó với ngưỡng nêu trong Tài liệu — vd: 26% so với ngưỡng \"từ 25% trở "
            "lên\" nghĩa là ĐÃ đạt/vượt ngưỡng; 60 so với giới hạn \"tối đa 50\" nghĩa là ĐÃ VƯỢT giới "
            "hạn, không phải \"phù hợp\"/\"đáp ứng điều kiện\". Viết phép so sánh đó ra trước khi kết luận.)"
        )
    return base


# =========================
# BUILD CITATION
# =========================
# Canonical URL for the consolidated Enterprise Law text — used whenever a
# document is relabeled to "VBHN 67/VBHN-VPQH" below (see build_citation).
# Some indexed chunks carry so_ky_hieu="59/2020/QH14" with a source_url that
# points at the *original* law's lược đồ page instead of the VBHN 67 text,
# which — if printed as-is under the "VBHN 67" label — shows a URL that
# doesn't match the document name displayed to the user.
_VBHN_67_URL = "https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-67-vbhn-vpqh-45865.htm"


def _detect_khoan(content: str, answer: str) -> str | None:
    """A law chunk is indexed as a whole Điều, but the answer usually only
    draws on one khoản inside it (nhận xét 25/7 flagged citing bare "Điều 74"
    when the answer only used khoản 1's definition sentence). Split the
    chunk into its numbered khoản and, only when the answer's wording
    overlaps one of them more than any other, name that khoản instead of
    the whole article — ambiguous cases fall back to citing the article as
    a whole rather than guessing."""
    khoan_matches = list(re.finditer(r'(?:^|\n)(\d+)\.\s+(.+?)(?=\n\d+\.\s|\Z)', content, re.DOTALL))
    if len(khoan_matches) < 2:
        return None
    answer_words = {w for w in tokenize(answer) if w not in STOPWORDS}
    if not answer_words:
        return None
    best_n, best_score, second_score = None, -1, -1
    for m in khoan_matches:
        words = {w for w in tokenize(m.group(2)) if w not in STOPWORDS}
        score = len(answer_words & words)
        if score > best_score:
            best_n, second_score, best_score = m.group(1), best_score, score
        elif score > second_score:
            second_score = score
    if best_n is not None and best_score > 0 and best_score > second_score:
        return best_n
    return None


def _resolve_primary_article_ref(meta: dict, content: str = "", answer: str = "") -> str:
    """Primary article reference, upgraded to "Khoản N Điều X" when the
    answer clearly drew from just one khoản inside the article (see
    _detect_khoan). Shared by build_citation() and build_legal_basis_line()
    so Nguồn chính and Căn cứ pháp lý can never disagree with each other —
    both are derived from best_doc's own metadata, never from LLM output.
    Mixing in article numbers the LLM happened to write let a number it
    recalled (or invented) get credited to a source document that never
    mentioned it, and in whatever order the answer happened to mention it
    (e.g. "Điều 76; Điều 74" when the actual primary basis was Điều 74)."""
    article_ref = meta.get("article_reference", "")
    if content and answer and article_ref and ";" not in article_ref \
            and not article_ref.lower().startswith("khoản"):
        khoan_n = _detect_khoan(content, answer)
        if khoan_n:
            article_ref = f"Khoản {khoan_n} {article_ref}"
    return article_ref


def build_legal_basis_line(meta: dict, secondary_docs=None, content: str = "", answer: str = "") -> str:
    """Builds the answer body's "Căn cứ pháp lý" line entirely from best_doc's
    own metadata — replaces an earlier version where the LLM was asked to
    write this line itself from a prompt hint, which in practice could drift
    from Nguồn chính (LLM wrote "Điều 74" while Nguồn chính correctly said
    "Điều 21" — nhận xét 25/7 follow-up).

    Only the primary article — secondary_docs is accepted for call-site
    compatibility with build_citation() but deliberately unused here. An
    earlier version blended in up to 3 secondary_docs articles here (reasoning
    that a "thành lập X" question often needs hồ sơ + trình tự + định nghĩa
    together), but that applied indiscriminately to plain single-article
    lookup questions too ("Theo Luật Doanh nghiệp 2020, quy định về X là gì?")
    — for those, listing 2-3 tangential articles alongside the one actually
    asked about reads as imprecise rather than thorough, which is exactly
    what both an independent human review and the LLM judge flagged
    (2026-07-29 review: 17/20 low-legal_accuracy rows correctly named the
    right article here but buried it among tangential ones). Secondary/
    related articles still surface separately in build_citation()'s own
    "📎 Nguồn tham khảo" footer — this line just no longer duplicates them."""
    so_ky_hieu = meta.get("so_ky_hieu", "")
    if so_ky_hieu and not is_known_citation_source(so_ky_hieu):
        so_ky_hieu = ""

    primary_ref = _resolve_primary_article_ref(meta, content, answer)
    if not primary_ref:
        return ""

    line = primary_ref
    law_name = f"{meta.get('loai_van_ban', '')} {so_ky_hieu}".strip()
    if "67/VBHN" in so_ky_hieu or "59/2020" in so_ky_hieu:
        law_name = "Văn bản hợp nhất Luật Doanh nghiệp số 67/VBHN-VPQH năm 2025"
    if law_name:
        line += f" — {law_name}"
    return line


def build_citation(meta: dict, secondary_docs=None, content: str = "", answer: str = "") -> str:
    article_ref = _resolve_primary_article_ref(meta, content, answer)
    topic = meta.get("topic", "")
    so_ky_hieu = meta.get("so_ky_hieu", "")
    loai = meta.get("loai_van_ban", "")
    source_url = meta.get("source_url", "")

    # Refuse to print a document code that isn't actually indexed in chroma_db
    # right now — catches stale/fabricated so_ky_hieu before it reaches the user.
    if so_ky_hieu and not is_known_citation_source(so_ky_hieu):
        so_ky_hieu = ""
        source_url = ""

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
        # Relabeled to the VBHN 67 text — the URL must point there too,
        # not at whichever page this specific chunk's metadata happened to
        # carry (see _VBHN_67_URL docstring above).
        source_url = _VBHN_67_URL
    if law_name:
        parts.append(law_name)

    citation = " — ".join(parts) if parts else meta.get("nguon_thu_thap", "")
    result = f"\n\n📖 Nguồn chính: {citation}"
    if source_url and ("vbpl.vn" in source_url or "congbao.chinhphu.vn" in source_url):
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
def _is_out_of_scope(question: str, out_of_scope_keywords: list = None) -> bool:
    q = question.lower()
    keywords = out_of_scope_keywords if out_of_scope_keywords else OUT_OF_SCOPE_KEYWORDS
    if not any(kw in q for kw in keywords):
        return False
    # An OUT_OF_SCOPE_KEYWORDS hit alone over-blocks: "C đang bị truy cứu
    # trách nhiệm hình sự nhưng muốn thành lập doanh nghiệp tư nhân..." is a
    # genuine Enterprise Law eligibility question (Điều 17) that only
    # mentions "hình sự" as a disqualifying condition, not a criminal-law
    # question. Same for "gia đình góp vốn" (family funding a company) and
    # "góp vốn bằng quyền sử dụng đất" (land-use rights as capital) — both
    # got wrongly blocked in the 2026-07-25 eval review. If the question also
    # carries a clear Enterprise Law signal, treat it as in-scope; the answer
    # pipeline still falls back to "Không đủ dữ liệu để kết luận." if nothing
    # relevant is actually retrieved.
    return not any(sig in q for sig in _BUSINESS_CONTEXT_SIGNALS)


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


# Returned by ask_rag()'s catch-all below on any unhandled failure (most
# commonly a Groq connection/rate-limit error with no retry at this layer,
# unlike evaluate_engine._llm_score). Exported as a constant rather than a
# bare string literal so evaluate_engine can recognize it and route the
# question into the connection-error bucket instead of scoring "❌ Lỗi hệ
# thống." as if it were a real RAG answer (2026-07-28 eval review — 18/55
# low-score rows in one run were this placeholder, not genuine bad answers).
RAG_SYSTEM_ERROR_MESSAGE = "❌ Lỗi hệ thống."


def ask_rag(question: str, return_debug: bool = False, voice: str = "formal"):
    try:
        question = str(question)

        # Admin/teacher-tagged keyword boosts (see database.database.keyword /
        # source_keyword) — loaded once per question, small table, no caching
        # needed. See _score_doc() for how this is used.
        from database.database import get_source_keywords_map, get_active_out_of_scope_keywords
        try:
            source_keywords_map = get_source_keywords_map()
        except Exception:
            source_keywords_map = {}
        try:
            out_of_scope_keywords = get_active_out_of_scope_keywords()
        except Exception:
            out_of_scope_keywords = None  # _is_out_of_scope() falls back to OUT_OF_SCOPE_KEYWORDS

        # Pre-check: question outside business law scope
        if _is_out_of_scope(question, out_of_scope_keywords):
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
        docs = retrieve_docs(question, better_q, source_keywords_map)

        if not docs:
            return "⚠️ Không tìm thấy thông tin đủ liên quan trong cơ sở dữ liệu. Vui lòng hỏi rõ hơn hoặc kiểm tra câu hỏi có thuộc phạm vi Luật Doanh nghiệp không."

        # STEP 2.1: drop documents whose business-entity type conflicts with
        # the question (e.g. "TNHH một thành viên" question vs "TNHH hai
        # thành viên trở lên" document) — see filter_compatible_docs above.
        docs = filter_compatible_docs(question, docs)
        if not docs:
            return (
                "⚠️ Hệ thống chưa tìm thấy nguồn phù hợp đúng loại hình doanh nghiệp "
                "và đúng nội dung câu hỏi. Vui lòng cung cấp câu hỏi chi tiết hơn."
            )

        # STEP 3: rerank — use the ORIGINAL question, not the LLM-rewritten
        # one: rewrite_query() can drop the exact qualifier ("một thành viên"
        # vs "hai thành viên") that distinguishes otherwise near-identical docs.
        best_doc = select_best_doc(question, docs, source_keywords_map)
        if not best_doc:
            return "❌ Không đủ dữ liệu để trả lời câu hỏi này."

        # STEP 4: rank the full candidate-document pool (best_doc first) —
        # used both to fill "extra context" docs and, per STEP 5-8.1's retry
        # loop below, as fallback primary documents.
        # (Retrieval order puts kw/semantic hits first and any
        # procedure-intent augmented docs last — a positional docs[:3]
        # would silently drop exactly the hồ sơ/trình tự articles the
        # procedure-intent boost above was added to surface.)
        q_counter = Counter([w for w in tokenize(question) if w not in STOPWORDS])
        intent = detect_query_constraints(question)["intent"]
        ranked_extra = sorted(
            (d for d in docs if d.page_content != best_doc.page_content),
            key=lambda d: _score_doc(question, d, q_counter, intent, source_keywords_map),
            reverse=True,
        )
        if intent == "procedure":
            # Fill the extra context slots with establishment-doc-type docs
            # first (see _ESTABLISHMENT_DOC_TYPES) so a "thành lập X"
            # question's hồ sơ/trình tự/cấp GCN articles all make it into
            # context, even when a single one of them still scores below an
            # ownership/rights article that shares more raw vocabulary.
            procedure_first = [
                d for d in ranked_extra
                if d.metadata.get("doc_type", "") in _ESTABLISHMENT_DOC_TYPES
            ]
            others = [d for d in ranked_extra if d not in procedure_first]
            ranked_extra = procedure_first + others
        all_ranked = [best_doc] + ranked_extra

        # STEP 5-8.1: generate an answer from the best-ranked doc; if
        # validate_answer_citations() rejects it (the answer names an "Điều N"
        # never actually seen in that doc's context — usually the LLM
        # inventing/misremembering a number), retry with the 2nd- and then
        # 3rd-best-ranked doc as primary before giving up. Each attempt
        # rebuilds its own context/classification/prompt around its primary doc.
        winner = None
        for primary in all_ranked[:3]:
            extra_docs   = [d for d in all_ranked if d is not primary][:3]
            context_parts = [primary.page_content] + [d.page_content for d in extra_docs]
            context = "\n\n---\n\n".join(context_parts)[:3000]

            q_type      = classify_question(question, primary)
            article_ref = primary.metadata.get("article_reference", "")
            doc_topic   = primary.metadata.get("topic", "")
            prompt      = build_prompt(context, question, q_type, article_ref, doc_topic, voice=voice)

            answer = _llm_invoke_with_retry(prompt)
            answer = clean_answer(answer)

            if validate_answer_citations(answer, context):
                winner = (primary, extra_docs, context, answer, q_type)
                break

        if not winner:
            return (
                "⚠️ Hệ thống đã thử nhiều tài liệu liên quan nhưng vẫn không tìm được "
                "câu trả lời có căn cứ pháp lý khớp với tài liệu truy xuất. "
                "Vui lòng đặt lại câu hỏi cụ thể hơn."
            )

        best_doc, extra_docs, context, answer, q_type = winner

        # STEP 8.2: splice in "Căn cứ pháp lý" — built the same way as the
        # citation footer below (best_doc/extra_docs metadata only), never
        # written by the LLM itself (see build_legal_basis_line docstring).
        legal_basis = build_legal_basis_line(best_doc.metadata, extra_docs, best_doc.page_content, answer)
        if legal_basis:
            marker = "**Phân tích:**"
            idx = answer.find(marker)
            if idx != -1:
                answer = answer[:idx] + f"**Căn cứ pháp lý:** {legal_basis}\n" + answer[idx:]

        # STEP 9: citation — built strictly from best_doc/extra_docs metadata,
        # never from article numbers appearing in the LLM-generated answer.
        final_answer = answer + build_citation(best_doc.metadata, extra_docs, best_doc.page_content, answer)

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
        return RAG_SYSTEM_ERROR_MESSAGE


# Runs once at import time, at the true bottom of the module so every name it
# needs (_ENTITY_AGNOSTIC_DOC_TYPES, detect_entity_type) is already bound —
# see the note next to the earlier startup-call block above.
backfill_entity_type_tags()