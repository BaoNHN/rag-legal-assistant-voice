# groq_keys.py
# Shared multi-key rotation for every Groq call site in the app.
#
# groqkey.txt historically held exactly one key. As of 2026-07-29 it can hold
# several, semicolon-separated ("gsk_AAA;gsk_BBB;gsk_CCC") — added so one
# account hitting its rate/quota limit doesn't stall every LLM call (RAG
# answer generation in rag_engine.py, LLM-judge scoring in evaluate_engine.py)
# for the rest of the run. Both call sites share this module's rotation
# pointer so a key already marked limited by one isn't immediately retried by
# the other.
import os
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "groqkey.txt")

_lock = threading.Lock()
_idx = 0


def _load() -> list:
    if not os.path.exists(KEY_PATH):
        return []
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    return [k.strip() for k in raw.split(";") if k.strip()]


_keys = _load()


def get_keys() -> list:
    """All configured keys, in rotation order. Re-reads groqkey.txt each call
    so editing the file takes effect on the next request without a restart."""
    global _keys
    fresh = _load()
    if fresh:
        _keys = fresh
    return list(_keys)


def current_key() -> str:
    keys = get_keys()
    if not keys:
        raise RuntimeError("groqkey.txt has no usable API key")
    with _lock:
        return keys[_idx % len(keys)]


def rotate_key() -> str:
    """Advance to the next key (wrapping around) and return it."""
    global _idx
    keys = get_keys()
    if not keys:
        raise RuntimeError("groqkey.txt has no usable API key")
    with _lock:
        _idx = (_idx + 1) % len(keys)
        return keys[_idx]


_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "quota", "too many requests")


def is_rate_limit_error(e: Exception) -> bool:
    """True for a Groq rate-limit/quota error — checked both by type (the
    groq client raises RateLimitError, a distinct exception a caller should
    never accidentally string-match past) and by message text (langchain_groq
    sometimes wraps it, losing the original exception type)."""
    try:
        from groq import RateLimitError
        if isinstance(e, RateLimitError):
            return True
    except ImportError:
        pass
    return any(m in str(e).lower() for m in _RATE_LIMIT_MARKERS)
