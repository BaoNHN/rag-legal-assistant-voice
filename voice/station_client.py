"""
voice/station_client.py
Thin adapter over the installable `clone-voice-client` package (see
D:\\hoc\\project\\clone-voice-client) — the actual HTTP logic for talking to
clone-voice-station (the standalone voice/STT/TTS/RVC service, see
D:\\hoc\\project\\clone-voice-station) now lives in that package so any new
AI-assistant project can `pip install` it instead of copying a file around.
This module exists only so app.py's existing `station_client.xxx(...)` call
sites keep working unchanged.

This app authenticates itself to the station with a shared API key (read once
at import time from voice_station_key.txt) and identifies each end user as an
opaque external_user_id (str(session["user_id"])) — the station has no login
of its own and trusts this app to have already checked who's logged in /
who's admin. All functions degrade gracefully when the station is
unreachable — see clone_voice_client.VoiceStationClient's own docstring.
"""

import json
import os
import shutil
import uuid

from clone_voice_client import (
    VoiceStationClient, VoiceStationError, MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "voice_station_key.txt")

# Local STT needs a working ffmpeg to decode browser mic audio (webm/opus) --
# on this machine, the conda-forge ffmpeg on PATH in the shared rag_env fails
# to launch at all (STATUS_ENTRYPOINT_NOT_FOUND, a DLL conflict with its
# dynamically-linked build; confirmed for real via `ffmpeg -version` producing
# no output and no error). clone-voice-station already ships a static ffmpeg
# binary that sidesteps this (its own voice/stt.py, bin/ffmpeg.exe) -- reuse
# that instead of bundling a second copy here. clone_voice_client's local_stt
# module reads CLONE_VOICE_FFMPEG_DIR at its own import time (lazy, only when
# transcribe_local() is first called), so this must be set before then; doing
# it here at station_client's own import time (this module is imported once,
# early, at app startup) guarantees that. A no-op if the sibling repo isn't
# at this relative path (e.g. a deployment without clone-voice-station
# checked out alongside) -- local STT then falls back to whatever ffmpeg is
# on PATH, same as before this fix.
_SIBLING_FFMPEG_DIR = os.path.join(os.path.dirname(BASE_DIR), "clone-voice-station", "bin")
if os.path.isfile(os.path.join(_SIBLING_FFMPEG_DIR, "ffmpeg.exe")):
    os.environ.setdefault("CLONE_VOICE_FFMPEG_DIR", _SIBLING_FFMPEG_DIR)

# Live-adjustable clone-voice-station URL (added 2026-08-12): for a local demo
# where clone-voice-station is tunnelled through ngrok (a different URL every
# run), editing VOICE_STATION_URL and restarting the whole app is slower than
# a manager just pasting the new URL into /admin/voice_models. This file, if
# present, wins over the env var at startup; see get_station_url()/
# set_station_url() below for the runtime path -- VoiceStationClient.base_url
# is a plain mutable attribute read fresh on every call, so updating it here
# takes effect immediately, no restart needed.
STATION_URL_OVERRIDE_PATH = os.path.join(BASE_DIR, "voice_station_url_override.txt")


def _load_station_url_override():
    if os.path.isfile(STATION_URL_OVERRIDE_PATH):
        with open(STATION_URL_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            url = f.read().strip()
            if url:
                return url
    return None


_client = VoiceStationClient.from_key_file(
    KEY_PATH,
    base_url=_load_station_url_override() or os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090"),
    request_timeout=int(os.getenv("VOICE_STATION_TIMEOUT", "15")),
    # Local RVC fallback (no Colab endpoint configured) reloads the model fresh
    # every call via subprocess -- measured ~34s for a short phrase on this
    # hardware, so the old 30s default was cutting it off just before it
    # finished. 90s covers that with real margin for longer answers.
    speak_timeout=int(os.getenv("VOICE_STATION_SPEAK_TIMEOUT", "90")),
    upload_timeout=int(os.getenv("VOICE_STATION_UPLOAD_TIMEOUT", "30")),
)


def get_own_api_key() -> str:
    """The key this app authenticates itself to the station with — also used
    to verify an inbound webhook call really came from the station, since it
    echoes this same key back in X-Api-Key (see app.py's POST /voice/webhook)."""
    return _client.get_own_api_key()


def is_available() -> bool:
    return _client.is_available()


def get_station_url() -> str:
    return _client.base_url


def set_station_url(url: str) -> dict:
    """Points this process at a different clone-voice-station instance
    (e.g. a fresh ngrok URL for a local demo) without a restart, and persists
    the choice so the next startup picks it up too."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("URL không được để trống.")
    _client.base_url = url
    with open(STATION_URL_OVERRIDE_PATH, "w", encoding="utf-8") as f:
        f.write(url)
    return {"url": url, "available": _client.is_available()}


# ── Scripts / consent / profiles ────────────────────────────────────────────
def get_scripts() -> list:
    return _client.get_scripts()


def has_voice_consent(external_user_id: str) -> bool:
    return _client.has_voice_consent(external_user_id)


def record_voice_consent(external_user_id: str):
    _client.record_voice_consent(external_user_id)


def list_voice_profiles(external_user_id: str) -> list:
    return _client.list_voice_profiles(external_user_id)


def create_voice_profile(external_user_id: str, name: str) -> int:
    return _client.create_voice_profile(external_user_id, name)


def update_voice_profile(profile_id: int, external_user_id: str, name: str = None, is_default: bool = None):
    return _client.update_voice_profile(profile_id, external_user_id, name=name, is_default=is_default)


def delete_voice_profile(profile_id: int, external_user_id: str) -> dict:
    return _client.delete_voice_profile(profile_id, external_user_id)


def get_voice_profile_status(profile_id: int, external_user_id: str) -> dict:
    return _client.get_voice_profile_status(profile_id, external_user_id)


# ── Samples ──────────────────────────────────────────────────────────────────
def upload_voice_sample(profile_id: int, external_user_id: str, script_id: str,
                         filename: str, content: bytes) -> dict:
    return _client.upload_voice_sample(profile_id, external_user_id, script_id, filename, content)


def list_voice_samples(profile_id: int, external_user_id: str) -> list:
    return _client.list_voice_samples(profile_id, external_user_id)


def delete_voice_sample(profile_id: int, sample_id: int, external_user_id: str):
    return _client.delete_voice_sample(profile_id, sample_id, external_user_id)


def train_voice_profile(profile_id: int, external_user_id: str) -> dict:
    return _client.train_voice_profile(profile_id, external_user_id)


# ── Transcribe (STT — input half of the voice loop) ─────────────────────────
def transcribe(filename: str, content: bytes, mime: str = None, language: str = "vi") -> dict:
    return _client.transcribe(filename, content, mime=mime, language=language)


# ── Local STT (added 2026-08-12) ─────────────────────────────────────────────
# Lets an admin download a .stt-pack.zip from clone-voice-station's STT Lab
# (/stt-lab, once an adapter finishes training) and run Whisper for this
# app's own mic transcription right in this process, via
# clone_voice_client.local_stt / VoiceStationClient.transcribe_local() --
# instead of every request always calling clone-voice-station over the
# network. Requires `pip install clone-voice-client[local]` (openai-whisper +
# torch + transformers + peft); if that extra isn't installed,
# transcribe_local() below raises VoiceStationError with an actionable
# message rather than crashing. Multiple packs can be uploaded; at most one
# is "active" at a time (or none, for plain untrained-Whisper local mode).
STT_PACKS_DIR = os.path.join(BASE_DIR, "stt_local_packs")
STT_PACKS_INDEX_PATH = os.path.join(STT_PACKS_DIR, "index.json")

_loaded_pack_cache = {}  # pack_id -> local_stt.load_pack() result, lazy (avoids re-extracting the zip every request)


def _read_stt_packs_index() -> dict:
    if os.path.isfile(STT_PACKS_INDEX_PATH):
        with open(STT_PACKS_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"packs": [], "active_id": None, "local_enabled": False}


def _write_stt_packs_index(index: dict):
    os.makedirs(STT_PACKS_DIR, exist_ok=True)
    with open(STT_PACKS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_stt_local_packs() -> dict:
    index = _read_stt_packs_index()
    for p in index["packs"]:
        p["is_warm"] = p["id"] in _loaded_pack_cache
    return index


def is_stt_local_mode_enabled() -> bool:
    return _read_stt_packs_index().get("local_enabled", False)


def set_stt_local_mode(enabled: bool):
    index = _read_stt_packs_index()
    index["local_enabled"] = bool(enabled)
    _write_stt_packs_index(index)


def upload_stt_local_pack(filename: str, content: bytes) -> dict:
    """Saves an uploaded .stt-pack.zip and validates it can actually be loaded
    (catches a corrupt/wrong zip immediately instead of only failing the
    first time someone tries to transcribe with it)."""
    from clone_voice_client import local_stt

    os.makedirs(STT_PACKS_DIR, exist_ok=True)
    pack_id  = uuid.uuid4().hex[:12]
    zip_path = os.path.join(STT_PACKS_DIR, f"{pack_id}.stt-pack.zip")
    with open(zip_path, "wb") as f:
        f.write(content)

    try:
        pack = local_stt.load_pack(zip_path)
    except Exception as e:
        os.remove(zip_path)
        raise ValueError(f"File pack không hợp lệ: {e}")

    entry = {
        "id": pack_id,
        "label": filename or f"pack-{pack_id}",
        "tier": pack["tier"],
        "base_model": pack["base_model"],
        "zip_path": zip_path,
    }
    index = _read_stt_packs_index()
    index["packs"].append(entry)
    if not index.get("active_id"):
        index["active_id"] = pack_id
    _write_stt_packs_index(index)
    return entry


def delete_stt_local_pack(pack_id: str):
    index = _read_stt_packs_index()
    entry = next((p for p in index["packs"] if p["id"] == pack_id), None)
    if not entry:
        raise ValueError("Không tìm thấy pack.")
    if os.path.isfile(entry["zip_path"]):
        os.remove(entry["zip_path"])
    extracted_dir = f"{entry['zip_path']}_extracted"
    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir, ignore_errors=True)
    index["packs"] = [p for p in index["packs"] if p["id"] != pack_id]
    if index.get("active_id") == pack_id:
        index["active_id"] = index["packs"][0]["id"] if index["packs"] else None
    _write_stt_packs_index(index)
    _loaded_pack_cache.pop(pack_id, None)


def set_active_stt_local_pack(pack_id):
    index = _read_stt_packs_index()
    if pack_id is not None and not any(p["id"] == pack_id for p in index["packs"]):
        raise ValueError("Không tìm thấy pack.")
    index["active_id"] = pack_id
    _write_stt_packs_index(index)


def _get_active_pack_loaded():
    """Returns the loaded pack dict for transcribe_local(), or None if no pack
    is active (falls through to plain untrained Whisper, no hotword bias)."""
    from clone_voice_client import local_stt

    index     = _read_stt_packs_index()
    active_id = index.get("active_id")
    if not active_id:
        return None
    if active_id in _loaded_pack_cache:
        return _loaded_pack_cache[active_id]
    entry = next((p for p in index["packs"] if p["id"] == active_id), None)
    if not entry:
        return None
    pack = local_stt.load_pack(entry["zip_path"])
    _loaded_pack_cache[active_id] = pack
    return pack


def transcribe_local(filename: str, content: bytes, mime: str = None, language: str = "vi") -> dict:
    pack = _get_active_pack_loaded()
    return _client.transcribe_local(filename, content, mime=mime, language=language, pack=pack)


# ── Speak (TTS + optional RVC) ───────────────────────────────────────────────
def speak(text: str, external_user_id: str, profile_id: int = None) -> dict:
    return _client.speak(text, external_user_id, profile_id)


# ── Admin (client-wide, gated by is_admin() in app.py before calling these) ─────
def list_all_voice_profiles() -> list:
    return _client.list_all_voice_profiles()


def admin_retrain_voice_model(profile_id: int) -> dict:
    return _client.admin_retrain_voice_model(profile_id)


def admin_disable_voice_model(profile_id: int) -> dict:
    return _client.admin_disable_voice_model(profile_id)


def admin_delete_voice_model(profile_id: int) -> dict:
    return _client.admin_delete_voice_model(profile_id)


# ── Notifications: manager-triggered delete/disable events ──────────────────────
def register_webhook(webhook_url: str):
    _client.register_webhook(webhook_url)


def poll_undelivered_notifications(external_user_id: str) -> list:
    return _client.poll_undelivered_notifications(external_user_id)


def ack_notification(notification_id: int):
    _client.ack_notification(notification_id)
