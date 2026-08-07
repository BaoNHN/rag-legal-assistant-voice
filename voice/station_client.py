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

import os

from clone_voice_client import (
    VoiceStationClient, VoiceStationError, MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "voice_station_key.txt")

_client = VoiceStationClient.from_key_file(
    KEY_PATH,
    base_url=os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090"),
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


def get_rvc_endpoint() -> dict:
    return _client.get_rvc_endpoint()


def set_rvc_endpoint(endpoint: str) -> dict:
    return _client.set_rvc_endpoint(endpoint)


# ── Notifications: manager-triggered delete/disable events ──────────────────────
def register_webhook(webhook_url: str):
    _client.register_webhook(webhook_url)


def poll_undelivered_notifications(external_user_id: str) -> list:
    return _client.poll_undelivered_notifications(external_user_id)


def ack_notification(notification_id: int):
    _client.ack_notification(notification_id)
