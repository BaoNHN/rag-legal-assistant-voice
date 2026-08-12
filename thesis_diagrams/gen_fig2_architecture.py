import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

fig, ax = plt.subplots(figsize=(24, 15.5))
ax.set_xlim(0, 24)
ax.set_ylim(-2.6, 15)
ax.axis("off")

FS_TITLE = 11.5
FS_BODY = 9.3
FS_LABEL = 8.6

def cluster(x, y, w, h, title, fc, ec, title_fs=FS_TITLE):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                          linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=1)
    ax.add_patch(box)
    ax.text(x + 0.18, y + h - 0.32, title, fontsize=title_fs, fontweight="bold", color=ec, va="top")
    return box

def box(x, y, w, h, text, fc, ec="#555555", fs=FS_BODY, zorder=3, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
             wrap=True, fontweight=weight, zorder=zorder + 1)
    return b

def arrow(p1, p2, label=None, style="-|>", color="#333333", dashed=False, lw=1.3,
          label_pos=0.5, fs=FS_LABEL, curve=0.0, label_off=(0, 0.12)):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                         color=color, zorder=2, linestyle="--" if dashed else "-",
                         connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + label_off[0]
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_off[1]
        ax.text(mx, my, label, fontsize=fs, ha="center", va="center", color=color,
                 zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

# ---------------------------------------------------------------- clusters
cluster(0.4, 11.7, 5.0, 2.9, "1. Client", "#FDEBD3", "#E67E22")
cluster(9.2, 12.3, 4.4, 2.3, "2. Cloud LLM", "#FADBD8", "#C0392B")
cluster(18.6, 11.7, 5.0, 2.9, "3. Manager (operator)", "#FDEBD3", "#E67E22")

cluster(0.4, 0.6, 11.6, 10.6, "4. rag-legal-assistant \u2014 FastAPI service (port 8000)", "#EBF5FB", "#2E86C1")
cluster(12.4, 0.6, 11.2, 10.6, "5. clone-voice-station \u2014 FastAPI service (port 8090)", "#FEF5E7", "#CA6F1E")

cluster(0.4, -2.4, 11.6, 2.7, "6. Build-time RAG indexing pipeline (one-off)", "#F4ECF7", "#7D3C98")
cluster(12.4, -2.4, 11.2, 2.7, "7. Colab GPU instance \u2014 Flask, ephemeral, behind cloudflared tunnel", "#FDEBD3", "#B9770E")

# ---------------------------------------------------------------- 1. client
box(0.8, 12.75, 1.9, 1.1, "User", "#F5CBA7")
box(2.9, 12.05, 2.3, 1.8, "Browser\nchat UI + mic\n(MediaRecorder)\nvoice-profile &\nadmin pages", "#FAD7A0", fs=8.0)

# ---------------------------------------------------------------- 2. cloud LLM
box(9.6, 12.65, 3.6, 1.2, "Groq API\nLlama-3.1-8B-Instant", "#F1948A", fs=8.6)

# ---------------------------------------------------------------- 3. manager
box(19.0, 12.75, 1.9, 1.1, "Manager", "#F5CBA7")
# (manager -> dashboard arrow drawn later, target inside clone-voice-station)

# ---------------------------------------------------------------- 4. rag-legal-assistant internals
box(0.8, 9.35, 10.8, 1.05, "FastAPI routes (app.py)\n/get  /voice/*  /admin/voice_models  /admin/station_url", "#AED6F1", fs=8.6, weight="bold")

box(0.8, 6.75, 6.2, 2.1, "RAG engine\n(engine/rag_engine.py)\nrewrite \u2192 retrieve \u2192 rerank \u2192 prompt", "#A9DFBF", fs=8.8)
box(7.3, 7.75, 4.3, 1.1, "engine/groq_keys.py\nmulti-key rotation\n(groqkey.txt, ';'-separated)", "#A9DFBF", fs=8.0)
box(7.3, 6.55, 4.3, 1.0, "voice/station_client.py\nthin adapter", "#D7BDE2", fs=8.3)

box(0.8, 4.85, 3.0, 1.5, "ChromaDB\nlegal-doc vectors\n(BGE-small-en-v1.5)", "#F7F9F9", fs=8.2)
box(4.1, 4.85, 3.0, 1.5, "SQLite chat.db\nusers . chats . messages", "#F7F9F9", fs=8.2)

box(7.3, 4.85, 4.3, 1.5, "clone_voice_client SDK\n(installable pip package)\nVoiceStationClient", "#D2B4DE", fs=8.3, weight="bold")

box(0.8, 1.1, 10.8, 3.2,
    "consumes SDK for:  consent \u2022 profile CRUD \u2022 sample upload \u2022 train \u2022 status\n"
    "/api/transcribe \u2022 /api/speak \u2022 admin retrain/disable/delete \u2022 station_url\n"
    "webhook self-registration \u2022 notification polling",
    "#EAF2F8", ec="#AED6F1", fs=8.4)

# ---------------------------------------------------------------- 5. clone-voice-station internals
box(12.8, 9.35, 10.4, 1.05, "Client API (X-Api-Key, scoped by client_id + external_user_id)\n/api/transcribe  /api/speak  /api/profiles  /api/consent  /api/webhook", "#AED6F1", fs=8.4, weight="bold")

box(12.8, 7.55, 5.0, 1.55, "Manager dashboard\ncookie session + CSRF\nclients \u2022 cross-client profiles\nRVC config \u2022 realism test", "#D7BDE2", fs=8.0)
box(18.1, 7.55, 5.1, 1.55, "engine/voice_engine.py\nspeak_text(): TTS \u2192 RVC \u2192\nAI-disclosure prefix +\nprovenance watermark", "#A9DFBF", fs=8.0)

box(12.8, 5.55, 5.0, 1.7, "voice/stt.py \u2014 local Whisper-small (fallback)\nvoice/tts.py \u2014 edge-TTS\n(vi-VN-HoaiMyNeural / NamMinhNeural)\nvoice/rvc_client.py \u2014 Colab HTTP client", "#AED6F1", fs=7.7)
box(18.1, 5.55, 5.1, 1.7, "SQLite voice_station.db\nclients \u2022 voice_profiles \u2022 voice_samples\nvoice_consent \u2022 settings \u2022 managers \u2022 notifications", "#F7F9F9", fs=7.7)

box(12.8, 3.85, 10.4, 1.3, "voice_storage/ (local backup of trained .pth/.index models)   \u2022   voice_samples/ (raw recordings by profile_id)", "#F7F9F9", fs=8.0)

box(12.8, 1.1, 10.4, 2.3,
    "MIN_TRAIN_SAMPLES = 5   \u2022   MAX_CLONED_VOICES_PER_USER = 2\n"
    "multi-tenant: every request scoped by client API key + external_user_id\n"
    "\u2014 rag-legal-assistant is the first seeded client, not the only one",
    "#FEF9E7", ec="#F7DC6F", fs=8.2)

# ---------------------------------------------------------------- 6. build-time pipeline
box(0.8, -2.0, 2.6, 1.5, "HuggingFace Hub\nvietnamese-legal-\ndocuments", "#F5EEF8", fs=7.3)
box(3.7, -2.0, 3.0, 1.5, "build_db.py\nmetadata-only\nsector + status filter", "#D7BDE2", fs=7.5, weight="bold")
box(7.0, -2.0, 2.2, 1.5, "BGE embeddings\nbge-small-en-v1.5", "#F5EEF8", fs=7.3)
box(9.5, -2.0, 2.2, 1.5, "ChromaDB\npersisted to\nchroma_db/", "#F5EEF8", fs=7.3)

# ---------------------------------------------------------------- 7. colab
box(12.8, -2.0, 3.4, 1.5, "PhoWhisper-large\n(STT, primary)", "#FAD7A0", fs=7.9)
box(16.4, -2.0, 3.4, 1.5, "F5-TTS-Vietnamese-\nViVoice (zero-shot\nbaseline / alt engine)", "#FAD7A0", fs=7.7)
box(20.0, -2.0, 3.0, 1.5, "RVC v2\nHuBERT + RMVPE +\nHiFi-GAN + FAISS", "#FAD7A0", fs=7.5)

# ================================================================== arrows
# user <-> browser
arrow((2.7, 13.5), (2.9, 13.5), color="#B9770E")

# client -> rag-legal-assistant
arrow((3.9, 12.05), (3.9, 10.4), label="HTTPS: chat, /voice/transcribe,\n/voice/speak, profile & consent", color="#B9770E", label_pos=0.5, label_off=(2.3, 0.35), fs=8.0)

# rag routes -> RAG engine (internal)
arrow((3.9, 9.35), (3.9, 8.85), color="#2E86C1")
# RAG engine <-> ChromaDB / chat.db
arrow((2.3, 6.75), (2.3, 6.35), color="#229954")
arrow((5.6, 6.75), (5.6, 6.35), color="#229954")
# routes -> groq_keys -> Groq API
arrow((9.45, 9.35), (9.45, 8.85), color="#2E86C1")
arrow((11.6, 9.6), (10.6, 13.15), label="prompt / completion", color="#C0392B", label_pos=0.55, label_off=(-1.0, -0.9), fs=8.3)

# routes -> station_client adapter
arrow((9.45, 7.75), (9.45, 7.05), color="#2E86C1")
# adapter -> SDK
arrow((9.45, 6.55), (9.45, 6.35), color="#7D3C98")
# SDK -> clone-voice-station client API (the main cross-service HTTP call)
arrow((11.6, 5.6), (12.8, 9.6), label="HTTPS + X-Api-Key\ntranscribe / speak / profiles /\nconsent / train / admin / webhook",
      color="#7D3C98", lw=2.0, label_pos=0.45, label_off=(0.05, 1.55), fs=8.4)

# station client API -> internals
arrow((15.3, 9.35), (15.3, 9.1), color="#CA6F1E")
arrow((20.65, 9.35), (20.65, 9.1), color="#CA6F1E")
arrow((15.3, 7.55), (15.3, 7.25), color="#CA6F1E")
arrow((20.65, 7.55), (20.65, 7.25), color="#CA6F1E")
arrow((15.3, 5.55), (15.3, 5.15), color="#CA6F1E")
arrow((20.65, 5.55), (20.65, 5.15), color="#CA6F1E")

# stt/tts/rvc_client -> Colab (over cloudflared tunnel)
arrow((15.3, 5.55), (15.3, -0.2), label="STT + TTS-base + RVC\nrequests, over cloudflared\ntunnel (HTTPS)", color="#B9770E",
      lw=1.8, label_pos=0.5, label_off=(-2.2, -2.1), fs=8.2)
arrow((18.1, -1.1), (17.9, 5.55), color="#B9770E", lw=1.4)

# webhook: clone-voice-station -> rag-legal-assistant (dashed, event-driven)
arrow((12.8, 2.2), (11.6, 5.9), label="webhook POST /voice/webhook\n(manager disable/delete \u2192 notify)\n+ polling fallback /voice/notifications",
      color="#943126", dashed=True, lw=1.4, label_pos=0.5, label_off=(0.35, -1.4), fs=7.8, curve=-0.15)

# manager -> manager dashboard
arrow((19.4, 12.05), (17.3, 9.1), label="cookie session + CSRF\n(separate from rag-legal-\nassistant's own admin page)", color="#B9770E",
      lw=1.5, label_pos=0.5, label_off=(2.5, 1.0), fs=7.8)

# build-time pipeline internal flow
arrow((3.4, -1.1), (3.7, -1.1), color="#7D3C98")
arrow((6.7, -1.1), (7.0, -1.1), color="#7D3C98")
arrow((9.2, -1.1), (9.5, -1.1), color="#7D3C98")
# reused at runtime -> ChromaDB in rag-legal-assistant
arrow((10.6, -0.5), (2.3, 4.85), label="reused at runtime", color="#7D3C98", dashed=True, lw=1.3,
      label_pos=0.6, label_off=(3.6, 1.3), fs=8.0, curve=0.12)

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure2_architecture.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
