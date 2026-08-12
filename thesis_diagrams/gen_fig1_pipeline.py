import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 5.4))
ax.set_xlim(0, 15)
ax.set_ylim(-0.3, 5.0)
ax.axis("off")

def box(x, y, w, h, text, fc, ec="#555555", fs=9.0, weight="normal"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
             wrap=True, fontweight=weight, zorder=4)
    return b

def arrow(p1, p2, label=None, color="#333333", lw=1.3, label_pos=0.5, fs=8.2, label_off=(0, 0.16), curve=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                         color=color, zorder=2, connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + label_off[0]
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_off[1]
        ax.text(mx, my, label, fontsize=fs, ha="center", va="center", color=color, zorder=5,
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

y0 = 3.1
h0 = 1.6

b1 = box(0.2, y0, 2.2, h0, "1. Audio input\n(user speech,\nbrowser mic)", "#F5CBA7")
b2 = box(2.85, y0, 2.75, h0, "2. ASR\nPhoWhisper-large (Colab,\nprimary) / Whisper-small\n(local fallback)\nspeech \u2192 text", "#AED6F1", fs=8.4)
b3 = box(6.05, y0, 2.75, h0, "LLM\nLlama-3.1-8B-Instant\n(Groq, key-rotated)\nquestion \u2192 answer", "#A9DFBF", fs=8.6)
b3r = box(6.05, 0.9, 2.75, 1.4, "RAG retrieval\nBGE-small-en-v1.5 +\nChromaDB", "#A9DFBF", fs=8.6)
b4 = box(9.25, y0, 2.35, h0, "4. TTS\nedge-TTS\n(vi-VN-HoaiMyNeural)\ntext \u2192 base WAV", "#AED6F1", fs=8.5)

b5 = box(9.25, 0.9, 2.35, 1.4, "5. RVC v2\nColab GPU, cloudflared\ntunnel \u2192 target-\nspeaker WAV", "#FAD7A0", fs=8.3)
b6 = box(11.95, 0.9, 2.6, 1.4, "6. Audio output\nstreamed to browser,\nplayed in chat UI", "#FAD7A0", fs=8.4)

# stage-3 bracket label
ax.annotate("", xy=(6.0, y0 + h0 + 0.32), xytext=(8.85, y0 + h0 + 0.32),
            arrowprops=dict(arrowstyle="-", color="#555555", lw=1.0))
ax.text(7.4, y0 + h0 + 0.5, "Stage: language understanding\n(LLM + RAG, benchmarked separately)",
        fontsize=8.0, ha="center", va="bottom", color="#555555")

# main flow arrows
arrow((2.4, y0 + h0 / 2), (2.85, y0 + h0 / 2), label="WAV / webm", label_off=(0, 0.24))
arrow((5.6, y0 + h0 / 2), (6.05, y0 + h0 / 2), label="user question (text)", label_off=(0, 0.24))
arrow((8.8, y0 + h0 / 2), (9.25, y0 + h0 / 2), label="answer (text)", label_off=(0, 0.24))
arrow((7.4, 2.3), (7.4, y0), label="context", color="#229954", label_off=(0.55, 0))

arrow((10.42, y0), (10.42, 2.3), label="base WAV (vi-VN)", label_off=(1.15, 0.2), color="#B9770E")
arrow((11.6, 1.6), (11.95, 1.6), label="target-speaker\nWAV", label_off=(0, 0.32), color="#B9770E")

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure1_pipeline.png")
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("saved", out)
