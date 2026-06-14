/**
 * static/voice.js
 * ══════════════════════════════════════════════════════════════════════════════
 * Voice feature for rag-legal-assistant-master.
 * Handles:
 *   - Mic recording (MediaRecorder API → /voice/transcribe)
 *   - Auto-speak of RAG answers (/voice/speak → <audio> playback)
 *   - RVC status check on load
 *
 * ADD to templates/index.html just before </body>:
 *   <script src="{{ url_for('static', filename='voice.js') }}"></script>
 * ══════════════════════════════════════════════════════════════════════════════
 */

/* ── State ─────────────────────────────────────────────────────────────────── */
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;

/* ── Init: check RVC status on page load ────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  fetch("/voice/status")
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById("rvcStatus");
      if (!el) return;
      if (data.rvc_available) {
        el.textContent = "✓ RVC online";
        el.style.color = "#4ade80";
      } else if (data.rvc_endpoint) {
        el.textContent = "⚠ RVC offline (TTS fallback)";
        el.style.color = "#fbbf24";
      } else {
        el.textContent = "TTS only";
        el.style.color = "#9ca3af";
      }
    })
    .catch(() => {});
});

/* ── Mic recording ──────────────────────────────────────────────────────────── */
async function toggleMic() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMime() });
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const mime  = mediaRecorder.mimeType || "audio/webm";
      const blob  = new Blob(audioChunks, { type: mime });
      stream.getTracks().forEach(t => t.stop());
      await sendAudioForTranscription(blob, mime);
    };

    mediaRecorder.start(200);   // collect chunks every 200ms
    isRecording = true;
    updateMicUI(true);

  } catch (err) {
    console.error("[Voice] Mic access denied:", err);
    showToast("Không thể truy cập microphone. Kiểm tra quyền trình duyệt.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  updateMicUI(false);
}

function getSupportedMime() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg", "audio/mp4"];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

/* ── Send audio to /voice/transcribe ─────────────────────────────────────────── */
async function sendAudioForTranscription(blob, mime) {
  showToast("Đang nhận dạng giọng nói…");

  const formData = new FormData();
  formData.append("audio", blob, "recording" + (mime.includes("webm") ? ".webm" : ".ogg"));

  try {
    const resp = await fetch("/voice/transcribe", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (data.error) {
      showToast("Lỗi: " + data.error);
      return;
    }

    const text = data.text || "";
    if (!text) {
      showToast("Không nhận được giọng nói. Vui lòng thử lại.");
      return;
    }

    // Put transcribed text into chat input and auto-send
    const input = document.getElementById("chatInput");
    if (input) {
      input.value = text;
      showToast(`✓ Nhận dạng: "${text.substring(0, 60)}${text.length > 60 ? "…" : ""}"`);
      // Trigger send after short delay so user can see the text
      setTimeout(() => {
        const sendBtn = document.getElementById("sendButton");
        if (sendBtn) sendBtn.click();
      }, 400);
    }

  } catch (err) {
    console.error("[Voice] Transcription error:", err);
    showToast("Lỗi nhận dạng giọng nói. Kiểm tra kết nối server.");
  }
}

/* ── Speak answer via /voice/speak ──────────────────────────────────────────── */
async function speakAnswer(text) {
  const autoSpeak = document.getElementById("autoSpeakCheck");
  if (autoSpeak && !autoSpeak.checked) return;

  const voice   = document.getElementById("ttsVoiceSelect")?.value || "vi-VN-HoaiMyNeural";
  const useRvc  = document.getElementById("useRvcCheck")?.checked ?? true;

  try {
    const resp = await fetch("/voice/speak", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, voice, use_rvc: useRvc }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const audioBlob = await resp.blob();
    const audioUrl  = URL.createObjectURL(audioBlob);
    const player    = document.getElementById("voicePlayer");

    if (player) {
      player.src = audioUrl;
      player.style.display = "block";
      player.play().catch(e => console.warn("[Voice] Autoplay blocked:", e));

      // Log latency headers for thesis evaluation
      const tts = resp.headers.get("X-Latency-TTS-ms");
      const rvc = resp.headers.get("X-Latency-RVC-ms");
      const rvcUsed = resp.headers.get("X-RVC-Used");
      if (tts) console.log(`[Voice] TTS: ${tts}ms | RVC: ${rvc}ms | RVC used: ${rvcUsed}`);
    }

  } catch (err) {
    console.error("[Voice] Speak error:", err);
  }
}

/* ── Hook into the existing sendMessage flow ────────────────────────────────── */
/**
 * Call this from the existing script.js after the bot reply is appended.
 * Example — in your existing addMessage() or after fetch("/get") resolves:
 *
 *   if (sender === "bot") {
 *     speakAnswer(text);        // ← add this one line
 *   }
 */

/* ── UI helpers ─────────────────────────────────────────────────────────────── */
function updateMicUI(recording) {
  const btn    = document.getElementById("micBtn");
  const status = document.getElementById("recordingStatus");
  if (!btn) return;

  if (recording) {
    btn.style.background = "#ef4444";
    btn.innerHTML = '<span class="material-icons" style="font-size:20px;">stop</span>';
    if (status) status.style.display = "block";
  } else {
    btn.style.background = "rgba(255,255,255,0.1)";
    btn.innerHTML = '<span class="material-icons" style="font-size:20px;">mic</span>';
    if (status) status.style.display = "none";
  }
}

let _toastTimer = null;
function showToast(msg) {
  let toast = document.getElementById("voiceToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "voiceToast";
    Object.assign(toast.style, {
      position: "fixed", bottom: "80px", left: "50%",
      transform: "translateX(-50%)",
      background: "#1e1e2e", color: "#e2e8f0",
      padding: "8px 16px", borderRadius: "8px",
      fontSize: "13px", zIndex: "9999",
      border: "1px solid rgba(255,255,255,0.15)",
      boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
      transition: "opacity .3s",
    });
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = "1";
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { toast.style.opacity = "0"; }, 3000);
}
