const sendButton = document.getElementById("sendButton");
const micButton = document.getElementById("micButton");
const chatInput = document.getElementById("chatInput");
const chatbox = document.getElementById("chatbox");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.querySelector(".new-chat");
const mainContainer = document.getElementById("mainContainer");

let currentChatId = null;
let chats = {};  // { chatId: [{role:'user', text:'hi'}, ...] }

// Reserved title for the auto-created import-law status chat — must match
// database.database.NOTIFICATION_CHAT_TITLE. Chats with this exact title
// are read-only (no sending) and excluded from the 5-chat cap.
const NOTIFICATION_CHAT_TITLE = "Thông báo";

// ── Response voice/tone toggle (formal = nghiêm túc, casual = đời thường) ──
// Per-browser preference (localStorage), default "formal" — matches the
// system's existing strict legal-writing style unless the user opts out.
let currentVoice = localStorage.getItem("voice") || "formal";

function updateVoiceButton() {
    const checkbox = document.getElementById("voiceToggleInput");
    const label = document.getElementById("voiceToggleLabel");
    if (!checkbox || !label) return;
    checkbox.checked = currentVoice === "casual";
    label.textContent = currentVoice === "casual" ? "😊 Đời thường" : "🎓 Nghiêm túc";
}

function toggleVoice() {
    currentVoice = currentVoice === "casual" ? "formal" : "casual";
    localStorage.setItem("voice", currentVoice);
    updateVoiceButton();
}

// ── Lock/unlock the input bar (read-only "Thông báo" chat, or message cap hit) ──
function setInputLocked(locked, reason) {
    chatInput.disabled = locked;
    sendButton.disabled = locked;
    chatInput.placeholder = locked
        ? (reason || `Đoạn chat '${NOTIFICATION_CHAT_TITLE}' chỉ để xem, không thể gửi tin nhắn.`)
        : "Nhập câu hỏi về Luật Doanh nghiệp…";
}

// ── Welcome / Chat mode toggle ────────────────────
function enterChatMode() {
    mainContainer.classList.add("is-chatting");
}

function enterWelcomeMode() {
    mainContainer.classList.remove("is-chatting");
}

// ── Suggestion chips ──────────────────────────────
document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
        chatInput.value = chip.dataset.prompt;
        chatInput.focus();
    });
});

// load Chat From DataBase
async function loadChatFromDB(chatId) {
    currentChatId = chatId;
    clearChatbox();
    enterChatMode();

    const chatElem = document.querySelector(`.chat-item[data-id="${chatId}"]`);
    setInputLocked(chatElem?.dataset.title === NOTIFICATION_CHAT_TITLE);

    const messages = await fetch(`/get_chat_messages?chat_id=${chatId}`)
        .then(r => r.json());

    chats[chatId] = messages;

    messages.forEach(m => {
        displayMessage(m.text, m.role === "user");
    });
}

// Everything before "📖 Nguồn chính" (citations/links) — this is what gets
// read aloud by the speaker button, so sources are never spoken.
function extractSpokenText(text) {
    return text.split(/\n*📖 Nguồn chính:/)[0].trim().replace(/\*\*/g, '');
}

// Escapes text for safe insertion into innerHTML. Needed anywhere user-typed
// text (chat messages, chat titles) or LLM output (which can echo back
// retrieved-document or user-question content) gets rendered as HTML rather
// than left as plain DOM text -- without this, a message/title containing
// "<" or "&" renders as live markup instead of literal characters.
function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s ?? '';
    return div.innerHTML;
}

// Format assistant message: parse structured sections + citation block
function formatAssistantHTML(text) {
    const parts = text.split(/\n*📖 Nguồn chính:/);
    // Strip ** bold markers so they never appear in UI
    const bodyText = escapeHtml(parts[0].trim().replace(/\*\*/g, ''));

    // indexOf-based section extractor — more reliable than regex for Vietnamese
    function getSection(label) {
        const marker = label + ':';
        const start = bodyText.indexOf(marker);
        if (start === -1) return null;
        const from = start + marker.length;
        const siblings = ['Kết luận:', 'Căn cứ pháp lý:', 'Phân tích:', 'Lưu ý:'];
        let end = bodyText.length;
        for (const m of siblings) {
            if (m === marker) continue;
            const i = bodyText.indexOf(m, from);
            if (i !== -1 && i < end) end = i;
        }
        return bodyText.substring(from, end).trim() || null;
    }

    const conclusion  = getSection('Kết luận');
    const legalBasis  = getSection('Căn cứ pháp lý');
    const analysis    = getSection('Phân tích');
    const note        = getSection('Lưu ý');

    let html = '';
    if (conclusion || analysis) {
        if (conclusion)
            html += `<div class="msg-section msg-conclusion"><span class="section-label">Kết luận</span>${conclusion.replace(/\n/g, '<br>')}</div>`;
        if (legalBasis)
            html += `<div class="msg-section msg-legal-basis"><span class="section-label">Căn cứ pháp lý</span>${legalBasis.replace(/\n/g, '<br>')}</div>`;
        if (analysis)
            html += `<div class="msg-section msg-analysis"><span class="section-label">Phân tích</span>${analysis.replace(/\n/g, '<br>')}</div>`;
        if (note)
            html += `<div class="msg-section msg-note"><span class="section-label">Lưu ý</span>${note.replace(/\n/g, '<br>')}</div>`;
    } else {
        html = `<div class="msg-body">${bodyText.replace(/\n/g, '<br>')}</div>`;
    }

    // Citation blocks
    if (parts.length > 1) {
        const citationRaw = escapeHtml(parts[1].trim());
        const citParts = citationRaw.split(/\n📎 Nguồn tham khảo:/);
        const primary = citParts[0].trim().replace(/\n/g, '<br>');
        html += `<div class="msg-citation">📖 Nguồn chính: ${primary}</div>`;
        if (citParts.length > 1) {
            const secondary = citParts[1].trim().replace(/\n/g, '<br>');
            html += `<div class="msg-citation-secondary">📎 Nguồn tham khảo:<br>${secondary}</div>`;
        }
    }
    return html;
}

// Display chat bubbles
async function displayMessage(message, isUser) {
    const msgElem = document.createElement('div');
    if (isUser) {
        msgElem.innerHTML = escapeHtml(message).replace(/\n/g, "<br>");
    } else {
        msgElem.innerHTML = formatAssistantHTML(message);
    }
    msgElem.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;
    chatbox.appendChild(msgElem);

    // Speaker (read-aloud) button — assistant messages only, skip the "thinking…" placeholder
    if (!isUser && message !== "⏳ Đang suy nghĩ...") {
        appendSpeakButton(msgElem, extractSpokenText(message));
    }

    chatbox.scrollTop = chatbox.scrollHeight;

    if (!isUser) {
        msgElem.style.opacity = 0;
        await new Promise(resolve => setTimeout(resolve, 300));
        msgElem.style.opacity = 1;
    }
}

// ── Read-aloud (speaker) button ───────────────────────────────────────────
let _activeSpeakAudio = null;

function appendSpeakButton(container, spokenText) {
    const btn = document.createElement('button');
    btn.className = 'msg-speak-btn';
    btn.title = 'Đọc to nội dung này';
    btn.innerHTML = '<span class="material-icons" style="font-size:16px;">volume_up</span>';
    btn.addEventListener('click', () => speakMessage(spokenText, btn));
    container.appendChild(btn);
}

async function speakMessage(text, btn) {
    // Stop any currently playing speech (including this same button = toggle-off)
    const wasPlayingThis = btn.classList.contains('playing');
    if (_activeSpeakAudio) {
        _activeSpeakAudio.pause();
        _activeSpeakAudio = null;
    }
    document.querySelectorAll('.msg-speak-btn.playing, .msg-speak-btn.loading')
        .forEach(b => b.classList.remove('playing', 'loading'));
    if (wasPlayingThis) return;

    btn.classList.add('loading');
    try {
        const profileId = window.__selectedVoiceProfileId || null;
        const resp = await fetch('/voice/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, profile_id: profileId })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);

        const blob  = await resp.blob();
        const url   = URL.createObjectURL(blob);
        const audio = new Audio(url);
        _activeSpeakAudio = audio;

        btn.classList.remove('loading');
        btn.classList.add('playing');
        await audio.play();

        audio.onended = () => {
            btn.classList.remove('playing');
            if (_activeSpeakAudio === audio) _activeSpeakAudio = null;
        };
    } catch (e) {
        btn.classList.remove('loading', 'playing');
        console.error('[Voice] Đọc to thất bại:', e);
    }
}

// ── Mic (speech-to-text) button ───────────────────────────────────────────
// Live mode: every _liveTranscribeIntervalMs while still recording, the
// audio captured so far is re-sent to Whisper and the chat input is updated
// with the running transcript — recording itself only stops when the user
// clicks the mic button again.
let _mediaRecorder = null;
let _recordingChunks = [];
let _liveTranscribeTimer = null;
let _liveTranscribeInFlight = false;
let _lastLiveTranscript = null;

// Local (in-process, no network hop) can afford a tighter tick than remote
// (an HTTP round trip to clone-voice-station plus its own inference) — same
// local/remote-aware cadence voice-lab-example's /compare page uses for its
// own local-vs-lazy comparison (1s/2s there). Which one applies is decided
// fresh at the start of each recording via GET /voice/status's
// local_stt_enabled, so toggling local mode in admin_voice_models.html takes
// effect on the very next recording, no reload needed.
const LIVE_TRANSCRIBE_INTERVAL_LOCAL_MS  = 1000;
const LIVE_TRANSCRIBE_INTERVAL_REMOTE_MS = 2000;

let _liveTranscribeIntervalMs = LIVE_TRANSCRIBE_INTERVAL_REMOTE_MS; // set per-recording in micButton's click handler below

// Stale-response guard: a live tick's request and the final stop-triggered
// request can both be in flight at once (stopping the recorder doesn't
// cancel a tick that's already mid-fetch), and nothing about HTTP guarantees
// they resolve in the order they were sent — a slow, older/partial-audio
// tick response can land after the final, complete-audio response and
// clobber it. Tag every request with a sequence number at send time; when a
// response comes back, only apply it if it's newer than the last one
// actually applied.
let _requestSeq = 0;
let _appliedSeq  = 0;

// Sliding window for live ticks — caps how much audio a tick re-transcribes
// so cost stays flat instead of growing with total recording length (tick N
// used to have to re-send/re-infer all of the N*500ms recorded so far).
// _recordingChunks[0] carries the WebM header (Matroska Segment/Tracks info)
// that every later chunk depends on to decode — later chunks are just codec
// SimpleBlocks with no header of their own — so it's always kept even once
// it ages out of the time window, with the most recent chunks appended after it.
const WINDOW_CHUNK_COUNT = 14; // ~7s at the 500ms MediaRecorder timeslice below

function _windowedChunks() {
    if (_recordingChunks.length <= WINDOW_CHUNK_COUNT) return _recordingChunks;
    const header = _recordingChunks[0];
    const recent = _recordingChunks.slice(-WINDOW_CHUNK_COUNT);
    return recent.includes(header) ? recent : [header, ...recent];
}

// Sends the given chunks (the full recording for the final transcribe, or a
// capped window for a live tick — see _windowedChunks()) and returns the
// transcript.
async function transcribeChunksSoFar(chunks) {
    const formData = new FormData();
    // Actual recorder mimeType, not a hardcoded guess (matches voice-lab-example's
    // compare.js) -- MediaRecorder's default container/codec isn't guaranteed
    // to be exactly "audio/webm" across browsers.
    formData.append('audio', new Blob(chunks, { type: _mediaRecorder?.mimeType || 'audio/webm' }), 'recording.webm');
    const resp = await fetch('/voice/transcribe', { method: 'POST', body: formData });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.message || data.error || data.detail || ('HTTP ' + resp.status));
    return data.text || '';
}

// Applies a transcript to the chat input only if `seq` is newer than the
// last one actually applied (drops a late-arriving, now-superseded
// response instead of clobbering fresher text), and only if the user hasn't
// started editing the input themselves mid-recording (don't clobber their
// own typing either).
function _applyLiveTranscript(seq, text) {
    if (!text || seq < _appliedSeq) return;
    if (!(chatInput.value === '' || chatInput.value === _lastLiveTranscript)) return;
    _appliedSeq = seq;
    chatInput.value = text;
    _lastLiveTranscript = text;
}

async function liveTranscribeTick() {
    if (_liveTranscribeInFlight || _recordingChunks.length === 0) return;
    _liveTranscribeInFlight = true;
    const seq = ++_requestSeq;
    try {
        const text = await transcribeChunksSoFar(_windowedChunks());
        _applyLiveTranscript(seq, text);
    } catch (e) {
        // Best-effort — the final transcribe on stop is authoritative, so
        // a dropped live tick just means one less mid-recording update.
        console.warn('[Voice] Live transcribe tick failed:', e);
    } finally {
        _liveTranscribeInFlight = false;
    }
}

micButton?.addEventListener('click', async () => {
    if (_mediaRecorder && _mediaRecorder.state === 'recording') {
        _mediaRecorder.stop();  // triggers onstop below, which sends the final audio off
        return;
    }

    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
        alert('Không thể truy cập microphone: ' + e.message);
        return;
    }

    // Decided fresh per recording (not cached) so toggling local mode in
    // admin_voice_models.html applies to the very next recording started.
    // Best-effort — a failed/unauthorized status check just keeps the
    // slower remote-paced default rather than blocking the mic.
    try {
        const status = await fetch('/voice/status').then(r => r.json());
        _liveTranscribeIntervalMs = status.local_stt_enabled
            ? LIVE_TRANSCRIBE_INTERVAL_LOCAL_MS
            : LIVE_TRANSCRIBE_INTERVAL_REMOTE_MS;
    } catch (e) {
        _liveTranscribeIntervalMs = LIVE_TRANSCRIBE_INTERVAL_REMOTE_MS;
    }

    _recordingChunks = [];
    _lastLiveTranscript = null;
    _requestSeq = 0;
    _appliedSeq = 0;
    _mediaRecorder = new MediaRecorder(stream);
    _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _recordingChunks.push(e.data); };
    _mediaRecorder.onstop = async () => {
        clearInterval(_liveTranscribeTimer);
        _liveTranscribeTimer = null;
        stream.getTracks().forEach(t => t.stop());
        micButton.classList.remove('recording');
        micButton.classList.add('transcribing');

        // Issued after every live tick for this recording, so this seq
        // number is guaranteed the highest — it always wins
        // _applyLiveTranscript's check, even if a straggling tick response
        // arrives after this one does.
        const seq = ++_requestSeq;
        try {
            // Fill the input rather than auto-send — Whisper can mishear Vietnamese
            // legal terms, so the user gets a chance to review/edit before sending.
            const text = await transcribeChunksSoFar(_recordingChunks); // full recording, not windowed
            _applyLiveTranscript(seq, text);
            chatInput.focus();
        } catch (e) {
            console.error('[Voice] Nhận diện giọng nói thất bại:', e);
            alert('Không nhận diện được giọng nói: ' + e.message);
        } finally {
            micButton.classList.remove('transcribing');
        }
    };

    // 500ms timeslice so chunks accumulate progressively instead of only at
    // stop — required for liveTranscribeTick to have growing audio to send.
    _mediaRecorder.start(500);
    micButton.classList.add('recording');
    _liveTranscribeTimer = setInterval(liveTranscribeTick, _liveTranscribeIntervalMs);
});

async function callApi(prompt) {
    chatInput.value = "Đang gửi…";
    chatInput.disabled = true;
    sendButton.disabled = true;

    try {
        const response = await fetch("/get", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                chat_id: currentChatId,
                voice: currentVoice
            })
        });

        // 🔥 FIX: đọc text trước
        const text = await response.text();

        // 🔥 FIX: parse JSON an toàn
        try {
            return JSON.parse(text);
        } catch (e) {
            console.error("JSON PARSE ERROR:", text);
            return {
                status: "error",
                text: "❌ Server trả dữ liệu không hợp lệ"
            };
        }

    } catch (err) {
        console.error("FETCH ERROR:", err);
        return {
            status: "error",
            text: "❌ Không kết nối được server"
        };
    } finally {
        chatInput.value = "";
        chatInput.disabled = false;
        sendButton.disabled = false;
    }
}


chatInput.focus();

// Click send button event
sendButton.addEventListener('click', async () => {
    const message = chatInput.value.trim();
    if (!message) return;

    try {
        // Guests get one ephemeral, never-persisted chat (chat_id stays
        // null) — skip chat creation/rename entirely, /get handles the
        // per-session message cap itself (see app.py).
        if (!window.isGuest) {
            // 1️⃣ Ensure chat exists
            if (!currentChatId) {
                await createNewChat();
                // createNewChat() bails out (leaving currentChatId unset) when the
                // 5-chat cap is hit and shows a picker instead — stop here so the
                // message isn't sent against a nonexistent chat.
                if (!currentChatId) return;
            }

            // 2️⃣ Ensure memory exists (🔥 FIX crash)
            if (!chats[currentChatId]) {
                chats[currentChatId] = [];
            }

            // 3️⃣ Save user message
            chats[currentChatId].push({ role: "user", text: message });

            // 4️⃣ Auto rename chat
            const chatElem = document.querySelector(`.chat-item[data-id="${currentChatId}"]`);
            if (chatElem) {
                const titleElem = chatElem.querySelector(".chat-title");

                if (titleElem && titleElem.innerText === "Đoạn chat mới") {
                    const newName = message.slice(0, 20);

                    const renameRes  = await fetch("/rename_chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            chat_id: currentChatId,
                            title: newName
                        })
                    });
                    const renameData = await renameRes.json();

                    if (renameData.status !== "error") {
                        renameChat(currentChatId, newName);
                    }
                }
            }
        }

        // 5️⃣ Switch to chat mode on first message
        enterChatMode();

        // Display user message
        displayMessage(message, true);
        chatInput.value = "";

        // 6️⃣ Show loading
        displayMessage("⏳ Đang suy nghĩ...", false);

        // 7️⃣ Call API
        const data = await callApi(message);
        console.log("API RESPONSE:", data);

        // 8️⃣ Remove loading (🔥 FIX UI)
        const loadingMsg = chatbox.lastChild;
        if (loadingMsg) loadingMsg.remove();

        // 9️⃣ Handle response
        if (data && data.status === "success") {
            displayMessage(data.text, false);
        } else if (data && data.status === "limit") {
            displayMessage(data.text, false);
            setInputLocked(true, "Đã đạt giới hạn số câu hỏi cho đoạn chat này.");
        } else {
            displayMessage(data?.text || "❌ Lỗi không xác định", false);
        }

    } catch (err) {
        console.error("SEND ERROR:", err);
        displayMessage("❌ Server error: " + err.message, false);
    }
});
newChatBtn.addEventListener("click", () => {
createNewChat();
});

// Create New Chat
async function createNewChat() {

    const res = await fetch("/create_chat", {
        method: "POST"
    });

    const data = await res.json();

    if (res.status === 409) {
        showChatLimitPicker(data.message, data.chats || []);
        return;
    }

    currentChatId = data.chat_id;

    chats[currentChatId] = [];

    createChatItem(currentChatId, "Đoạn chat mới");

    clearChatbox();
    enterWelcomeMode();
    setInputLocked(false);
}

// ── Chat-limit picker: shown when /create_chat returns 409 (5-chat cap hit).
// Lets the user delete an old chat right from the modal, then retries.
function showChatLimitPicker(message, chatsToPick) {
    const overlay = document.createElement("div");
    overlay.className = "chat-modal-overlay";

    const rows = chatsToPick.map(c => `
        <div class="chat-modal-row" data-id="${c.id}">
            <span title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
            <button class="chat-modal-delete">Xoá</button>
        </div>
    `).join("");

    overlay.innerHTML = `
        <div class="chat-modal">
            <h3>Đã đạt giới hạn đoạn chat</h3>
            <p>${escapeHtml(message) || "Vui lòng xoá một đoạn chat cũ trước khi tạo mới."}</p>
            ${rows}
            <button class="chat-modal-close">Đóng</button>
        </div>
    `;

    overlay.querySelectorAll(".chat-modal-delete").forEach(btn => {
        btn.addEventListener("click", async () => {
            const row = btn.closest(".chat-modal-row");
            await deleteChat(row.dataset.id);
            overlay.remove();
            createNewChat();
        });
    });

    overlay.querySelector(".chat-modal-close").addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.remove();
    });

    document.body.appendChild(overlay);
}

function clearChatbox() {
    chatbox.innerHTML = "";
}

// load Sidebar Chats that contains old chat from db
async function loadSidebarChats() {
    chatList.innerHTML = "";

    const chatData = await fetch("/list_chats").then(r => r.json());

    chatData.forEach(chat => {
        createChatItem(chat.id, chat.title);
    });
}

window.onload = () => {
    loadSidebarChats();
    updateVoiceButton();
};

// change title chat name
function renameChat(chatId, newName) {
    const items = document.querySelectorAll(".chat-item");
    items.forEach(i => {
        if (i.dataset.id === chatId) {
            i.dataset.title = newName;
            const titleElem = i.querySelector(".chat-title");
            titleElem.innerText = newName;
            titleElem.title = newName;
        }
    });
}

function createChatItem(chatId, title) {
    const item = document.createElement("div");
    item.className = "chat-item";
    item.dataset.id = chatId;
    item.dataset.title = title;

    item.innerHTML = `
        <span class="chat-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
        <span class="chat-options">⋯</span>
    `;

    item.querySelector(".chat-title").addEventListener("click", () => {
        loadChatFromDB(chatId);
    });

    item.querySelector(".chat-options").addEventListener("click", (e) => {
        e.stopPropagation();
        openChatMenu(chatId, item);
    });

    chatList.appendChild(item);
}

function openChatMenu(chatId, chatItem) {
    // Create menu
    const menu = document.createElement("div");
    menu.className = "chat-menu";
    menu.innerHTML = `
        <div class="menu-item rename-item">Đổi tên</div>
        <div class="menu-item delete-item">Xoá</div>
    `;

    // Position menu next to the chat item
    chatItem.appendChild(menu);

    // Rename
menu.querySelector(".rename-item").addEventListener("click", async () => {
    const newName = prompt("Nhập tên chat mới:");
    if (newName && newName.trim() !== "") {

        const res  = await fetch("/rename_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                chat_id: chatId,
                title: newName.trim()
            })
        });
        const data = await res.json();

        if (data.status === "error") {
            alert(data.message || "Không thể đổi tên đoạn chat này.");
        } else {
            renameChat(chatId, newName.trim());
        }
    }
    menu.remove();
});

    // Delete
    menu.querySelector(".delete-item").addEventListener("click", () => {
        deleteChat(chatId);
        menu.remove();
    });

    // Clicking outside closes the menu
    document.addEventListener("click", function closeMenu(e) {
        if (!menu.contains(e.target)) {
            menu.remove();
            document.removeEventListener("click", closeMenu);
        }
    });
}

async function deleteChat(chatId) {
    await fetch("/delete_chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: chatId })
    });

    // Remove from memory
    delete chats[chatId];

    // Remove from sidebar
    const item = document.querySelector(`.chat-item[data-id="${chatId}"]`);
    if (item) item.remove();

    // Clear chatbox if deleting active chat
    if (currentChatId === chatId) {
        clearChatbox();
        currentChatId = null;
        setInputLocked(false);
    }
}

chatInput.addEventListener("keydown", function (event) {

    // ENTER → send message
    if (event.key === "Enter") {
        event.preventDefault();
        sendButton.click();
    }
});

function logout() {
    fetch("/logout", { method: "POST" })
    .then(() => window.location.href = "/")
}