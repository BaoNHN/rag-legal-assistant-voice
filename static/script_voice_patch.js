/**
 * script_voice_patch.js
 * ══════════════════════════════════════════════════════════════════════════════
 * This is NOT a standalone file. It shows the ONE change needed in script.js
 * to wire voice playback into the existing chat flow.
 *
 * In your existing static/script.js, find the function that appends the bot
 * reply to the chatbox (likely addMessage or the fetch("/get") .then block).
 *
 * It will look something like this:
 * ──────────────────────────────────────────────────────────────────────────────
 *
 *   function addMessage(sender, text) {
 *     const div = document.createElement("div");
 *     div.className = "message " + sender;
 *     div.textContent = text;
 *     document.getElementById("chatbox").appendChild(div);
 *     chatbox.scrollTop = chatbox.scrollHeight;
 *   }
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * Add ONE line so it becomes:
 * ──────────────────────────────────────────────────────────────────────────────
 *
 *   function addMessage(sender, text) {
 *     const div = document.createElement("div");
 *     div.className = "message " + sender;
 *     div.textContent = text;
 *     document.getElementById("chatbox").appendChild(div);
 *     chatbox.scrollTop = chatbox.scrollHeight;
 *
 *     if (sender === "bot" && typeof speakAnswer === "function") {
 *       speakAnswer(text);    // ← ADD THIS LINE ONLY
 *     }
 *   }
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * If your script.js uses a fetch chain instead of addMessage, find the block
 * that looks like:
 *
 *   .then(data => {
 *     const botText = data.answer || data.response || data.text;
 *     // ... append to DOM ...
 *   })
 *
 * And add after the DOM append:
 *
 *     if (typeof speakAnswer === "function") speakAnswer(botText);
 *
 * ══════════════════════════════════════════════════════════════════════════════
 */
