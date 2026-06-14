const sendButton = document.getElementById("sendButton");
const chatInput = document.getElementById("chatInput");
const chatbox = document.getElementById("chatbox");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.querySelector(".new-chat");

let currentChatId = null;
let chats = {};  // { chatId: [{role:'user', text:'hi'}, ...] }

// load Chat From DataBase
async function loadChatFromDB(chatId) {
    currentChatId = chatId;
    clearChatbox();

    const messages = await fetch(`/get_chat_messages?chat_id=${chatId}`)
        .then(r => r.json());

    chats[chatId] = messages;

    messages.forEach(m => {
        displayMessage(m.text, m.role === "user");
    });
}

// Display chat bubbles
async function displayMessage(message, isUser) {
                const msgElem = document.createElement('div');
            msgElem.innerHTML = message.replace(/\n/g, "<br>");
            msgElem.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;
            chatbox.appendChild(msgElem);
            chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom

            // For assistant messages, add a slight delay before displaying
            if (!isUser) {
                msgElem.style.opacity = 0; // Start with opacity 0
                await new Promise(resolve => setTimeout(resolve, 300)); // Delay
                msgElem.style.opacity = 1; // Fade in
            }
}

async function callApi(prompt) {
    chatInput.value = "Typing...";
    chatInput.disabled = true;
    sendButton.disabled = true;

    try {
        const response = await fetch("/get", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                chat_id: currentChatId
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
        // 1️⃣ Ensure chat exists
        if (!currentChatId) {
            await createNewChat();
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

            if (titleElem && titleElem.innerText === "New Chat") {
                const newName = message.slice(0, 20);
                titleElem.innerText = newName;

                await fetch("/rename_chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        chat_id: currentChatId,
                        title: newName
                    })
                });
            }
        }

        // 5️⃣ Display user message
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
            speakAnswer(data.text);
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

    currentChatId = data.chat_id;   // ✅ lấy từ server

    chats[currentChatId] = [];

    createChatItem(currentChatId, "New Chat");

    clearChatbox();
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

window.onload = loadSidebarChats;

// change title chat name
function renameChat(chatId, newName) {
    const items = document.querySelectorAll(".chat-item");
    items.forEach(i => {
        if (i.dataset.id === chatId) {
            i.querySelector(".chat-title").innerText = newName;
        }
    });
}

function createChatItem(chatId, title) {
    const item = document.createElement("div");
    item.className = "chat-item";
    item.dataset.id = chatId;

    item.innerHTML = `
        <span class="chat-title">${title}</span>
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
        <div class="menu-item rename-item">Rename</div>
        <div class="menu-item delete-item">Delete</div>
    `;

    // Position menu next to the chat item
    chatItem.appendChild(menu);

    // Rename
menu.querySelector(".rename-item").addEventListener("click", async () => {
    const newName = prompt("Enter new chat name:");
    if (newName && newName.trim() !== "") {

        renameChat(chatId, newName.trim());

        // 🔥 update DB
        await fetch("/rename_chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                chat_id: chatId,
                title: newName.trim()
            })
        });
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