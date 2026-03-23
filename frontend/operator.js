/**
 * Maskki — Operator Dashboard Client
 *
 * Connects to /ws/operator for real-time customer session monitoring,
 * chat mirroring, operator override messaging, and training feedback.
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);

const dom = {
    // Status
    statusDot: $("#op-status-dot"),
    statusText: $("#op-status-text"),
    opIdDisplay: $("#op-id-display"),

    // Session sidebar
    sessionList: $("#session-list"),
    refreshBtn: $("#refresh-sessions-btn"),

    // Video
    watchingLabel: $("#watching-label"),
    videoFrame: $("#customer-video-frame"),
    videoPlaceholder: $("#customer-video-placeholder"),
    unsubscribeBtn: $("#unsubscribe-btn"),

    // Documents
    documentViewer: $("#document-viewer"),
    documentList: $("#document-list"),

    // Chat
    chatMessages: $("#op-chat-messages"),
    sessionInfo: $("#session-info"),
    messageInput: $("#op-message-input"),
    sendBtn: $("#op-send-btn"),

    // Training
    feedbackGood: $("#feedback-good"),
    feedbackBad: $("#feedback-bad"),
    feedbackNote: $("#feedback-note"),
};


// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

const state = {
    ws: null,
    operatorId: null,
    connected: false,
    subscribedSessionId: null,
    lastBotMessageId: null,
    messageCounter: 0,
};


// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

function getWsBase() {
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") return "ws://localhost:8000";
    return "wss://ws.maskki.com";
}

function setStatus(status, text) {
    dom.statusDot.className = `dot ${status}`;
    dom.statusText.textContent = text;
}

function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return str.replace(/[&<>"']/g, (c) => map[c]);
}

function formatTime(timestamp) {
    const d = new Date(timestamp * 1000);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}


// ═══════════════════════════════════════════════════════════════════
// WebSocket Connection
// ═══════════════════════════════════════════════════════════════════

function connect() {
    if (state.ws) return;

    const url = `${getWsBase()}/ws/operator`;
    setStatus("connecting", "Connecting...");

    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
        // Wait for auth_required or authenticated
    };

    ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        handleMessage(data);
    };

    ws.onclose = () => {
        state.ws = null;
        state.connected = false;
        setStatus("disconnected", "Disconnected");
        disableControls();
    };

    ws.onerror = () => {
        setStatus("disconnected", "Connection error");
    };
}

function handleMessage(data) {
    switch (data.type) {
        case "auth_required":
            // Auto-auth with no key for dev mode
            sendAuth();
            break;

        case "authenticated":
            state.operatorId = data.operator_id;
            state.connected = true;
            setStatus("connected", "Connected");
            dom.opIdDisplay.textContent = `Operator: ${data.operator_id}`;
            break;

        case "session_list":
            renderSessionList(data.sessions);
            break;

        case "subscribed":
            state.subscribedSessionId = data.session_id;
            dom.watchingLabel.textContent = `Monitoring: ${data.session_id}`;
            dom.unsubscribeBtn.disabled = false;
            dom.messageInput.disabled = false;
            dom.sendBtn.disabled = false;
            dom.feedbackGood.disabled = false;
            dom.feedbackBad.disabled = false;
            dom.feedbackNote.disabled = false;
            dom.videoPlaceholder.style.display = "none";
            dom.sessionInfo.textContent = data.session_id;
            clearChat();
            addChatMessage("system", "✅ Subscribed to session");
            break;

        case "unsubscribed":
            state.subscribedSessionId = null;
            dom.watchingLabel.textContent = "Not monitoring any session";
            dom.unsubscribeBtn.disabled = true;
            dom.videoPlaceholder.style.display = "flex";
            dom.videoFrame.src = "";
            disableControls();
            clearChat();
            addChatMessage("system", "Unsubscribed from session");
            break;

        case "customer_video_frame":
            displayVideoFrame(data.frame);
            break;

        case "customer_message":
            displayChatMessage(data);
            break;

        case "document_received":
            displayDocument(data);
            break;

        case "session_ended":
            addChatMessage("system", "🔴 Customer session ended");
            state.subscribedSessionId = null;
            dom.watchingLabel.textContent = "Session ended";
            dom.videoPlaceholder.style.display = "flex";
            dom.videoFrame.src = "";
            disableControls();
            // Refresh session list
            requestSessionList();
            break;

        case "message_sent":
            addChatMessage("operator", data.text);
            break;

        case "feedback_recorded":
            addChatMessage("system", `📝 Feedback recorded for message ${data.message_id}`);
            break;

        case "error":
            addChatMessage("system", `⚠️ ${data.error}${data.detail ? ": " + data.detail : ""}`);
            break;

        case "pong":
            break;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Session List
// ═══════════════════════════════════════════════════════════════════

function renderSessionList(sessions) {
    if (!sessions || sessions.length === 0) {
        dom.sessionList.innerHTML = '<div class="empty-state">No active sessions</div>';
        return;
    }

    let html = "";
    for (const s of sessions) {
        const isActive = s.session_id === state.subscribedSessionId;
        const connTime = new Date(s.connected_at * 1000).toLocaleTimeString();
        html += `
            <div class="session-card ${isActive ? 'active' : ''}" data-session-id="${s.session_id}">
                <div class="session-card-header">
                    <span class="session-id">${s.session_id.slice(0, 10)}...</span>
                    <span class="session-badge ${s.is_recording ? 'recording' : 'idle'}">
                        ${s.is_recording ? "🎙️ Recording" : "⏸️ Idle"}
                    </span>
                </div>
                <div class="session-card-meta">
                    <span>🌐 ${s.language.toUpperCase()}</span>
                    <span>🤖 ${s.avatar_id}</span>
                    <span>🕐 ${connTime}</span>
                </div>
                <button class="subscribe-btn" onclick="subscribeToSession('${s.session_id}')">
                    ${isActive ? 'Monitoring' : 'Monitor'}
                </button>
            </div>
        `;
    }
    dom.sessionList.innerHTML = html;
}

function subscribeToSession(sessionId) {
    if (!state.ws || !state.connected) return;
    state.ws.send(JSON.stringify({
        type: "subscribe",
        session_id: sessionId,
    }));
}

function unsubscribe() {
    if (!state.ws || !state.connected) return;
    state.ws.send(JSON.stringify({ type: "unsubscribe" }));
}

function requestSessionList() {
    if (!state.ws || !state.connected) return;
    state.ws.send(JSON.stringify({ type: "refresh_sessions" }));
}


// ═══════════════════════════════════════════════════════════════════
// Video Feed
// ═══════════════════════════════════════════════════════════════════

function displayVideoFrame(base64) {
    dom.videoFrame.src = `data:image/jpeg;base64,${base64}`;
    dom.videoFrame.style.display = "block";
    dom.videoPlaceholder.style.display = "none";
}


// ═══════════════════════════════════════════════════════════════════
// Chat Mirror
// ═══════════════════════════════════════════════════════════════════

function clearChat() {
    dom.chatMessages.innerHTML = "";
}

function addChatMessage(role, text, extras) {
    state.messageCounter++;
    const msgId = `msg_${state.messageCounter}`;

    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.id = msgId;

    let html = `<div class="msg-text">${escapeHtml(text)}</div>`;

    if (extras) {
        const metaParts = [];
        if (extras.emotion) metaParts.push(`Emotion: ${extras.emotion}`);
        if (extras.latency_ms) metaParts.push(`${extras.latency_ms}ms`);
        if (extras.timestamp) metaParts.push(formatTime(extras.timestamp));
        if (metaParts.length) {
            html += `<div class="msg-meta">${metaParts.map(p => `<span>${p}</span>`).join("")}</div>`;
        }
    }

    div.innerHTML = html;
    dom.chatMessages.appendChild(div);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;

    // Track last bot message for feedback
    if (role === "bot") {
        state.lastBotMessageId = msgId;
    }

    return msgId;
}

function displayChatMessage(data) {
    addChatMessage(data.role, data.text, {
        emotion: data.emotion,
        latency_ms: data.latency_ms,
        timestamp: data.timestamp,
    });
}


// ═══════════════════════════════════════════════════════════════════
// Documents
// ═══════════════════════════════════════════════════════════════════

function displayDocument(data) {
    dom.documentViewer.style.display = "block";

    const div = document.createElement("div");
    div.className = "doc-item";
    div.innerHTML = `
        <a href="${data.image_url}" target="_blank" class="doc-thumb-link">
            <img src="${data.image_url}" alt="${data.filename}" class="doc-thumb">
        </a>
        <div class="doc-info">
            <span class="doc-name">${escapeHtml(data.filename)}</span>
            <span class="doc-size">${(data.size_bytes / 1024).toFixed(1)} KB</span>
        </div>
    `;
    dom.documentList.appendChild(div);

    addChatMessage("system", `📄 Document received: ${data.filename}`);
}


// ═══════════════════════════════════════════════════════════════════
// Operator Message
// ═══════════════════════════════════════════════════════════════════

function sendOperatorMessage() {
    const text = dom.messageInput.value.trim();
    if (!text || !state.ws || !state.connected) return;

    state.ws.send(JSON.stringify({
        type: "operator_message",
        text: text,
    }));

    dom.messageInput.value = "";
}


// ═══════════════════════════════════════════════════════════════════
// Training Feedback
// ═══════════════════════════════════════════════════════════════════

function sendFeedback(quality) {
    if (!state.ws || !state.connected || !state.lastBotMessageId) return;

    state.ws.send(JSON.stringify({
        type: "training_feedback",
        message_id: state.lastBotMessageId,
        quality: quality,
        note: dom.feedbackNote.value.trim(),
    }));

    dom.feedbackNote.value = "";

    // Visual feedback
    if (quality === "good") {
        dom.feedbackGood.classList.add("pressed");
        setTimeout(() => dom.feedbackGood.classList.remove("pressed"), 500);
    } else {
        dom.feedbackBad.classList.add("pressed");
        setTimeout(() => dom.feedbackBad.classList.remove("pressed"), 500);
    }
}


// ═══════════════════════════════════════════════════════════════════
// Auth
// ═══════════════════════════════════════════════════════════════════

function sendAuth() {
    // In dev mode, send empty auth (server allows no-key connections)
    if (state.ws) {
        state.ws.send(JSON.stringify({
            type: "auth",
            api_key: "",
        }));
    }
}


// ═══════════════════════════════════════════════════════════════════
// Controls
// ═══════════════════════════════════════════════════════════════════

function disableControls() {
    dom.messageInput.disabled = true;
    dom.sendBtn.disabled = true;
    dom.feedbackGood.disabled = true;
    dom.feedbackBad.disabled = true;
    dom.feedbackNote.disabled = true;
}


// ═══════════════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════════════

dom.refreshBtn.addEventListener("click", requestSessionList);
dom.unsubscribeBtn.addEventListener("click", unsubscribe);

dom.sendBtn.addEventListener("click", sendOperatorMessage);
dom.messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendOperatorMessage();
    }
});

dom.feedbackGood.addEventListener("click", () => sendFeedback("good"));
dom.feedbackBad.addEventListener("click", () => sendFeedback("bad"));

// Keep-alive ping every 30s
setInterval(() => {
    if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: "ping" }));
    }
}, 30000);

// Auto-refresh session list every 10s
setInterval(() => {
    if (state.connected) requestSessionList();
}, 10000);


// ═══════════════════════════════════════════════════════════════════
// Auto-connect on page load
// ═══════════════════════════════════════════════════════════════════

window.addEventListener("load", () => {
    connect();
});
