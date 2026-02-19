/**
 * SmartTalker Frontend — WebRTC + WebSocket Chat Client
 *
 * Connects to:
 *   /ws/chat  — text chat + audio streaming (WebSocket)
 *   /ws/rtc   — WebRTC signaling for low-latency mic audio
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);

const dom = {
    // Status
    statusDot:    $("#status-dot"),
    statusText:   $("#status-text"),
    sessionId:    $("#session-id-display"),

    // Chat
    messages:     $("#chat-messages"),
    chatInput:    $("#chat-input"),
    sendBtn:      $("#send-btn"),

    // Media
    remoteVideo:  $("#remote-video"),
    videoPlaceholder: $("#video-placeholder"),
    responseAudio: $("#response-audio"),
    visualizer:   $("#audio-visualizer"),

    // RTC Controls
    connectBtn:    $("#connect-btn"),
    disconnectBtn: $("#disconnect-btn"),
    micBtn:        $("#mic-btn"),
    micLabel:      $(".mic-label"),

    // Config
    cfgLanguage:  $("#cfg-language"),
    cfgAvatar:    $("#cfg-avatar"),
    cfgVoice:     $("#cfg-voice"),
    cfgVideo:     $("#cfg-video"),
    cfgApply:     $("#cfg-apply"),

    // Latency
    latencyText:  $("#latency-text"),
};


// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

const state = {
    // WebSocket chat
    chatWs: null,
    chatSessionId: null,
    chatConnected: false,

    // WebRTC
    rtcWs: null,
    rtcSessionId: null,
    peerConnection: null,
    localStream: null,
    isMicActive: false,

    // Audio recording via MediaRecorder (for push-to-talk)
    mediaRecorder: null,
    audioChunks: [],

    // Visualizer
    audioContext: null,
    analyser: null,
    animFrameId: null,
};


// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

function getWsBase() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}`;
}

function setStatus(status, text) {
    dom.statusDot.className = `dot ${status}`;
    dom.statusText.textContent = text;
}

function addMessage(role, text, extras) {
    const div = document.createElement("div");
    div.className = `msg ${role}`;

    let html = `<div class="msg-text">${escapeHtml(text)}</div>`;

    if (extras) {
        const metaParts = [];
        if (extras.emotion) metaParts.push(`Emotion: ${extras.emotion}`);
        if (extras.latency) metaParts.push(`${extras.latency}ms`);
        if (metaParts.length) {
            html += `<div class="msg-meta">${metaParts.map(p => `<span>${p}</span>`).join("")}</div>`;
        }
        if (extras.audioUrl) {
            html += `<div class="msg-audio"><audio controls src="${extras.audioUrl}" preload="auto"></audio></div>`;
        }
        if (extras.videoUrl) {
            html += `<div class="msg-video"><video controls src="${extras.videoUrl}" preload="auto"></video></div>`;
        }
    }

    div.innerHTML = html;
    dom.messages.appendChild(div);
    dom.messages.scrollTop = dom.messages.scrollHeight;
}

function addSystemMessage(text) {
    const div = document.createElement("div");
    div.className = "msg system";
    div.textContent = text;
    dom.messages.appendChild(div);
    dom.messages.scrollTop = dom.messages.scrollHeight;
}

function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return str.replace(/[&<>"']/g, (c) => map[c]);
}


// ═══════════════════════════════════════════════════════════════════
// WebSocket Chat (/ws/chat)
// ═══════════════════════════════════════════════════════════════════

function connectChat() {
    if (state.chatWs) return;

    const url = `${getWsBase()}/ws/chat`;
    setStatus("connecting", "Connecting...");

    const ws = new WebSocket(url);
    state.chatWs = ws;

    ws.onopen = () => {
        // Wait for session_init message
    };

    ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        switch (data.type) {
            case "session_init":
                state.chatSessionId = data.session_id;
                state.chatConnected = true;
                setStatus("connected", "Connected");
                dom.sessionId.textContent = `Session: ${data.session_id}`;
                dom.chatInput.disabled = false;
                dom.sendBtn.disabled = false;
                addSystemMessage(data.message || "Connected to SmartTalker");
                break;

            case "text_response":
                addMessage("bot", data.text, {
                    emotion: data.emotion,
                    latency: data.latency_ms,
                    audioUrl: data.audio_url,
                    videoUrl: data.video_url,
                });
                dom.latencyText.textContent = `Last response: ${data.latency_ms}ms`;

                // Auto-play response audio
                if (data.audio_url) {
                    dom.responseAudio.src = data.audio_url;
                    dom.responseAudio.play().catch(() => {});
                }

                // Show video if available
                if (data.video_url) {
                    dom.remoteVideo.src = data.video_url;
                    dom.remoteVideo.style.display = "block";
                    dom.videoPlaceholder.style.display = "none";
                    dom.remoteVideo.play().catch(() => {});
                }
                break;

            case "audio_ack":
                addSystemMessage(`Audio received: ${(data.bytes_received / 1024).toFixed(1)} KB`);
                break;

            case "config_ack":
                addSystemMessage(`Settings updated (lang=${data.language}, avatar=${data.avatar_id})`);
                break;

            case "pong":
                break;

            case "error":
                addSystemMessage(`Error: ${data.error}${data.detail ? " - " + data.detail : ""}`);
                break;

            default:
                console.log("Unknown message type:", data.type);
        }
    };

    ws.onclose = () => {
        state.chatWs = null;
        state.chatConnected = false;
        setStatus("disconnected", "Disconnected");
        dom.chatInput.disabled = true;
        dom.sendBtn.disabled = true;
        addSystemMessage("Disconnected from server");
    };

    ws.onerror = () => {
        addSystemMessage("Connection error");
    };
}

function disconnectChat() {
    if (state.chatWs) {
        state.chatWs.close();
        state.chatWs = null;
    }
}

function sendTextChat() {
    const text = dom.chatInput.value.trim();
    if (!text || !state.chatWs || !state.chatConnected) return;

    state.chatWs.send(JSON.stringify({
        type: "text_chat",
        text: text,
        language: dom.cfgLanguage.value,
    }));

    addMessage("user", text);
    dom.chatInput.value = "";
}

function sendConfig() {
    if (!state.chatWs || !state.chatConnected) return;

    state.chatWs.send(JSON.stringify({
        type: "config",
        avatar_id: dom.cfgAvatar.value,
        voice_id: dom.cfgVoice.value || null,
        language: dom.cfgLanguage.value,
        enable_video: dom.cfgVideo.checked,
    }));
}


// ═══════════════════════════════════════════════════════════════════
// WebRTC (/ws/rtc) — Push-to-Talk with MediaRecorder
// ═══════════════════════════════════════════════════════════════════

async function connectWebRTC() {
    // First ensure chat WebSocket is connected
    if (!state.chatConnected) {
        connectChat();
        // Wait for chat connection
        await new Promise((resolve) => {
            const check = setInterval(() => {
                if (state.chatConnected) { clearInterval(check); resolve(); }
            }, 100);
            setTimeout(() => { clearInterval(check); resolve(); }, 5000);
        });
    }

    // Get microphone access
    try {
        state.localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000,
            },
            video: false,
        });
    } catch (err) {
        addSystemMessage(`Microphone access denied: ${err.message}`);
        return;
    }

    // Connect to signaling WebSocket
    const rtcUrl = `${getWsBase()}/ws/rtc`;
    const ws = new WebSocket(rtcUrl);
    state.rtcWs = ws;

    ws.onopen = () => {
        addSystemMessage("WebRTC signaling connected");
    };

    ws.onmessage = async (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        switch (data.type) {
            case "ready":
                state.rtcSessionId = data.session_id;
                addSystemMessage(`WebRTC session ready: ${data.session_id.slice(0, 8)}...`);
                await createPeerConnection();
                break;

            case "answer":
                if (state.peerConnection) {
                    await state.peerConnection.setRemoteDescription(
                        new RTCSessionDescription({ type: "answer", sdp: data.sdp })
                    );
                }
                break;

            case "ice_candidate":
                if (state.peerConnection && data.candidate) {
                    await state.peerConnection.addIceCandidate(
                        new RTCIceCandidate(data.candidate)
                    );
                }
                break;

            case "audio_response":
                // Server processed audio and returned response via signaling
                addMessage("bot", data.text || "", {
                    emotion: data.emotion,
                    latency: data.latency_ms,
                    audioUrl: data.audio_url,
                    videoUrl: data.video_url,
                });
                dom.latencyText.textContent = `Last response: ${data.latency_ms}ms`;
                if (data.audio_url) {
                    dom.responseAudio.src = data.audio_url;
                    dom.responseAudio.play().catch(() => {});
                }
                if (data.video_url) {
                    dom.remoteVideo.src = data.video_url;
                    dom.remoteVideo.style.display = "block";
                    dom.videoPlaceholder.style.display = "none";
                    dom.remoteVideo.play().catch(() => {});
                }
                break;

            case "error":
                addSystemMessage(`WebRTC error: ${data.error}`);
                break;
        }
    };

    ws.onclose = () => {
        state.rtcWs = null;
        addSystemMessage("WebRTC signaling disconnected");
        updateRTCControls(false);
    };

    ws.onerror = () => {
        addSystemMessage("WebRTC signaling connection error");
    };

    updateRTCControls(true);
}

async function createPeerConnection() {
    const config = {
        iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
        ],
    };

    const pc = new RTCPeerConnection(config);
    state.peerConnection = pc;

    // Add local audio track
    if (state.localStream) {
        state.localStream.getTracks().forEach((track) => {
            pc.addTrack(track, state.localStream);
        });
    }

    // Handle remote tracks (server may send audio/video back)
    pc.ontrack = (event) => {
        const [stream] = event.streams;
        if (event.track.kind === "video") {
            dom.remoteVideo.srcObject = stream;
            dom.remoteVideo.style.display = "block";
            dom.videoPlaceholder.style.display = "none";
        } else if (event.track.kind === "audio") {
            dom.responseAudio.srcObject = stream;
            dom.responseAudio.play().catch(() => {});
        }
    };

    // ICE candidate forwarding
    pc.onicecandidate = (event) => {
        if (event.candidate && state.rtcWs?.readyState === WebSocket.OPEN) {
            state.rtcWs.send(JSON.stringify({
                type: "ice_candidate",
                candidate: event.candidate.toJSON(),
            }));
        }
    };

    pc.onconnectionstatechange = () => {
        addSystemMessage(`WebRTC: ${pc.connectionState}`);
        if (pc.connectionState === "connected") {
            dom.micBtn.disabled = false;
            startVisualizer();
        }
        if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
            dom.micBtn.disabled = true;
        }
    };

    // Create and send offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    state.rtcWs.send(JSON.stringify({
        type: "offer",
        sdp: offer.sdp,
    }));
}

function disconnectWebRTC() {
    // Stop mic
    if (state.isMicActive) toggleMic();

    // Close peer connection
    if (state.peerConnection) {
        state.peerConnection.close();
        state.peerConnection = null;
    }

    // Stop local stream
    if (state.localStream) {
        state.localStream.getTracks().forEach((t) => t.stop());
        state.localStream = null;
    }

    // Close signaling WebSocket
    if (state.rtcWs) {
        if (state.rtcWs.readyState === WebSocket.OPEN) {
            state.rtcWs.send(JSON.stringify({ type: "hangup" }));
        }
        state.rtcWs.close();
        state.rtcWs = null;
    }

    // Stop visualizer
    stopVisualizer();

    // Reset video
    dom.remoteVideo.srcObject = null;
    dom.remoteVideo.src = "";
    dom.remoteVideo.style.display = "none";
    dom.videoPlaceholder.style.display = "flex";

    updateRTCControls(false);
    addSystemMessage("WebRTC disconnected");
}


// ═══════════════════════════════════════════════════════════════════
// Push-to-Talk (records mic audio, sends via /ws/chat as binary)
// ═══════════════════════════════════════════════════════════════════

function toggleMic() {
    if (state.isMicActive) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (!state.localStream) {
        addSystemMessage("No microphone access");
        return;
    }

    // Use the chat WebSocket for audio streaming (binary protocol)
    if (!state.chatWs || !state.chatConnected) {
        addSystemMessage("Chat not connected — connect first");
        return;
    }

    state.audioChunks = [];

    // Create MediaRecorder from the local stream
    const recorder = new MediaRecorder(state.localStream, {
        mimeType: getSupportedMimeType(),
    });
    state.mediaRecorder = recorder;

    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.audioChunks.push(event.data);
            // Stream chunk to server via WebSocket binary
            event.data.arrayBuffer().then((buf) => {
                if (state.chatWs?.readyState === WebSocket.OPEN) {
                    state.chatWs.send(buf);
                }
            });
        }
    };

    recorder.onstop = () => {
        // Signal audio_end to the chat WebSocket
        if (state.chatWs?.readyState === WebSocket.OPEN) {
            state.chatWs.send(JSON.stringify({ type: "audio_end" }));
        }
        addMessage("user", "[Voice message sent]");
    };

    // Signal audio_start
    const format = getAudioFormat();
    state.chatWs.send(JSON.stringify({
        type: "audio_start",
        format: format,
        language: dom.cfgLanguage.value,
    }));

    // Start recording in 250ms chunks for real-time streaming
    recorder.start(250);

    state.isMicActive = true;
    dom.micBtn.classList.remove("mic-off");
    dom.micBtn.classList.add("mic-on");
    dom.micBtn.querySelector(".mic-label").textContent = "Recording... (click to stop)";
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        state.mediaRecorder.stop();
    }
    state.mediaRecorder = null;
    state.isMicActive = false;

    dom.micBtn.classList.remove("mic-on");
    dom.micBtn.classList.add("mic-off");
    dom.micBtn.querySelector(".mic-label").textContent = "Push to Talk";
}

function getSupportedMimeType() {
    const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
    ];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return "audio/webm";
}

function getAudioFormat() {
    const mime = getSupportedMimeType();
    if (mime.includes("webm")) return "webm";
    if (mime.includes("ogg")) return "ogg";
    if (mime.includes("mp4")) return "m4a";
    return "webm";
}


// ═══════════════════════════════════════════════════════════════════
// Audio Visualizer
// ═══════════════════════════════════════════════════════════════════

function startVisualizer() {
    if (!state.localStream || state.animFrameId) return;

    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        state.audioContext = ctx;

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        state.analyser = analyser;

        const source = ctx.createMediaStreamSource(state.localStream);
        source.connect(analyser);

        drawVisualizer();
    } catch (err) {
        console.warn("Visualizer init failed:", err);
    }
}

function drawVisualizer() {
    const canvas = dom.visualizer;
    const canvasCtx = canvas.getContext("2d");
    const analyser = state.analyser;
    if (!analyser) return;

    const bufLen = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufLen);

    function draw() {
        state.animFrameId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = "#242836";
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufLen) * 2.5;
        let x = 0;

        for (let i = 0; i < bufLen; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            const hue = state.isMicActive ? 0 : 250;
            const saturation = state.isMicActive ? 80 : 60;
            canvasCtx.fillStyle = `hsl(${hue}, ${saturation}%, ${50 + (dataArray[i] / 255) * 30}%)`;
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    draw();
}

function stopVisualizer() {
    if (state.animFrameId) {
        cancelAnimationFrame(state.animFrameId);
        state.animFrameId = null;
    }
    if (state.audioContext) {
        state.audioContext.close().catch(() => {});
        state.audioContext = null;
    }
    state.analyser = null;

    // Clear canvas
    const canvas = dom.visualizer;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#242836";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}


// ═══════════════════════════════════════════════════════════════════
// UI Controls
// ═══════════════════════════════════════════════════════════════════

function updateRTCControls(connected) {
    dom.connectBtn.disabled = connected;
    dom.disconnectBtn.disabled = !connected;
    dom.micBtn.disabled = !connected;
    if (!connected) {
        dom.micBtn.classList.remove("mic-on");
        dom.micBtn.classList.add("mic-off");
        dom.micBtn.querySelector(".mic-label").textContent = "Push to Talk";
    }
}


// ═══════════════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════════════

dom.connectBtn.addEventListener("click", connectWebRTC);
dom.disconnectBtn.addEventListener("click", disconnectWebRTC);
dom.micBtn.addEventListener("click", toggleMic);

dom.sendBtn.addEventListener("click", sendTextChat);
dom.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendTextChat();
    }
});

dom.cfgApply.addEventListener("click", sendConfig);

// Keep-alive ping every 30s
setInterval(() => {
    if (state.chatWs?.readyState === WebSocket.OPEN) {
        state.chatWs.send(JSON.stringify({ type: "ping" }));
    }
}, 30000);


// ═══════════════════════════════════════════════════════════════════
// Auto-connect on page load
// ═══════════════════════════════════════════════════════════════════

window.addEventListener("load", () => {
    connectChat();
});

window.addEventListener("beforeunload", () => {
    disconnectWebRTC();
    disconnectChat();
});
