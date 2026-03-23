/**
 * Maskki Frontend — WebRTC + WebSocket Chat Client
 *
 * Features:
 *   - Avatar video clips (pre-generated via RunPod worker)
 *   - Auto camera/mic activation on session start
 *   - Document scanning and local save
 *   - Video frame relay to operator dashboard
 *   - Operator override message handling
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
    statusDot: $("#status-dot"),
    statusText: $("#status-text"),
    sessionId: $("#session-id-display"),

    // Chat
    messages: $("#chat-messages"),
    chatInput: $("#chat-input"),
    sendBtn: $("#send-btn"),

    // Media
    avatarVideo: $("#avatar-video"),
    localVideo: $("#local-video"),
    responseAudio: $("#response-audio"),
    visualizer: $("#audio-visualizer"),

    // RTC Controls
    connectBtn: $("#connect-btn"),
    disconnectBtn: $("#disconnect-btn"),
    micBtn: $("#mic-btn"),
    micLabel: $(".mic-label"),
    scanDocBtn: $("#scan-doc-btn"),

    // Scan
    scanCanvas: $("#scan-canvas"),
    scanDownload: $("#scan-download"),

    // Config
    cfgLanguage: $("#cfg-language"),
    cfgAvatar: $("#cfg-avatar"),
    cfgVoice: $("#cfg-voice"),
    cfgApply: $("#cfg-apply"),

    // VRM Elements
    vrmCanvas: $("#vrm-canvas"),

    // Latency
    latencyText: $("#latency-text"),
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

    // Video frame relay
    videoFrameInterval: null,

    // Avatar state
    avatarType: "video", // "video" or "vrm"
    currentClip: "idle",
};


// ═══════════════════════════════════════════════════════════════════
// Avatar Video Clips (pre-generated via RunPod worker)
// ═══════════════════════════════════════════════════════════════════

/**
 * Avatar clips are generated via the RunPod avatar-generation worker
 * and uploaded to the server. 4 states per avatar:
 *   idle.mp4, thinking.mp4, talking_happy.mp4, talking_sad.mp4
 *
 * Served from: /clips/{avatarId}/{state}.mp4
 */
function getAvatarClipsBase() {
    const avatarId = dom.cfgAvatar ? dom.cfgAvatar.value : "default";
    return `${getApiBase()}/clips/${avatarId}/`;
}

const EMOTION_TO_CLIP = {
    neutral: "talking_happy",
    happy: "talking_happy",
    sad: "talking_sad",
    surprised: "talking_happy",
    angry: "talking_sad",
    fearful: "talking_sad",
    disgusted: "talking_sad",
    contempt: "talking_sad",
};

function setAvatarClip(clipName) {
    if (state.avatarType === "vrm") {
        if (window.VRMAvatar) window.VRMAvatar.setEmotion(clipName);
        return;
    }

    if (!dom.avatarVideo) return;
    if (state.currentClip === clipName) return;

    state.currentClip = clipName;
    const src = `${getAvatarClipsBase()}${clipName}.mp4`;

    dom.avatarVideo.src = src;
    dom.avatarVideo.loop = (clipName === "idle" || clipName === "thinking");
    dom.avatarVideo.play().catch(() => { });
}

async function initAvatarClips() {
    // 1. Fetch info to check if avatar has VRM
    const avatarId = dom.cfgAvatar ? dom.cfgAvatar.value : "default";
    try {
        const res = await fetch(`${getApiBase()}/api/v1/avatars/${avatarId}/vrm-info`);
        if (res.ok) {
            const data = await res.json();
            state.avatarType = data.avatar_type;
            if (state.avatarType === "vrm" && data.vrm_url) {
                dom.avatarVideo.style.display = "none";
                dom.vrmCanvas.style.display = "block";

                // Init VRM module
                if (window.VRMAvatar) {
                    window.VRMAvatar.init("vrm-canvas");
                    window.VRMAvatar.load(data.vrm_url);
                }
            } else {
                dom.avatarVideo.style.display = "block";
                dom.vrmCanvas.style.display = "none";
                setAvatarClip("idle");
            }
        }
    } catch (err) {
        console.warn("Could not fetch vrm-info, defaulting to video", err);
        dom.avatarVideo.style.display = "block";
        if (dom.vrmCanvas) dom.vrmCanvas.style.display = "none";
        setAvatarClip("idle");
    }
}

function setupAvatarAudioSync() {
    // Kick off VRM Lip-sync analyzer on the Audio response 
    if (window.VRMAvatar) {
        // Must happen after a user gesture (like Connect or Push to talk)
        window.VRMAvatar.startLipSync("response-audio");
    }

    // When TTS audio finishes → return avatar to idle
    dom.responseAudio.addEventListener("ended", () => {
        setAvatarClip("idle");
    });
    dom.responseAudio.addEventListener("error", () => {
        setAvatarClip("idle");
    });
}


// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

function getWsBase() {
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") return "ws://localhost:8000";
    return "wss://ws.maskki.com";
}

function getApiBase() {
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") return "http://localhost:8000";
    return "https://ws.maskki.com";
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
                addSystemMessage(data.message || "Connected to Maskki");

                // Start lip-sync context logic if in VRM mode
                setupAvatarAudioSync();

                // Auto-activate camera & mic on session start
                autoActivateMedia();
                break;

            case "thinking":
                setAvatarClip("thinking");
                break;

            case "body_state":
                // Server tells us which clip to play
                if (data.clip_url && state.avatarType === "video") {
                    state.currentClip = data.state;
                    dom.avatarVideo.src = data.clip_url;
                    dom.avatarVideo.loop = (data.state === "idle" || data.state === "thinking");
                    dom.avatarVideo.play().catch(() => { });
                } else {
                    setAvatarClip(data.state || "neutral");
                }
                break;

            case "voice":
                // Play TTS audio; lip_sync data available in data.lip_sync
                if (data.audio_url) {
                    dom.responseAudio.src = data.audio_url;
                    dom.responseAudio.play().catch(() => { });
                }
                break;

            case "response":
                // Final text response with metadata
                addMessage("bot", data.text, {
                    emotion: data.emotion,
                    latency: data.latency_ms,
                });
                dom.latencyText.textContent = `Last response: ${data.latency_ms}ms`;

                // If no audio was sent, return to idle after short delay
                if (!dom.responseAudio.src || dom.responseAudio.paused) {
                    setTimeout(() => setAvatarClip("idle"), 2000);
                }
                break;

            case "operator_override":
                addMessage("bot", data.text, { emotion: "operator" });
                break;

            case "document_saved":
                addSystemMessage(`Document saved: ${data.filename} (${(data.size_bytes / 1024).toFixed(1)} KB)`);
                break;

            case "audio_ack":
                break;

            case "config_ack":
                addSystemMessage(`Settings updated (lang=${data.language}, avatar=${data.avatar_id})`);
                break;

            case "pong":
                break;

            case "error":
                addSystemMessage(`Error: ${data.error}${data.detail ? " - " + data.detail : ""}`);
                setAvatarClip("idle");
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

        // Release camera/mic on disconnect
        releaseMedia();
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

    // Avatar → thinking while waiting for response
    setAvatarClip("thinking");
}

function sendConfig() {
    if (!state.chatWs || !state.chatConnected) return;

    state.chatWs.send(JSON.stringify({
        type: "config",
        avatar_id: dom.cfgAvatar.value,
        voice_id: dom.cfgVoice.value || null,
        language: dom.cfgLanguage.value,
    }));

    // Also re-init avatar visual representations since avatar_id changed
    initAvatarClips();
}


// ═══════════════════════════════════════════════════════════════════
// Auto Camera/Mic Activation
// ═══════════════════════════════════════════════════════════════════

async function autoActivateMedia() {
    try {
        state.localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000,
            },
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
            },
        });

        // Show local camera in PiP
        dom.localVideo.srcObject = state.localStream;
        dom.localVideo.style.display = "block";

        // Enable controls
        dom.micBtn.disabled = false;
        dom.scanDocBtn.disabled = false;
        dom.disconnectBtn.disabled = false;
        dom.connectBtn.disabled = true;

        // Start audio visualizer
        startVisualizer();

        // Start relaying video frames to the server (for operator)
        startVideoFrameRelay();

        addSystemMessage("📹 Camera & microphone activated");

    } catch (err) {
        addSystemMessage(`⚠️ Camera/mic access denied: ${err.message}`);
        // Try audio-only fallback
        try {
            state.localStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true },
                video: false,
            });
            dom.micBtn.disabled = false;
            dom.disconnectBtn.disabled = false;
            dom.connectBtn.disabled = true;
            startVisualizer();
            addSystemMessage("🎤 Microphone activated (camera unavailable)");
        } catch (err2) {
            addSystemMessage(`❌ No media access: ${err2.message}`);
        }
    }
}

function releaseMedia() {
    stopVideoFrameRelay();

    if (state.localStream) {
        state.localStream.getTracks().forEach((t) => t.stop());
        state.localStream = null;
    }

    dom.localVideo.srcObject = null;
    dom.localVideo.style.display = "none";

    stopVisualizer();

    dom.micBtn.disabled = true;
    dom.scanDocBtn.disabled = true;
    dom.disconnectBtn.disabled = true;
    dom.connectBtn.disabled = false;

    // Reset avatar to idle
    setAvatarClip("idle");
}


// ═══════════════════════════════════════════════════════════════════
// Video Frame Relay (customer camera → operator via WebSocket)
// ═══════════════════════════════════════════════════════════════════

function startVideoFrameRelay() {
    if (state.videoFrameInterval) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    state.videoFrameInterval = setInterval(() => {
        if (!state.localStream || !state.chatWs || !state.chatConnected) return;

        const videoTrack = state.localStream.getVideoTracks()[0];
        if (!videoTrack || !videoTrack.enabled) return;

        const video = dom.localVideo;
        if (video.videoWidth === 0 || video.videoHeight === 0) return;

        canvas.width = 320;
        canvas.height = Math.round(320 * (video.videoHeight / video.videoWidth));
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.5);
        const base64 = dataUrl.split(",")[1];

        state.chatWs.send(JSON.stringify({
            type: "video_frame",
            frame: base64,
        }));
    }, 500);
}

function stopVideoFrameRelay() {
    if (state.videoFrameInterval) {
        clearInterval(state.videoFrameInterval);
        state.videoFrameInterval = null;
    }
}


// ═══════════════════════════════════════════════════════════════════
// Document Scanning
// ═══════════════════════════════════════════════════════════════════

function captureDocument() {
    if (!state.localStream) {
        addSystemMessage("⚠️ Camera not available for document scan");
        return;
    }

    const videoTrack = state.localStream.getVideoTracks()[0];
    if (!videoTrack) {
        addSystemMessage("⚠️ No camera track available");
        return;
    }

    const video = dom.localVideo;
    const canvas = dom.scanCanvas;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Flash effect
    dom.localVideo.classList.add("scan-flash");
    setTimeout(() => dom.localVideo.classList.remove("scan-flash"), 400);

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `document_${timestamp}.jpg`;

    canvas.toBlob((blob) => {
        if (!blob) return;

        const url = URL.createObjectURL(blob);
        const link = dom.scanDownload;
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);

        addSystemMessage(`📄 Document captured: ${filename}`);

        const reader = new FileReader();
        reader.onload = () => {
            const base64 = reader.result.split(",")[1];
            if (state.chatWs && state.chatConnected) {
                state.chatWs.send(JSON.stringify({
                    type: "document_upload",
                    image: base64,
                    filename: filename,
                }));
            }
        };
        reader.readAsDataURL(blob);
    }, "image/jpeg", 0.92);
}


// ═══════════════════════════════════════════════════════════════════
// WebRTC (/ws/rtc) — Push-to-Talk with MediaRecorder
// ═══════════════════════════════════════════════════════════════════

async function connectWebRTC() {
    if (!state.chatConnected) {
        connectChat();
        await new Promise((resolve) => {
            const check = setInterval(() => {
                if (state.chatConnected) { clearInterval(check); resolve(); }
            }, 100);
            setTimeout(() => { clearInterval(check); resolve(); }, 5000);
        });
    }

    if (!state.localStream) {
        try {
            state.localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000,
                },
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user",
                },
            });
            dom.localVideo.srcObject = state.localStream;
            dom.localVideo.style.display = "block";
        } catch (err) {
            addSystemMessage(`Microphone access denied: ${err.message}`);
            return;
        }
    }

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
                addMessage("bot", data.text || "", {
                    emotion: data.emotion,
                    latency: data.latency_ms,
                    audioUrl: data.audio_url,
                });
                dom.latencyText.textContent = `Last response: ${data.latency_ms}ms`;

                setAvatarClip(EMOTION_TO_CLIP[data.emotion] || "talking_happy");

                if (data.audio_url) {
                    dom.responseAudio.src = data.audio_url;
                    dom.responseAudio.play().catch(() => { });
                } else {
                    setTimeout(() => setAvatarClip("idle"), 2000);
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

    if (state.localStream) {
        state.localStream.getAudioTracks().forEach((track) => {
            pc.addTrack(track, state.localStream);
        });
    }

    // Handle remote audio tracks (avatar is 3D — not streamed)
    pc.ontrack = (event) => {
        if (event.track.kind === "audio") {
            const [stream] = event.streams;
            dom.responseAudio.srcObject = stream;
            dom.responseAudio.play().catch(() => { });
            setupAvatarAudioSync();
        }
    };

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
        }
        if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
            dom.micBtn.disabled = true;
        }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    state.rtcWs.send(JSON.stringify({
        type: "offer",
        sdp: offer.sdp,
    }));
}

function disconnectWebRTC() {
    if (state.isMicActive) toggleMic();

    if (state.peerConnection) {
        state.peerConnection.close();
        state.peerConnection = null;
    }

    releaseMedia();

    if (state.rtcWs) {
        if (state.rtcWs.readyState === WebSocket.OPEN) {
            state.rtcWs.send(JSON.stringify({ type: "hangup" }));
        }
        state.rtcWs.close();
        state.rtcWs = null;
    }

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

    if (!state.chatWs || !state.chatConnected) {
        addSystemMessage("Chat not connected — connect first");
        return;
    }

    state.audioChunks = [];

    const audioStream = new MediaStream(state.localStream.getAudioTracks());

    const recorder = new MediaRecorder(audioStream, {
        mimeType: getSupportedMimeType(),
    });
    state.mediaRecorder = recorder;

    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.audioChunks.push(event.data);
            event.data.arrayBuffer().then((buf) => {
                if (state.chatWs?.readyState === WebSocket.OPEN) {
                    state.chatWs.send(buf);
                }
            });
        }
    };

    recorder.onstop = () => {
        if (state.chatWs?.readyState === WebSocket.OPEN) {
            state.chatWs.send(JSON.stringify({ type: "audio_end" }));
        }
        addMessage("user", "[Voice message sent]");
        // Avatar → thinking while waiting for response
        setAvatarClip("thinking");
    };

    const format = getAudioFormat();
    state.chatWs.send(JSON.stringify({
        type: "audio_start",
        format: format,
        language: dom.cfgLanguage.value,
    }));

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
        state.audioContext.close().catch(() => { });
        state.audioContext = null;
    }
    state.analyser = null;

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
    dom.scanDocBtn.disabled = !connected;
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
dom.disconnectBtn.addEventListener("click", () => {
    disconnectWebRTC();
    disconnectChat();
});
dom.micBtn.addEventListener("click", toggleMic);
dom.scanDocBtn.addEventListener("click", captureDocument);

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
    initAvatarClips();
    setupAvatarAudioSync();
    connectChat();
});

window.addEventListener("beforeunload", () => {
    releaseMedia();
    disconnectWebRTC();
    disconnectChat();
});
