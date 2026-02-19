"""WebRTC integration for browser-based real-time video chat.

Handles peer connection setup, media tracks, audio recording,
pipeline processing, and ICE candidate negotiation.

Signaling Protocol (JSON over WebSocket at /ws/rtc):
    Client -> Server: offer, ice_candidate, process, hangup
    Server -> Client: answer, ice_candidate, ready, audio_response, error

Audio Flow:
    1. Browser sends mic audio via WebRTC audio track
    2. Server records incoming audio frames to a temp WAV file
    3. Client sends {"type": "process"} to trigger pipeline processing
    4. Server stops recording, runs ASR -> LLM -> TTS pipeline
    5. Server sends response (text + audio URL) back via signaling WebSocket
"""

from __future__ import annotations

import asyncio
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from src.config import Settings
from src.pipeline.orchestrator import SmartTalkerPipeline
from src.utils.exceptions import WebRTCError
from src.utils.logger import setup_logger

logger = setup_logger("integrations.webrtc")

# Audio recording settings
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_CHANNELS = 1
_AUDIO_SAMPLE_WIDTH = 2  # 16-bit PCM

try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCConfiguration,
        RTCIceServer,
    )
    from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    RTCPeerConnection = Any
    RTCSessionDescription = Any
    RTCConfiguration = Any
    RTCIceServer = Any


@dataclass
class WebRTCSession:
    """Single WebRTC peer connection session.

    Wraps aiortc.RTCPeerConnection to manage media tracks,
    audio recording, and pipeline processing lifecycle.
    """

    session_id: str
    peer_connection: Any  # aiortc.RTCPeerConnection
    pipeline: SmartTalkerPipeline
    config: Settings
    storage_dir: Path = field(default_factory=lambda: Path("./outputs/webrtc"))

    # Audio recording state
    _audio_frames: list[bytes] = field(default_factory=list)
    _is_recording: bool = False
    _audio_track: Any = None
    _record_task: Optional[asyncio.Task] = None
    _audio_task: Optional[asyncio.Task] = None

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self) -> None:
        """Mark session as recording incoming audio frames."""
        self._audio_frames.clear()
        self._is_recording = True
        logger.info(f"Recording started for session {self.session_id}")

    def stop_recording(self) -> None:
        """Stop recording and cancel the frame capture task."""
        self._is_recording = False
        if self._record_task and not self._record_task.done():
            self._record_task.cancel()
        logger.info(f"Recording stopped for session {self.session_id}")

    def save_audio(self) -> Optional[Path]:
        """Save accumulated audio frames to a WAV file.

        Returns:
            Path to the saved WAV file, or None if no frames.
        """
        if not self._audio_frames:
            return None

        filename = f"rtc_{self.session_id[:12]}_{uuid.uuid4().hex[:8]}.wav"
        wav_path = self.storage_dir / filename

        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(_AUDIO_CHANNELS)
                wf.setsampwidth(_AUDIO_SAMPLE_WIDTH)
                wf.setframerate(_AUDIO_SAMPLE_RATE)
                for frame_data in self._audio_frames:
                    wf.writeframes(frame_data)

            size_kb = wav_path.stat().st_size / 1024
            logger.info(
                f"Audio saved: {filename}",
                extra={"size_kb": round(size_kb, 1), "frames": len(self._audio_frames)},
            )
            return wav_path

        except Exception as exc:
            logger.error(f"Failed to save audio: {exc}")
            return None

    async def capture_audio_frames(self, track: Any) -> None:
        """Continuously capture audio frames from a WebRTC audio track.

        Runs in a background task. Each frame's PCM data is accumulated
        in _audio_frames when _is_recording is True.

        Args:
            track: aiortc MediaStreamTrack (audio).
        """
        self._audio_track = track
        self._is_recording = True

        try:
            while True:
                frame = await track.recv()

                if self._is_recording:
                    # Convert AudioFrame to raw PCM bytes
                    # aiortc AudioFrame has .to_ndarray() for PCM data
                    try:
                        nd = frame.to_ndarray()
                        # Resample to mono 16kHz 16-bit if needed
                        pcm_bytes = nd.tobytes()
                        self._audio_frames.append(pcm_bytes)
                    except Exception:
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(f"Audio capture ended: {exc}")

    async def close(self) -> None:
        """Close peer connection and cleanup resources."""
        self._is_recording = False

        if self._record_task and not self._record_task.done():
            self._record_task.cancel()
            try:
                await self._record_task
            except asyncio.CancelledError:
                pass

        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass

        if self.peer_connection:
            await self.peer_connection.close()

        # Clean up temp audio files
        for f in self.storage_dir.glob(f"rtc_{self.session_id[:12]}_*"):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

        logger.info(f"WebRTC session closed: {self.session_id}")


class WebRTCSignalingHandler:
    """Manages WebRTC signaling, audio capture, and pipeline processing."""

    def __init__(self, pipeline: SmartTalkerPipeline, config: Settings) -> None:
        """Initialize the handler.

        Args:
            pipeline: SmartTalkerPipeline for audio processing.
            config: Application settings.
        """
        self.pipeline = pipeline
        self.config = config
        self._sessions: dict[str, WebRTCSession] = {}
        self._storage_dir = config.storage_base_dir / "webrtc"
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket signaling connection.

        Manages the full lifecycle: accept, create peer connection,
        handle signaling messages, and clean up on disconnect.

        Args:
            websocket: The active WebSocket connection.
        """
        await websocket.accept()

        if not AIORTC_AVAILABLE:
            logger.error("aiortc not installed — WebRTC disabled")
            await websocket.send_json({
                "type": "error",
                "error": "WebRTC not available (aiortc not installed)",
            })
            await websocket.close(code=1011, reason="WebRTC not supported")
            return

        session_id = str(uuid.uuid4())

        # Configure ICE servers
        ice_servers = []
        if self.config.webrtc_stun_servers:
            for url in self.config.webrtc_stun_servers.split(","):
                url = url.strip()
                if url:
                    ice_servers.append(RTCIceServer(urls=url))

        if self.config.webrtc_turn_server:
            ice_servers.append(RTCIceServer(
                urls=self.config.webrtc_turn_server,
                username=self.config.webrtc_turn_username,
                credential=self.config.webrtc_turn_password,
            ))

        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )

        session = WebRTCSession(
            session_id=session_id,
            peer_connection=pc,
            pipeline=self.pipeline,
            config=self.config,
            storage_dir=self._storage_dir,
        )
        self._sessions[session_id] = session

        # ── Track handlers ────────────────────────────────────────
        @pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.info(f"Audio track received for session {session_id}")
                session._record_task = asyncio.create_task(
                    session.capture_audio_frames(track)
                )

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track ended for session {session_id}")
                session.stop_recording()

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            logger.info(
                f"WebRTC connection state: {state}",
                extra={"session_id": session_id},
            )
            if state in ("failed", "closed"):
                session.stop_recording()

        # ── Send ready ────────────────────────────────────────────
        try:
            await websocket.send_json({
                "type": "ready",
                "session_id": session_id,
            })

            # ── Signaling loop ────────────────────────────────────
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "offer":
                    await self._handle_offer(pc, websocket, message)

                elif msg_type == "ice_candidate":
                    candidate = message.get("candidate")
                    if candidate and AIORTC_AVAILABLE:
                        try:
                            from aiortc import RTCIceCandidate
                            # Parse candidate fields from signaling message
                            sdp_mid = message.get("sdpMid", "")
                            sdp_mline_index = message.get("sdpMLineIndex", 0)
                            await pc.addIceCandidate(
                                RTCIceCandidate(
                                    sdpMid=sdp_mid,
                                    sdpMLineIndex=sdp_mline_index,
                                    candidate=candidate,
                                )
                            )
                            logger.debug(f"Added ICE candidate: {candidate[:60]}")
                        except Exception as ice_exc:
                            logger.warning(f"Failed to add ICE candidate: {ice_exc}")
                    else:
                        logger.debug(f"Received ICE candidate (ignored): {candidate}")

                elif msg_type == "process":
                    await self._handle_process(session, websocket, message)

                elif msg_type == "hangup":
                    break

                else:
                    logger.warning(f"Unknown signaling message: {msg_type}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as exc:
            logger.error(f"Signaling error: {exc}")
            try:
                await websocket.send_json({"type": "error", "error": str(exc)})
            except Exception:
                pass
        finally:
            await session.close()
            self._sessions.pop(session_id, None)

    async def _handle_offer(
        self,
        pc: Any,
        websocket: WebSocket,
        message: dict,
    ) -> None:
        """Handle an SDP offer and return an answer.

        Args:
            pc: RTCPeerConnection instance.
            websocket: The signaling WebSocket.
            message: Parsed offer message with "sdp" field.
        """
        offer = RTCSessionDescription(sdp=message["sdp"], type="offer")
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        await websocket.send_json({
            "type": "answer",
            "sdp": pc.localDescription.sdp,
        })
        logger.info("SDP offer/answer exchanged")

    async def _handle_process(
        self,
        session: WebRTCSession,
        websocket: WebSocket,
        message: dict,
    ) -> None:
        """Process recorded audio through the pipeline and send response.

        Stops recording, saves audio to WAV, runs the pipeline,
        and sends the result back via the signaling WebSocket.

        Args:
            session: The active WebRTC session.
            websocket: The signaling WebSocket for response delivery.
            message: Parsed process message (optional language field).
        """
        session.stop_recording()

        # Save recorded audio
        wav_path = session.save_audio()
        if not wav_path:
            await websocket.send_json({
                "type": "error",
                "error": "No audio recorded",
                "detail": "Press the mic button and speak before processing",
            })
            # Restart recording for next utterance
            session.start_recording()
            return

        language = message.get("language", "ar")

        try:
            result = await session.pipeline.process_audio(
                audio_path=str(wav_path),
                session_id=session.session_id,
                language=language,
            )

            # Build response with file URLs
            audio_url = f"/files/{Path(result.audio_path).name}" if result.audio_path else None
            video_url = f"/files/{Path(result.video_path).name}" if result.video_path else None

            # Copy audio to static files directory for serving
            if result.audio_path:
                static_dir = session.config.static_files_dir
                static_dir.mkdir(parents=True, exist_ok=True)
                src_path = Path(result.audio_path)
                if src_path.exists() and not str(src_path).startswith(str(static_dir)):
                    import shutil
                    shutil.copy2(str(src_path), str(static_dir / src_path.name))

            if result.video_path:
                static_dir = session.config.static_files_dir
                vid_path = Path(result.video_path)
                if vid_path.exists() and not str(vid_path).startswith(str(static_dir)):
                    import shutil
                    shutil.copy2(str(vid_path), str(static_dir / vid_path.name))

            await websocket.send_json({
                "type": "audio_response",
                "text": result.response_text,
                "emotion": result.detected_emotion,
                "audio_url": audio_url,
                "video_url": video_url,
                "latency_ms": result.total_latency_ms,
                "breakdown": result.breakdown,
            })

        except Exception as exc:
            logger.error(f"Pipeline processing failed: {exc}")
            await websocket.send_json({
                "type": "error",
                "error": "Pipeline processing failed",
                "detail": str(exc),
            })
        finally:
            # Clean up recorded input audio
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

            # Restart recording for next utterance
            session.start_recording()


async def webrtc_signaling_endpoint(
    websocket: WebSocket,
    handler: WebRTCSignalingHandler,
) -> None:
    """WebSocket endpoint for WebRTC signaling.

    Mounted by main.py at /ws/rtc.

    Args:
        websocket: The active WebSocket connection.
        handler: Instance of WebRTCSignalingHandler.
    """
    await handler.handle_connection(websocket)
