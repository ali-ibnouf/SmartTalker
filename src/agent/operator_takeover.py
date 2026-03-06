"""Operator takeover with cloned voice TTS.

When an operator takes over a session and sends text, this module:
1. Synthesizes the text through the employee's cloned voice (DashScope TTS)
2. Optionally renders a lip-synced video via RunPod (if avatar is video type)
3. Sends audio/video back to the visitor via the customer WS session

This provides a seamless experience — the visitor hears the employee's
voice even when a human operator is typing the responses.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("agent.operator_takeover")


async def synthesize_operator_message(
    *,
    text: str,
    pipeline: Any,
    avatar: Any,
    session_id: str,
    employee_id: str,
    customer_ws: Any,
    app: Any,
) -> dict:
    """Synthesize operator text through the employee's cloned voice.

    Args:
        text: The operator's text message.
        pipeline: SmartTalkerPipeline with TTS engine.
        avatar: Avatar record with voice_id and avatar_type.
        session_id: Current session ID.
        employee_id: Employee ID for billing/tracking.
        customer_ws: Customer WebSocket for sending audio/video.
        app: FastAPI app for accessing RunPod/R2 services.

    Returns:
        Dict with synthesis results (audio_duration_ms, video_url if rendered).
    """
    if not text or pipeline is None:
        return {"error": "Missing text or pipeline"}

    language = getattr(avatar, "language", "ar") if avatar else "ar"
    voice_id = getattr(avatar, "voice_id", "default") if avatar else "default"
    avatar_mode = getattr(avatar, "avatar_type", "vrm") if avatar else "vrm"

    result: dict = {"text": text}
    start = time.perf_counter()

    # Step 1: TTS synthesis
    try:
        tts_stream = await pipeline._tts.synthesize_stream(
            text=text,
            voice_id=voice_id,
            language=language,
            emotion="neutral",
        )
        audio_bytes = await tts_stream.collect_all()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        duration_ms = int(tts_stream.duration_seconds * 1000)

        # Send text + audio to visitor
        from starlette.websockets import WebSocketState
        if customer_ws and customer_ws.client_state == WebSocketState.CONNECTED:
            import json
            await customer_ws.send_text(json.dumps({
                "type": "text_response",
                "text": text,
                "emotion": "neutral",
                "session_id": session_id,
                "source": "operator",
            }))
            await customer_ws.send_text(json.dumps({
                "type": "audio_response",
                "audio": audio_b64,
                "duration_ms": duration_ms,
                "session_id": session_id,
            }))

        result["audio_duration_ms"] = duration_ms
        result["tts_cost_usd"] = tts_stream.cost_usd

    except Exception as exc:
        logger.warning(f"TTS synthesis failed for operator message: {exc}")
        result["tts_error"] = str(exc)
        return result

    # Step 2: Optional RunPod video render for video avatars
    if (
        avatar_mode == "video"
        and avatar
        and getattr(avatar, "photo_preprocessed", False)
        and getattr(avatar, "face_data_url", "")
        and audio_bytes
    ):
        try:
            from src.config import get_settings
            from src.services.runpod_client import RunPodServerless
            from src.services.r2_storage import R2Storage

            config = get_settings()
            r2 = R2Storage(config)
            runpod = RunPodServerless(config)

            audio_url = r2.upload_audio(f"op_{session_id}", audio_bytes)

            render_result = await runpod.render_lipsync(
                audio_url=audio_url,
                face_data_url=avatar.face_data_url,
                employee_id=employee_id,
                session_id=session_id,
            )

            import json
            from starlette.websockets import WebSocketState
            if customer_ws and customer_ws.client_state == WebSocketState.CONNECTED:
                await customer_ws.send_text(json.dumps({
                    "type": "video_url",
                    "url": render_result.video_url,
                    "session_id": session_id,
                }))

            result["video_url"] = render_result.video_url
            result["render_cost_usd"] = render_result.cost_usd

            await runpod.close()

        except Exception as exc:
            logger.warning(f"RunPod render failed for operator message: {exc}")
            result["render_error"] = str(exc)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result["total_ms"] = elapsed_ms

    logger.info(
        "Operator message synthesized",
        extra={"session_id": session_id, "duration_ms": elapsed_ms},
    )
    return result
