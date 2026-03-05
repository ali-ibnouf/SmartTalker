"""Integration tests for Phase 2 features.

11 scenarios covering:
01 - Full voice session (WebSocket /session)
02 - VRM fallback when RunPod render fails
03 - RunPod failure recovery (timeout handling)
04 - Tool calling (LLM → tool → response)
05 - Workflow execution (lead_capture template)
06 - Returning visitor (VisitorMemory loaded)
07 - Auto-learning (extract QA pairs from session)
08 - Billing near-zero (balance almost depleted)
09 - Cost tracking (_record_cost writes APICostRecord)
10 - Widget action_required (requires_confirmation tool)
11 - KB pre-match (FAQ match → KB context in response)

Environment: Python 3.10.11 on Windows, no numpy/torch.
All external services (DashScope, RunPod, DB) are mocked.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio


# ── Shared Helpers ──────────────────────────────────────────────────────────


def _make_mock_config():
    """Create a minimal mock config that satisfies Settings requirements."""
    config = MagicMock()
    config.api_key = "test-api-key-123"
    config.dashscope_api_key = "test-dashscope-key"
    config.dashscope_ws_url = "wss://test.example.com/ws"
    config.asr_model = "qwen3-asr-flash-realtime"
    config.llm_base_url = "https://llm.example.com/v1"
    config.llm_model_name = "qwen3-max"
    config.llm_api_key = "test-llm-key"
    config.llm_timeout = 30
    config.llm_max_tokens = 1024
    config.llm_temperature = 0.7
    config.llm_max_history = 10
    config.billing_enabled = True
    config.billing_rate_per_second = 0.001
    config.billing_grace_period_s = 0
    config.database_url = "sqlite+aiosqlite://"
    config.debug = False
    config.cors_origins = "*"
    config.storage_base_dir = MagicMock()
    config.static_files_dir = MagicMock()
    config.clips_dir = MagicMock()
    config.redis_url = "redis://localhost"
    config.webrtc_enabled = False
    config.runpod_api_key = "test-runpod-key"
    config.runpod_render_endpoint = "https://runpod.example.com/render"
    config.runpod_preprocess_endpoint = "https://runpod.example.com/preprocess"
    config.r2_access_key = "test-r2-access"
    config.r2_secret_key = "test-r2-secret"
    config.r2_bucket_name = "test-bucket"
    config.r2_endpoint_url = "https://r2.example.com"
    config.r2_public_url = "https://cdn.example.com"
    return config


def _make_mock_avatar(
    avatar_type: str = "vrm",
    photo_preprocessed: bool = False,
    face_data_url: str = "",
    voice_id: str = "default",
    language: str = "ar",
):
    """Create a mock avatar record."""
    avatar = MagicMock()
    avatar.id = "avatar-001"
    avatar.customer_id = "cust-001"
    avatar.name = "Test Avatar"
    avatar.avatar_type = avatar_type
    avatar.photo_preprocessed = photo_preprocessed
    avatar.face_data_url = face_data_url
    avatar.voice_id = voice_id
    avatar.language = language
    avatar.voice_model = "qwen3-tts-vc-realtime"
    avatar.vrm_url = "https://cdn.example.com/model.vrm"
    return avatar


def _make_mock_employee():
    """Create a mock Employee ORM object."""
    emp = MagicMock()
    emp.id = "emp-001"
    emp.customer_id = "cust-001"
    emp.name = "Sara"
    emp.role_title = "Customer Support"
    emp.role_description = "Handles customer queries professionally."
    emp.personality = json.dumps({"tone": "friendly", "style": "concise"})
    emp.guardrails = json.dumps({"blocked_topics": ["politics"], "max_response_length": 500})
    emp.language = "en"
    emp.is_active = True
    return emp


def _make_mock_db():
    """Create a mock Database with a session context manager."""
    db = MagicMock()
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None), scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    db.session = MagicMock(return_value=_AsyncContextManager(mock_session))
    return db, mock_session


class _AsyncContextManager:
    """Helper that wraps a mock into an async context manager."""

    def __init__(self, mock_obj):
        self._mock = mock_obj

    async def __aenter__(self):
        return self._mock

    async def __aexit__(self, *args):
        pass


# ── Fake ASR/TTS results ────────────────────────────────────────────────────


@dataclass
class FakeASRResult:
    text: str = "Hello, how are you?"
    language: str = "en"
    cost_usd: float = 0.0001
    latency_ms: int = 120


@dataclass
class FakeTTSStream:
    cost_usd: float = 0.0002
    duration_seconds: float = 1.5
    _audio: bytes = b"\x00\x01\x02" * 100

    async def collect_all(self) -> bytes:
        return self._audio


@dataclass
class FakeLLMResult:
    text: str = "I'm doing great, thanks for asking!"
    emotion: str = "happy"
    cost_usd: float = 0.0005
    tokens_used: int = 42
    latency_ms: int = 200


@dataclass
class FakeRenderResult:
    video_url: str = "https://cdn.example.com/render/output.mp4"
    cost_usd: float = 0.001
    execution_time_ms: int = 3000
    job_id: str = "runpod-job-123"


# =============================================================================
# Test 01: Full Voice Session
# =============================================================================


class TestPhase2_01_FullVoiceSession:
    """Test the WebSocket /session endpoint end-to-end.

    Flow: auth → audio_chunk → audio_end → expect text_response + audio_response.
    """

    @pytest.mark.asyncio
    async def test_phase2_01_full_voice_session(self):
        from src.api.ws_visitor import visitor_session_handler, _authenticate, _record_cost

        # Build mock websocket
        ws = AsyncMock()
        ws.accept = AsyncMock()

        # Build mock app with pipeline
        app = MagicMock()
        app.state.billing = None  # No billing for this test
        app.state.db = None

        asr_session_mock = AsyncMock()
        asr_session_mock.send_audio = AsyncMock()
        asr_session_mock.finish = AsyncMock(return_value=FakeASRResult())

        pipeline = MagicMock()
        pipeline._asr.create_session = AsyncMock(return_value=asr_session_mock)
        pipeline._tts.synthesize_stream = AsyncMock(return_value=FakeTTSStream())
        pipeline._llm.generate = AsyncMock(return_value=FakeLLMResult())
        app.state.pipeline = pipeline
        app.state.agent_engine = None  # Use direct LLM fallback

        ws.app = app

        # Prepare message sequence
        auth_msg = json.dumps({"type": "auth", "token": "jwt-token", "employee_id": "emp-001"})
        audio_chunk_msg = json.dumps({"type": "audio_chunk", "audio": base64.b64encode(b"\x00" * 320).decode()})
        audio_end_msg = json.dumps({"type": "audio_end"})

        message_queue = [auth_msg, audio_chunk_msg, audio_end_msg]
        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            if call_count < len(message_queue):
                msg = message_queue[call_count]
                call_count += 1
                return msg
            # Simulate disconnect after all messages processed
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect(code=1000)

        ws.receive_text = AsyncMock(side_effect=receive_side_effect)
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()

        # Mock authentication to succeed
        mock_avatar = _make_mock_avatar()
        with patch("src.api.ws_visitor._authenticate", new=AsyncMock(return_value=("cust-001", mock_avatar))):
            with patch("src.api.ws_visitor.get_settings", return_value=_make_mock_config()):
                await visitor_session_handler(ws)

        # Verify websocket was accepted
        ws.accept.assert_called_once()

        # Gather all sent messages
        sent_messages = []
        for call in ws.send_text.call_args_list:
            sent_messages.append(json.loads(call[0][0]))

        # Expect: auth_ok, text_response, audio_response
        msg_types = [m["type"] for m in sent_messages]
        assert "auth_ok" in msg_types, f"Expected auth_ok, got: {msg_types}"
        assert "text_response" in msg_types, f"Expected text_response, got: {msg_types}"
        assert "audio_response" in msg_types, f"Expected audio_response, got: {msg_types}"

        # Verify text_response content
        text_resp = next(m for m in sent_messages if m["type"] == "text_response")
        assert text_resp["text"] == "I'm doing great, thanks for asking!"
        assert "session_id" in text_resp

        # Verify audio_response has base64 audio
        audio_resp = next(m for m in sent_messages if m["type"] == "audio_response")
        assert len(audio_resp["audio"]) > 0
        assert audio_resp["duration_ms"] == 1500


# =============================================================================
# Test 02: VRM Fallback
# =============================================================================


class TestPhase2_02_VRMFallback:
    """When RunPod render fails, the system should send fallback_vrm."""

    @pytest.mark.asyncio
    async def test_phase2_02_vrm_fallback(self):
        from src.api.ws_visitor import _process_and_respond

        ws = AsyncMock()
        ws.send_text = AsyncMock()

        app = MagicMock()
        app.state.db = None
        app.state.agent_engine = None

        pipeline = MagicMock()
        pipeline._llm.generate = AsyncMock(return_value=FakeLLMResult())
        pipeline._tts.synthesize_stream = AsyncMock(return_value=FakeTTSStream())

        # Video avatar with face data (triggers RunPod path)
        avatar = _make_mock_avatar(
            avatar_type="video",
            photo_preprocessed=True,
            face_data_url="https://cdn.example.com/faces/data.pkl",
        )

        # Mock RunPod to raise an exception
        with patch("src.api.ws_visitor.get_settings", return_value=_make_mock_config()):
            with patch("src.services.runpod_client.RunPodServerless") as MockRunPod:
                mock_runpod_instance = AsyncMock()
                mock_runpod_instance.render_lipsync = AsyncMock(
                    side_effect=Exception("RunPod render failed: GPU OOM")
                )
                MockRunPod.return_value = mock_runpod_instance

                with patch("src.services.r2_storage.R2Storage") as MockR2:
                    mock_r2_instance = MagicMock()
                    mock_r2_instance.upload_audio = MagicMock(return_value="https://cdn.example.com/audio.wav")
                    MockR2.return_value = mock_r2_instance

                    await _process_and_respond(
                        websocket=ws,
                        pipeline=pipeline,
                        app=app,
                        agent_engine=None,
                        text="What are your store hours?",
                        session_id="test-session-123",
                        employee_id="emp-001",
                        customer_id="cust-001",
                        avatar=avatar,
                        avatar_mode="video",
                        turn_start=time.perf_counter(),
                    )

        # Parse sent messages
        sent = [json.loads(c[0][0]) for c in ws.send_text.call_args_list]
        msg_types = [m["type"] for m in sent]

        assert "text_response" in msg_types
        assert "audio_response" in msg_types
        assert "fallback_vrm" in msg_types, f"Expected fallback_vrm but got: {msg_types}"

        fallback = next(m for m in sent if m["type"] == "fallback_vrm")
        assert "reason" in fallback
        assert "RunPod render failed" in fallback["reason"]

        # video_url should NOT be present
        assert "video_url" not in msg_types


# =============================================================================
# Test 03: RunPod Failure Recovery (Timeout)
# =============================================================================


class TestPhase2_03_RunPodFailureRecovery:
    """Test that RunPod timeout results in graceful fallback_vrm."""

    @pytest.mark.asyncio
    async def test_phase2_03_runpod_timeout_recovery(self):
        from src.api.ws_visitor import _process_and_respond

        ws = AsyncMock()
        ws.send_text = AsyncMock()

        app = MagicMock()
        app.state.db = None
        app.state.agent_engine = None

        pipeline = MagicMock()
        pipeline._llm.generate = AsyncMock(return_value=FakeLLMResult())
        pipeline._tts.synthesize_stream = AsyncMock(return_value=FakeTTSStream())

        avatar = _make_mock_avatar(
            avatar_type="video",
            photo_preprocessed=True,
            face_data_url="https://cdn.example.com/faces/data.pkl",
        )

        # Mock RunPod to raise a timeout
        with patch("src.api.ws_visitor.get_settings", return_value=_make_mock_config()):
            with patch("src.services.runpod_client.RunPodServerless") as MockRunPod:
                mock_runpod_instance = AsyncMock()
                mock_runpod_instance.render_lipsync = AsyncMock(
                    side_effect=asyncio.TimeoutError("RunPod job timed out after 30s")
                )
                MockRunPod.return_value = mock_runpod_instance

                with patch("src.services.r2_storage.R2Storage") as MockR2:
                    mock_r2_instance = MagicMock()
                    mock_r2_instance.upload_audio = MagicMock(return_value="https://cdn.example.com/audio.wav")
                    MockR2.return_value = mock_r2_instance

                    await _process_and_respond(
                        websocket=ws,
                        pipeline=pipeline,
                        app=app,
                        agent_engine=None,
                        text="Tell me about your products",
                        session_id="test-session-timeout",
                        employee_id="emp-001",
                        customer_id="cust-001",
                        avatar=avatar,
                        avatar_mode="video",
                        turn_start=time.perf_counter(),
                    )

        sent = [json.loads(c[0][0]) for c in ws.send_text.call_args_list]
        msg_types = [m["type"] for m in sent]

        # Should gracefully fallback — no video_url, but fallback_vrm present
        assert "fallback_vrm" in msg_types, f"Expected fallback_vrm on timeout, got: {msg_types}"
        assert "video_url" not in msg_types

        # Text and audio should still be sent
        assert "text_response" in msg_types
        assert "audio_response" in msg_types


# =============================================================================
# Test 04: Tool Calling
# =============================================================================


class TestPhase2_04_ToolCalling:
    """Test text_message → LLM returns tool_call → tool execution → response."""

    @pytest.mark.asyncio
    async def test_phase2_04_tool_calling_via_agent_engine(self):
        from src.agent.engine import AgentEngine
        from src.agent.tool_executor import ToolResult

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        employee = _make_mock_employee()

        # Mock _load_employee
        engine._load_employee = AsyncMock(return_value=employee)

        # Mock _load_knowledge (no KB entries)
        engine._load_knowledge = AsyncMock(return_value=[])

        # Mock _load_visitor_memories (no memories)
        engine._load_visitor_memories = AsyncMock(return_value=[])

        # Mock _load_tools — return a search_knowledge tool
        from src.agent.builtin_tools import BUILTIN_TOOLS, get_openai_tool_definitions
        openai_tools = get_openai_tool_definitions(BUILTIN_TOOLS)
        engine._load_tools = AsyncMock(return_value=(openai_tools, {}))

        # Mock _load_visitor_profile
        engine._load_visitor_profile = AsyncMock(return_value=None)

        # Mock the LLM: first call returns a tool_call, second returns text
        tool_call_response = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call-001",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge",
                            "arguments": json.dumps({"query": "store hours"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 50},
        }

        text_response = {
            "choices": [{
                "message": {
                    "content": "Our store is open from 9 AM to 9 PM daily.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 30},
        }

        call_count = [0]

        async def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count[0] == 0:
                resp.json = MagicMock(return_value=tool_call_response)
            else:
                resp.json = MagicMock(return_value=text_response)
            call_count[0] += 1
            return resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.is_closed = False
        engine._client = mock_client

        # Mock the tool executor
        with patch.object(engine, "execute_tool", new=AsyncMock(
            return_value=ToolResult(
                tool_id="search_knowledge",
                success=True,
                data={"results": [{"question": "Store hours?", "answer": "9 AM to 9 PM"}]},
            )
        )):
            result = await engine.handle_message(
                session_id="session-tool-test",
                visitor_message="What are your store hours?",
                employee_id="emp-001",
                visitor_id="visitor-001",
                customer_id="cust-001",
            )

        assert result == "Our store is open from 9 AM to 9 PM daily."
        assert call_count[0] == 2  # Two LLM calls: tool_call + final text


# =============================================================================
# Test 05: Workflow Execution
# =============================================================================


class TestPhase2_05_WorkflowExecution:
    """Test WorkflowEngine.execute_workflow() with lead_capture template."""

    @pytest.mark.asyncio
    async def test_phase2_05_workflow_lead_capture(self):
        from src.pipeline.workflow_engine import WorkflowEngine, WORKFLOW_TEMPLATES
        from src.db.models import Workflow, WorkflowExecution, _uuid

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = WorkflowEngine(db=db, config=config)

        # Create a mock Workflow using the lead_capture template
        template = WORKFLOW_TEMPLATES["lead_capture"]
        workflow_id = "wf-lead-001"

        mock_workflow = MagicMock(spec=Workflow)
        mock_workflow.id = workflow_id
        mock_workflow.name = template["name"]
        mock_workflow.steps = json.dumps(template["steps"])
        mock_workflow.customer_id = "cust-001"

        # Mock DB to return the workflow
        async def mock_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none = MagicMock(return_value=mock_workflow)
            return result

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        # Execute the workflow — it should pause at the first ask_visitor step
        execution = await engine.execute_workflow(
            workflow_id=workflow_id,
            session_id="session-wf-001",
            visitor_id="visitor-001",
        )

        # The first step is ask_visitor ("What is your name?"), so it should pause
        assert execution.status == "waiting"
        assert execution.current_step == 0

        await engine.close()

    def test_phase2_05_workflow_template_structure(self):
        """Verify lead_capture template has expected step types."""
        from src.pipeline.workflow_engine import WORKFLOW_TEMPLATES

        template = WORKFLOW_TEMPLATES["lead_capture"]
        steps = template["steps"]

        assert len(steps) == 4
        assert steps[0]["type"] == "ask_visitor"
        assert steps[1]["type"] == "ask_visitor"
        assert steps[2]["type"] == "call_tool"
        assert steps[3]["type"] == "send_notification"

    def test_phase2_05_substitute_vars(self):
        """Test variable substitution in workflow engine."""
        from src.pipeline.workflow_engine import WorkflowEngine

        result = WorkflowEngine._substitute_vars(
            "Hello {{name}}, your email is {{email}}",
            {"name": "Alice", "email": "alice@example.com"},
        )
        assert result == "Hello Alice, your email is alice@example.com"


# =============================================================================
# Test 06: Returning Visitor
# =============================================================================


class TestPhase2_06_ReturningVisitor:
    """Test that VisitorMemory entries are loaded for known visitors."""

    @pytest.mark.asyncio
    async def test_phase2_06_returning_visitor_memories_loaded(self):
        from src.agent.engine import AgentEngine
        from src.db.models import VisitorMemory

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        employee = _make_mock_employee()

        # Create mock memory entries
        mem1 = MagicMock(spec=VisitorMemory)
        mem1.memory_type = "identity"
        mem1.content = "Visitor's name: Ahmed"
        mem1.importance = 0.9
        mem1.expires_at = None

        mem2 = MagicMock(spec=VisitorMemory)
        mem2.memory_type = "preference"
        mem2.content = "I prefer Arabic language support"
        mem2.importance = 0.6
        mem2.expires_at = None

        # Mock DB to return memory entries
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[mem1, mem2])))

        mock_session.execute = AsyncMock(return_value=mock_result)

        # Call the internal method directly
        memories = await engine._load_visitor_memories(
            mock_session, visitor_id="visitor-returning-001", employee_id="emp-001"
        )

        assert len(memories) == 2
        assert memories[0]["type"] == "identity"
        assert "Ahmed" in memories[0]["content"]
        assert memories[1]["type"] == "preference"

    @pytest.mark.asyncio
    async def test_phase2_06_memory_injected_into_system_prompt(self):
        from src.agent.engine import AgentEngine, AgentContext

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        employee = _make_mock_employee()
        memories = [
            {"type": "identity", "content": "Visitor's name: Ahmed", "importance": 0.9},
            {"type": "preference", "content": "Prefers Arabic support", "importance": 0.6},
        ]

        # Mock all loading methods
        engine._load_employee = AsyncMock(return_value=employee)
        engine._load_knowledge = AsyncMock(return_value=[])
        engine._load_visitor_memories = AsyncMock(return_value=memories)
        engine._load_tools = AsyncMock(return_value=([], {}))
        engine._load_visitor_profile = AsyncMock(return_value=None)

        from src.agent.engine import _SessionState
        session = _SessionState(
            employee_id="emp-001", visitor_id="visitor-returning-001",
            customer_id="cust-001", last_access=time.time(),
        )

        context = await engine.build_context(
            mock_session, employee, session, "Hello again"
        )

        # Verify memories are in the system prompt
        assert "Visitor Memory" in context.system_prompt
        assert "Ahmed" in context.system_prompt
        assert "Arabic support" in context.system_prompt


# =============================================================================
# Test 07: Auto-Learning
# =============================================================================


class TestPhase2_07_AutoLearning:
    """Test AutoLearningEngine.process_session() extracting QA pairs."""

    @pytest.mark.asyncio
    async def test_phase2_07_auto_learning_high_confidence(self):
        from src.pipeline.auto_learning import AutoLearningEngine

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AutoLearningEngine(db=db, config=config)

        # Mock transcript loading
        transcript = (
            "user: What are your return policies?\n"
            "assistant: You can return items within 30 days of purchase with a receipt."
        )
        engine._load_transcript = AsyncMock(return_value=transcript)

        # Mock LLM extraction — high confidence pair
        qa_pairs = [
            {
                "question": "What is the return policy?",
                "answer": "Items can be returned within 30 days with a receipt.",
                "confidence": 0.92,
                "category": "returns",
            },
        ]
        engine._extract_qa_pairs = AsyncMock(return_value=qa_pairs)

        await engine.process_session(
            session_id="session-learn-001",
            employee_id="emp-001",
            customer_id="cust-001",
        )

        # Verify DB session.add was called (for high confidence auto-approved entry)
        assert mock_session.add.called
        added_obj = mock_session.add.call_args[0][0]

        from src.db.models import EmployeeKnowledge
        assert isinstance(added_obj, EmployeeKnowledge)
        assert added_obj.approved is True
        assert added_obj.category == "returns"
        assert "return" in added_obj.question.lower()

        mock_session.commit.assert_awaited()

        await engine.close()

    @pytest.mark.asyncio
    async def test_phase2_07_auto_learning_medium_confidence(self):
        from src.pipeline.auto_learning import AutoLearningEngine

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AutoLearningEngine(db=db, config=config)

        engine._load_transcript = AsyncMock(return_value="user: Q\nassistant: A")

        # Medium confidence — should go to EmployeeLearning (pending review)
        qa_pairs = [
            {
                "question": "Do you offer warranties?",
                "answer": "Yes, all products come with a 1-year warranty.",
                "confidence": 0.65,
                "category": "product",
            },
        ]
        engine._extract_qa_pairs = AsyncMock(return_value=qa_pairs)

        await engine.process_session(
            session_id="session-learn-002",
            employee_id="emp-001",
            customer_id="cust-001",
        )

        assert mock_session.add.called
        added_obj = mock_session.add.call_args[0][0]

        from src.db.models import EmployeeLearning
        assert isinstance(added_obj, EmployeeLearning)
        assert added_obj.status == "pending"
        assert added_obj.learning_type == "qa_pair"
        assert added_obj.source == "auto"

        await engine.close()

    @pytest.mark.asyncio
    async def test_phase2_07_auto_learning_low_confidence_discarded(self):
        from src.pipeline.auto_learning import AutoLearningEngine

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AutoLearningEngine(db=db, config=config)

        engine._load_transcript = AsyncMock(return_value="user: Hi\nassistant: Hello!")

        # Low confidence — should be discarded (no DB add)
        qa_pairs = [
            {
                "question": "Hi",
                "answer": "Hello!",
                "confidence": 0.2,
                "category": "general",
            },
        ]
        engine._extract_qa_pairs = AsyncMock(return_value=qa_pairs)

        await engine.process_session(
            session_id="session-learn-003",
            employee_id="emp-001",
            customer_id="cust-001",
        )

        # add should not be called for low confidence
        assert not mock_session.add.called

        await engine.close()


# =============================================================================
# Test 08: Billing Near-Zero
# =============================================================================


class TestPhase2_08_BillingNearZero:
    """Test billing deduction when balance is almost depleted."""

    @pytest.mark.asyncio
    async def test_phase2_08_billing_near_zero_triggers_alert(self):
        from src.pipeline.billing import BillingEngine

        config = _make_mock_config()
        config.billing_enabled = True
        config.billing_rate_per_second = 0.001
        config.billing_grace_period_s = 0

        db, mock_session = _make_mock_db()

        billing = BillingEngine(config=config, db=db)
        await billing.load()

        # Mock get_balance to return near-zero balance (3% remaining)
        billing.get_balance = AsyncMock(return_value={
            "plan_seconds_remaining": 1500,
            "plan_seconds_total": 50000,
            "extra_seconds_remaining": 0,
            "total_remaining": 1500,
            "plan_renewal_date": "2026-04-01T00:00:00",
            "usage_pct": 97.0,
        })

        alert = await billing.check_balance_and_alert("cust-near-zero")

        assert alert is not None
        assert alert["level"] == "urgent"
        assert alert["customer_id"] == "cust-near-zero"
        assert alert["pct_remaining"] == 3.0

    @pytest.mark.asyncio
    async def test_phase2_08_billing_exhausted_critical(self):
        from src.pipeline.billing import BillingEngine

        config = _make_mock_config()
        config.billing_enabled = True
        config.billing_rate_per_second = 0.001
        config.billing_grace_period_s = 0

        billing = BillingEngine(config=config, db=None)
        await billing.load()

        # Mock get_balance to return fully exhausted balance
        billing.get_balance = AsyncMock(return_value={
            "plan_seconds_remaining": 0,
            "plan_seconds_total": 50000,
            "extra_seconds_remaining": 0,
            "total_remaining": 0,
            "plan_renewal_date": "2026-04-01T00:00:00",
            "usage_pct": 100.0,
        })

        alert = await billing.check_balance_and_alert("cust-exhausted")

        assert alert is not None
        assert alert["level"] == "critical"
        assert "exhausted" in alert["alert_message"].lower()


# =============================================================================
# Test 09: Cost Tracking
# =============================================================================


class TestPhase2_09_CostTracking:
    """Test that _record_cost() writes APICostRecord to DB."""

    @pytest.mark.asyncio
    async def test_phase2_09_record_cost_writes_to_db(self):
        from src.api.ws_visitor import _record_cost
        from src.db.models import APICostRecord

        db, mock_session = _make_mock_db()

        app = MagicMock()
        app.state.db = db

        await _record_cost(
            app=app,
            service="llm",
            customer_id="cust-001",
            session_id="session-cost-001",
            cost_usd=0.0012,
            tokens_used=150,
            duration_ms=350,
            details={"text_length": 42, "response_length": 180},
        )

        # Verify add was called with an APICostRecord
        assert mock_session.add.called
        record = mock_session.add.call_args[0][0]
        assert isinstance(record, APICostRecord)
        assert record.service == "llm"
        assert record.customer_id == "cust-001"
        assert record.session_id == "session-cost-001"
        assert record.cost_usd == 0.0012
        assert record.tokens_used == 150
        assert record.duration_ms == 350

        details = json.loads(record.details)
        assert details["text_length"] == 42
        assert details["response_length"] == 180

        mock_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_phase2_09_record_cost_skips_zero_cost(self):
        from src.api.ws_visitor import _record_cost

        db, mock_session = _make_mock_db()
        app = MagicMock()
        app.state.db = db

        await _record_cost(
            app=app,
            service="tts",
            customer_id="cust-001",
            session_id="session-free",
            cost_usd=0.0,  # Zero cost — should be skipped
        )

        # add should NOT be called for zero cost
        assert not mock_session.add.called

    @pytest.mark.asyncio
    async def test_phase2_09_record_cost_skips_no_db(self):
        from src.api.ws_visitor import _record_cost

        app = MagicMock()
        app.state.db = None  # No DB

        # Should not raise
        await _record_cost(
            app=app,
            service="asr",
            customer_id="cust-001",
            session_id="session-no-db",
            cost_usd=0.001,
        )


# =============================================================================
# Test 10: Widget action_required
# =============================================================================


class TestPhase2_10_ActionRequired:
    """Test that requires_confirmation tool triggers action_required WS message."""

    @pytest.mark.asyncio
    async def test_phase2_10_confirmation_callback_sends_action_required(self):
        from src.api.ws_visitor import _make_confirmation_callback

        ws = AsyncMock()
        ws.send_text = AsyncMock()

        # Simulate visitor approving the action
        ws.receive_text = AsyncMock(return_value=json.dumps({
            "type": "action_response",
            "approved": True,
        }))

        callback = _make_confirmation_callback(ws, session_id="session-confirm")

        # Call the callback as the agent would
        approved = await callback(
            tool_id="send_email",
            tool_name="Send Email",
            description="Send order confirmation email",
            parameters={"to": "customer@example.com", "subject": "Order #123"},
        )

        assert approved is True

        # Verify action_required was sent
        assert ws.send_text.called
        sent_msg = json.loads(ws.send_text.call_args_list[0][0][0])
        assert sent_msg["type"] == "action_required"
        assert sent_msg["tool_id"] == "send_email"
        assert sent_msg["tool_name"] == "Send Email"
        assert sent_msg["description"] == "Send order confirmation email"
        assert sent_msg["parameters"]["to"] == "customer@example.com"
        assert sent_msg["session_id"] == "session-confirm"

    @pytest.mark.asyncio
    async def test_phase2_10_confirmation_callback_declined(self):
        from src.api.ws_visitor import _make_confirmation_callback

        ws = AsyncMock()
        ws.send_text = AsyncMock()

        # Visitor declines
        ws.receive_text = AsyncMock(return_value=json.dumps({
            "type": "action_response",
            "approved": False,
        }))

        callback = _make_confirmation_callback(ws, session_id="session-decline")

        approved = await callback(
            tool_id="create_ticket",
            tool_name="Create Support Ticket",
            description="Create a ticket for complaint",
            parameters={"priority": "high"},
        )

        assert approved is False

    @pytest.mark.asyncio
    async def test_phase2_10_confirmation_callback_timeout(self):
        from src.api.ws_visitor import _make_confirmation_callback

        ws = AsyncMock()
        ws.send_text = AsyncMock()

        # Simulate timeout — receive_text never returns
        ws.receive_text = AsyncMock(side_effect=asyncio.TimeoutError())

        callback = _make_confirmation_callback(ws, session_id="session-timeout")

        approved = await callback(
            tool_id="transfer_to_human",
            tool_name="Transfer to Human",
            description="Transfer this conversation",
            parameters={},
        )

        # Timeout should default to False (not approved)
        assert approved is False

    @pytest.mark.asyncio
    async def test_phase2_10_tool_execution_with_confirmation(self):
        """Test that AgentEngine.execute_tool checks requires_confirmation."""
        from src.agent.engine import AgentEngine, AgentContext, _SessionState
        from src.agent.tool_executor import ToolResult
        from src.db.models import ToolRegistry

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        # Create a tool that requires confirmation
        tool_reg = MagicMock(spec=ToolRegistry)
        tool_reg.id = "tool-reg-001"
        tool_reg.tool_id = "send_email"
        tool_reg.name = "Send Email"
        tool_reg.description = "Sends an email"
        tool_reg.requires_confirmation = True
        tool_reg.api_url = ""

        context = MagicMock(spec=AgentContext)
        context.tool_registry_map = {"send_email": tool_reg}
        context.tools = []

        session = _SessionState(
            employee_id="emp-001", visitor_id="visitor-001",
            customer_id="cust-001", last_access=time.time(),
        )

        # Confirmation callback declines
        confirmation_cb = AsyncMock(return_value=False)

        result = await engine.execute_tool(
            db_session=mock_session,
            tool_id="send_email",
            input_data={"to": "test@example.com", "subject": "Test"},
            session=session,
            context=context,
            confirmation_callback=confirmation_cb,
        )

        assert result.success is False
        assert "declined" in result.error.lower()
        confirmation_cb.assert_awaited_once()


# =============================================================================
# Test 11: KB Pre-Match
# =============================================================================


class TestPhase2_11_KBPreMatch:
    """Test that high-confidence FAQ match includes KB context in response."""

    @pytest.mark.asyncio
    async def test_phase2_11_kb_entries_injected_into_prompt(self):
        from src.agent.engine import AgentEngine, _SessionState

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        employee = _make_mock_employee()

        # Mock KB with a highly relevant entry
        kb_entries = [
            {
                "id": "kb-001",
                "category": "pricing",
                "question": "What is the pricing for the starter plan?",
                "answer": "The starter plan costs $100/month and includes 50,000 seconds.",
            },
            {
                "id": "kb-002",
                "category": "pricing",
                "question": "What is included in the professional plan?",
                "answer": "The professional plan costs $200/month with 100,000 seconds.",
            },
        ]

        engine._load_employee = AsyncMock(return_value=employee)
        engine._load_knowledge = AsyncMock(return_value=kb_entries)
        engine._load_visitor_memories = AsyncMock(return_value=[])
        engine._load_tools = AsyncMock(return_value=([], {}))
        engine._load_visitor_profile = AsyncMock(return_value=None)

        session = _SessionState(
            employee_id="emp-001", visitor_id="visitor-001",
            customer_id="cust-001", last_access=time.time(),
        )

        context = await engine.build_context(
            mock_session, employee, session, "What is the pricing?"
        )

        # Verify KB entries are injected into the system prompt
        assert "Knowledge Base" in context.system_prompt
        assert "starter plan costs $100" in context.system_prompt
        assert "professional plan costs $200" in context.system_prompt

        # Verify knowledge_entries are stored in context
        assert len(context.knowledge_entries) == 2
        assert context.knowledge_entries[0]["category"] == "pricing"

    @pytest.mark.asyncio
    async def test_phase2_11_no_kb_match_no_injection(self):
        """When no KB entries match, prompt should not have KB section."""
        from src.agent.engine import AgentEngine, _SessionState

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        employee = _make_mock_employee()

        engine._load_employee = AsyncMock(return_value=employee)
        engine._load_knowledge = AsyncMock(return_value=[])  # No matches
        engine._load_visitor_memories = AsyncMock(return_value=[])
        engine._load_tools = AsyncMock(return_value=([], {}))
        engine._load_visitor_profile = AsyncMock(return_value=None)

        session = _SessionState(
            employee_id="emp-001", visitor_id="visitor-001",
            customer_id="cust-001", last_access=time.time(),
        )

        context = await engine.build_context(
            mock_session, employee, session, "Random unrelated question"
        )

        # No KB section should be present
        assert "Knowledge Base" not in context.system_prompt
        assert len(context.knowledge_entries) == 0

    @pytest.mark.asyncio
    async def test_phase2_11_kb_keyword_matching(self):
        """Test the _load_knowledge keyword matching logic."""
        from src.agent.engine import AgentEngine
        from src.db.models import EmployeeKnowledge

        db, mock_session = _make_mock_db()
        config = _make_mock_config()

        engine = AgentEngine(db=db, config=config)

        # Create mock KB entries
        kb1 = MagicMock(spec=EmployeeKnowledge)
        kb1.id = "kb-001"
        kb1.category = "shipping"
        kb1.question = "How long does shipping take?"
        kb1.answer = "Standard shipping takes 3-5 business days."

        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[kb1])))
        mock_session.execute = AsyncMock(return_value=mock_result)

        entries = await engine._load_knowledge(
            mock_session, employee_id="emp-001", query="How long does shipping take?"
        )

        assert len(entries) == 1
        assert entries[0]["question"] == "How long does shipping take?"
        assert "3-5 business days" in entries[0]["answer"]
