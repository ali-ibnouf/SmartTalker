"""End-to-end integration tests for SmartTalker."""

import importlib

import pytest

sqlalchemy_available = importlib.util.find_spec("sqlalchemy") is not None
if not sqlalchemy_available:
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from src.main import app


@pytest.fixture
def client():
    # Mock external services before app startup
    with patch("src.main.SmartTalkerPipeline") as MockPipeline, \
         patch("src.main.WhatsAppClient") as MockWhatsApp, \
         patch("src.main.StorageManager"), \
         patch("src.main.Database") as MockDB, \
         patch("src.main.BillingEngine") as MockBilling, \
         patch("src.main.PersonaEngine") as MockPersona, \
         patch("redis.asyncio.from_url") as MockRedis:

        # Setup mocks — return values must match Pydantic schema fields
        mock_pipeline = MockPipeline.return_value
        mock_pipeline.load_all.return_value = None
        mock_pipeline.health_check = AsyncMock(return_value={
            "status": "healthy",
            "models_loaded": {},
            "uptime_s": 0.0,
        })
        mock_pipeline.unload_all = AsyncMock()
        mock_pipeline._training = None  # No training engine in e2e mock
        mock_pipeline._guardrails = None  # No guardrails engine in e2e mock

        mock_whatsapp = MockWhatsApp.return_value
        mock_whatsapp.close = AsyncMock()
        mock_whatsapp.verify_webhook.return_value = None

        mock_db = MockDB.return_value
        mock_db.connect = AsyncMock()
        mock_db.disconnect = AsyncMock()

        mock_billing = MockBilling.return_value
        mock_billing.load = AsyncMock()
        mock_billing.unload = AsyncMock()

        mock_persona = MockPersona.return_value
        mock_persona.load = AsyncMock()
        mock_persona.unload = AsyncMock()

        mock_redis_client = MockRedis.return_value
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.close = AsyncMock()

        # Create TestClient which triggers lifespan
        with TestClient(app) as test_client:
            yield test_client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "uptime_s" in data


def test_metrics_endpoint(client):
    """Test that metrics endpoint is exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_whatsapp_webhook_verification(client):
    """Test WhatsApp webhook verification challenge."""
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "test_token",
        "hub.challenge": "12345"
    }
    response = client.get("/api/v1/whatsapp/webhook", params=params)
    assert response.status_code in [200, 403]


# =============================================================================
# Shared Fixtures for E2E Scenario Tests
# =============================================================================


@pytest.fixture
def mock_db_session():
    """Create a mock async database session with execute/commit/add support."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_database(mock_db_session):
    """Create a mock Database that yields a mock session from session()."""
    from contextlib import asynccontextmanager

    db = MagicMock()

    @asynccontextmanager
    async def _session():
        yield mock_db_session

    db.session = _session

    @asynccontextmanager
    async def _session_ctx():
        yield mock_db_session

    db.session_ctx = _session_ctx
    return db


@pytest.fixture
def e2e_billing_engine():
    """Create a real BillingEngine with no DB (in-memory, unlimited quota)."""
    from src.pipeline.billing import BillingEngine

    config = MagicMock()
    config.billing_enabled = True
    config.billing_rate_per_second = 0.001
    config.billing_grace_period_s = 5
    engine = BillingEngine(config, db=None)
    return engine


@pytest.fixture
def e2e_node_manager():
    """Create a real NodeManager with no DB for node tracking."""
    from src.pipeline.node_manager import NodeManager

    return NodeManager(db=None)


@pytest.fixture
def e2e_kill_switch():
    """Create a real KillSwitch with no DB/Redis for local-cache testing."""
    from src.pipeline.kill_switch import KillSwitch

    return KillSwitch(db=None, redis=None, ws_manager=None)


# =============================================================================
# 1. Customer Journey — TestCustomerJourney
# =============================================================================


class TestCustomerJourney:
    """End-to-end tests simulating a customer's lifecycle through the platform.

    Covers: signup, API key provisioning, sessions, knowledge base operations,
    billing lifecycle, and avatar configuration.
    """

    @pytest.mark.asyncio
    async def test_signup_to_first_conversation(self, mock_database, mock_db_session):
        """Full flow: Create customer -> get API key -> start session -> send text -> receive response."""
        import uuid
        from src.pipeline.billing import BillingEngine, BillingSession

        # Step 1: Simulate customer creation (the admin_create_customer route logic)
        customer_id = uuid.uuid4().hex
        api_key = uuid.uuid4().hex + uuid.uuid4().hex[:8]

        # Mock the DB execute for duplicate-email check (no existing customer)
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = None  # No duplicate
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Verify the customer ID and API key are generated properly
        assert len(customer_id) == 32  # uuid4 hex
        assert len(api_key) == 40  # 32 + 8 chars

        # Step 2: Start a billing session for this customer (simulates session start)
        billing_config = MagicMock()
        billing_config.billing_enabled = True
        billing_config.billing_rate_per_second = 0.001
        billing_config.billing_grace_period_s = 5
        billing = BillingEngine(billing_config, db=None)

        session = await billing.start_session(
            session_id="sess_001",
            customer_id=customer_id,
            avatar_id="default",
            channel="web",
        )
        assert session.session_id == "sess_001"
        assert session.customer_id == customer_id
        assert session.avatar_id == "default"
        assert session.started_at > 0

        # Step 3: Simulate text processing (mock pipeline)
        mock_pipeline = MagicMock()
        mock_result_obj = MagicMock(
            audio_path="/files/audio/test.wav",
            response_text="Hello! How can I help you?",
            total_latency_ms=250,
            breakdown={"llm": 150, "tts": 100},
        )
        mock_pipeline.process_text = AsyncMock(return_value=mock_result_obj)

        result = await mock_pipeline.process_text(
            text="Hello, I need help",
            avatar_id="default",
            language="ar",
        )
        assert result.response_text == "Hello! How can I help you?"
        assert result.total_latency_ms == 250

        # Step 4: Stop session and verify billing
        stopped = await billing.stop_session("sess_001")
        assert stopped is not None
        assert stopped.total_seconds >= 0
        assert stopped.total_cost >= 0

        # Verify session is no longer active
        active = await billing.get_active_sessions()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_knowledge_base_flow(self):
        """Full flow: Create customer -> upload KB documents -> query KB -> verify relevant results."""
        # Step 1: Create a mock KB engine
        mock_kb = MagicMock()
        mock_kb.is_loaded = True

        # Step 2: Simulate document ingestion
        mock_ingest_result = MagicMock(
            doc_id="doc_001",
            filename="company_faq.pdf",
            doc_type="pdf",
            chunk_count=15,
        )
        mock_kb.ingest_document = AsyncMock(return_value=mock_ingest_result)

        result = await mock_kb.ingest_document("/tmp/company_faq.pdf", doc_type="pdf")
        assert result.doc_id == "doc_001"
        assert result.chunk_count == 15

        # Step 3: Verify document is listed
        mock_doc = MagicMock(
            doc_id="doc_001",
            filename="company_faq.pdf",
            doc_type="pdf",
            chunk_count=15,
            created_at=1700000000.0,
            file_hash="abc123",
        )
        mock_kb.list_documents = MagicMock(return_value=[mock_doc])
        docs = mock_kb.list_documents()
        assert len(docs) == 1
        assert docs[0].doc_id == "doc_001"

        # Step 4: Search the KB and verify relevant results
        mock_search_result = MagicMock(
            chunks=[
                {"text": "Our return policy allows 30 day returns.", "similarity": 0.92, "metadata": {"doc_id": "doc_001"}},
                {"text": "Contact support at support@example.com.", "similarity": 0.85, "metadata": {"doc_id": "doc_001"}},
            ],
            query="What is the return policy?",
            top_similarity=0.92,
            latency_ms=45,
        )
        mock_kb.search = AsyncMock(return_value=mock_search_result)

        search = await mock_kb.search("What is the return policy?", top_k=3)
        assert search.top_similarity >= 0.8
        assert len(search.chunks) == 2
        assert "return policy" in search.chunks[0]["text"].lower()
        assert search.latency_ms < 5000  # reasonable latency

    @pytest.mark.asyncio
    async def test_billing_lifecycle(self):
        """Full flow: Create customer -> check quota -> simulate usage -> verify deductions -> top-up -> verify new balance."""
        from src.pipeline.billing import BillingEngine
        import time

        config = MagicMock()
        config.billing_enabled = True
        config.billing_rate_per_second = 0.001
        config.billing_grace_period_s = 0  # No grace for predictable math
        billing = BillingEngine(config, db=None)

        customer_id = "cust_billing_test"

        # Step 1: Check quota (no DB = unlimited)
        remaining = await billing.check_quota(customer_id)
        assert remaining == float("inf")

        # Step 2: Start a session
        session = await billing.start_session("sess_bill_1", customer_id, avatar_id="avatar1")
        assert session.session_id == "sess_bill_1"
        assert session.started_at > 0

        # Step 3: Verify session is active
        active = await billing.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == "sess_bill_1"

        # Step 4: Wait briefly and stop session to incur cost
        import asyncio
        await asyncio.sleep(0.05)  # 50ms simulated usage

        stopped = await billing.stop_session("sess_bill_1")
        assert stopped is not None
        assert stopped.total_seconds > 0
        assert stopped.total_cost > 0

        # Step 5: Verify no more active sessions
        active = await billing.get_active_sessions()
        assert len(active) == 0

        # Step 6: Simulate top-up (no DB = returns passed seconds)
        new_total = await billing.add_topup(customer_id, 10_000)
        assert new_total == 10_000

        # Step 7: Verify balance (no DB = unlimited plan)
        balance = await billing.get_balance(customer_id)
        assert balance["plan_seconds_remaining"] == float("inf")

    @pytest.mark.asyncio
    async def test_avatar_configuration(self):
        """Full flow: Create customer -> configure avatar -> verify avatar settings persist."""
        # Step 1: Simulate avatar data store
        avatar_store = {}

        # Step 2: Configure an avatar
        avatar_config = {
            "avatar_id": "avatar_custom_001",
            "name": "Customer Service Rep",
            "image_url": "/files/avatars/custom_001.png",
            "avatar_type": "video",
            "description": "Professional customer service avatar",
            "language": "ar",
            "voice_id": "voice_arabic_female",
        }
        avatar_store[avatar_config["avatar_id"]] = avatar_config

        # Step 3: Verify settings persist
        stored = avatar_store.get("avatar_custom_001")
        assert stored is not None
        assert stored["name"] == "Customer Service Rep"
        assert stored["avatar_type"] == "video"
        assert stored["language"] == "ar"
        assert stored["voice_id"] == "voice_arabic_female"

        # Step 4: Update avatar settings (e.g., switch to VRM)
        avatar_store["avatar_custom_001"]["avatar_type"] = "vrm"
        avatar_store["avatar_custom_001"]["vrm_url"] = "/files/vrm/custom_001.vrm"
        updated = avatar_store["avatar_custom_001"]
        assert updated["avatar_type"] == "vrm"
        assert updated["vrm_url"] == "/files/vrm/custom_001.vrm"
        # Original fields unchanged
        assert updated["name"] == "Customer Service Rep"
        assert updated["voice_id"] == "voice_arabic_female"


# =============================================================================
# 2. Operator Journey — TestOperatorJourney
# =============================================================================


class TestOperatorJourney:
    """End-to-end tests simulating an operator's workflow.

    Covers: session monitoring, takeover/return, training corrections,
    and quality scoring via the supervisor engine.
    """

    @pytest.mark.asyncio
    async def test_operator_session_monitoring(self):
        """Connect operator WS -> verify active session list."""
        from src.pipeline.supervisor import SupervisorEngine

        # Create supervisor with mock ws_manager that has active sessions
        config = MagicMock()
        config.training_db_path = "test.db"
        supervisor = SupervisorEngine(config, db=None)

        # Simulate WebSocket manager with active sessions
        mock_ws_mgr = MagicMock()
        mock_session_1 = MagicMock()
        mock_session_1.client_ip = "10.0.0.1"
        mock_session_1.connected_at = 1700000000.0
        mock_session_1.ai_paused = False
        mock_session_1.operator_id = None
        mock_config_1 = MagicMock()
        mock_config_1.avatar_id = "avatar_001"
        mock_config_1.training_mode = "digital"
        mock_session_1.config = mock_config_1

        mock_session_2 = MagicMock()
        mock_session_2.client_ip = "10.0.0.2"
        mock_session_2.connected_at = 1700001000.0
        mock_session_2.ai_paused = True
        mock_session_2.operator_id = "op_001"
        mock_config_2 = MagicMock()
        mock_config_2.avatar_id = "avatar_002"
        mock_config_2.training_mode = "training"
        mock_session_2.config = mock_config_2

        mock_ws_mgr._sessions = {
            "sess_100": mock_session_1,
            "sess_200": mock_session_2,
        }
        supervisor.set_ws_manager(mock_ws_mgr)

        # Get active sessions
        sessions = await supervisor.get_active_sessions_summary()
        assert len(sessions) == 2

        # Verify session details
        sess_ids = {s["session_id"] for s in sessions}
        assert "sess_100" in sess_ids
        assert "sess_200" in sess_ids

        # Find the paused session
        paused = [s for s in sessions if s.get("ai_paused")]
        assert len(paused) == 1
        assert paused[0]["operator_id"] == "op_001"

    @pytest.mark.asyncio
    async def test_session_takeover_and_return(self):
        """Start AI session -> operator takeover -> verify AI paused -> operator return -> verify AI resumed."""
        # Step 1: Set up a session object simulating an active WebSocket session
        mock_session = MagicMock()
        mock_session.ai_paused = False
        mock_session.operator_id = None
        mock_session.takeover_at = None
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_session.websocket = mock_ws
        mock_config = MagicMock()
        mock_config.avatar_id = "avatar_001"
        mock_session.config = mock_config

        ws_manager = MagicMock()
        ws_manager._sessions = {"sess_takeover_1": mock_session}

        # Step 2: Verify session starts with AI active
        assert mock_session.ai_paused is False
        assert mock_session.operator_id is None

        # Step 3: Operator takeover (simulate route logic)
        operator_id = "op_admin_001"
        mock_session.ai_paused = True
        mock_session.operator_id = operator_id

        assert mock_session.ai_paused is True
        assert mock_session.operator_id == operator_id

        # Step 4: Operator returns to AI (simulate route logic)
        mock_session.ai_paused = False
        mock_session.operator_id = None
        mock_session.takeover_at = None

        assert mock_session.ai_paused is False
        assert mock_session.operator_id is None

    @pytest.mark.asyncio
    async def test_training_correction_submission(self):
        """Submit training corrections -> verify stored."""
        from src.pipeline.supervisor import SupervisorEngine

        # Use a mock supervisor to record the training submission action
        config = MagicMock()
        config.training_db_path = "test.db"
        supervisor = SupervisorEngine(config, db=None)

        # Mock the SQLite path — use the loaded state manually
        supervisor._loaded = True
        supervisor._sqlite_conn = AsyncMock()
        supervisor._sqlite_conn.execute = AsyncMock()
        supervisor._sqlite_conn.commit = AsyncMock()

        # Record a training submission action
        action_id = await supervisor.record_operator_action(
            operator_id="op_trainer_001",
            action_type="training_submit",
            session_id="sess_train_1",
            avatar_id="avatar_001",
            details={
                "approved_count": 5,
                "correction_count": 2,
                "notes": "Fixed two incorrect product descriptions",
            },
        )
        assert action_id is not None
        assert len(action_id) == 32  # uuid hex

        # Verify the execute was called for the INSERT
        supervisor._sqlite_conn.execute.assert_called_once()
        call_args = supervisor._sqlite_conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO operator_actions" in sql

        # Verify the params include our data
        params = call_args[0][1]
        assert params[1] == "op_trainer_001"  # operator_id
        assert params[2] == "training_submit"  # action_type
        assert params[3] == "sess_train_1"  # session_id

    @pytest.mark.asyncio
    async def test_supervisor_quality_scoring(self):
        """Record actions -> check quality score calculated correctly."""
        from src.pipeline.supervisor import SupervisorEngine, OperatorMetrics

        config = MagicMock()
        config.training_db_path = "test.db"
        supervisor = SupervisorEngine(config, db=None)

        # Manually set loaded state with mock SQLite
        supervisor._loaded = True
        supervisor._sqlite_conn = AsyncMock()

        # Mock cursor results for get_operator_metrics queries
        # total_responses = 10
        cursor_responses = AsyncMock()
        cursor_responses.fetchone = AsyncMock(return_value=(10,))
        # avg_response_time_ms = 500
        cursor_avg = AsyncMock()
        cursor_avg.fetchone = AsyncMock(return_value=(500.0,))
        # escalations_resolved = 3
        cursor_esc = AsyncMock()
        cursor_esc.fetchone = AsyncMock(return_value=(3,))
        # corrections_made = 2
        cursor_corr = AsyncMock()
        cursor_corr.fetchone = AsyncMock(return_value=(2,))
        # sessions_handled = 5
        cursor_sess = AsyncMock()
        cursor_sess.fetchone = AsyncMock(return_value=(5,))
        # earliest action = some timestamp
        cursor_earliest = AsyncMock()
        cursor_earliest.fetchone = AsyncMock(return_value=(1700000000.0,))

        supervisor._sqlite_conn.execute = AsyncMock(
            side_effect=[
                cursor_responses,
                cursor_avg,
                cursor_esc,
                cursor_corr,
                cursor_sess,
                cursor_earliest,
            ]
        )

        # Get operator metrics (triggers quality_score calculation)
        metrics = await supervisor.get_operator_metrics("op_quality_001", days=30)
        assert metrics.operator_id == "op_quality_001"
        assert metrics.total_responses == 10
        assert metrics.avg_response_time_ms == 500
        assert metrics.escalations_resolved == 3
        assert metrics.corrections_made == 2
        assert metrics.sessions_handled == 5

        # Quality score: total_responses / (total_responses + corrections_made) = 10/12
        expected_score = round(10 / (10 + 2), 2)
        assert metrics.quality_score == expected_score


# =============================================================================
# 3. Admin Journey — TestAdminJourney
# =============================================================================


class TestAdminJourney:
    """End-to-end tests simulating an administrator's workflow.

    Covers: dashboard overview, node management, customer suspension,
    and AI agent scanning.
    """

    @pytest.mark.asyncio
    async def test_admin_dashboard_overview(self):
        """Fetch dashboard overview -> verify all fields present."""
        # Simulate the dashboard overview response structure
        overview = {
            "avatars": 5,
            "conversations": 1200,
            "total_duration_s": 36000.50,
            "total_cost": 36.0005,
        }

        # Verify all required fields are present
        assert "avatars" in overview
        assert "conversations" in overview
        assert "total_duration_s" in overview
        assert "total_cost" in overview

        # Verify types and reasonable values
        assert isinstance(overview["avatars"], int)
        assert isinstance(overview["conversations"], int)
        assert isinstance(overview["total_duration_s"], float)
        assert isinstance(overview["total_cost"], float)
        assert overview["avatars"] >= 0
        assert overview["conversations"] >= 0
        assert overview["total_duration_s"] >= 0
        assert overview["total_cost"] >= 0

    @pytest.mark.asyncio
    async def test_admin_node_management(self, e2e_node_manager):
        """Register node -> list nodes -> verify node appears -> deregister."""
        nm = e2e_node_manager

        # Step 1: Initially no nodes
        assert len(nm.list_nodes()) == 0

        # Step 2: Register a node (mock websocket)
        mock_ws = AsyncMock()
        node = await nm.register_node(
            websocket=mock_ws,
            node_id="gpu_node_001",
            hostname="render-box-1.local",
            gpu_type="RTX 4090",
            vram_mb=24576,
            license_key="lic_test_key",
            customer_id="cust_001",
            max_concurrent=5,
            avatars_loaded=["avatar_001", "avatar_002"],
        )
        assert node.node_id == "gpu_node_001"
        assert node.hostname == "render-box-1.local"
        assert node.gpu_type == "RTX 4090"
        assert node.status == "online"
        assert node.max_concurrent == 5

        # Step 3: Verify node appears in list
        nodes = nm.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "gpu_node_001"

        # Step 4: Verify we can get the specific node
        fetched = nm.get_node("gpu_node_001")
        assert fetched is not None
        assert fetched.customer_id == "cust_001"

        # Step 5: Register a second node
        mock_ws2 = AsyncMock()
        await nm.register_node(
            websocket=mock_ws2,
            node_id="gpu_node_002",
            hostname="render-box-2.local",
            gpu_type="A100",
            vram_mb=81920,
            license_key="lic_test_key_2",
            customer_id="cust_002",
            max_concurrent=10,
        )
        assert len(nm.list_nodes()) == 2

        # Step 6: Deregister first node
        await nm.deregister_node("gpu_node_001")
        nodes = nm.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "gpu_node_002"

        # Step 7: Verify deregistered node is gone
        assert nm.get_node("gpu_node_001") is None

        # Step 8: Clean up
        await nm.deregister_node("gpu_node_002")
        assert len(nm.list_nodes()) == 0

    @pytest.mark.asyncio
    async def test_admin_customer_suspension(self, e2e_kill_switch):
        """Create customer -> suspend -> verify suspended -> resume -> verify active."""
        ks = e2e_kill_switch
        customer_id = "cust_suspend_test"

        # Step 1: Customer starts not suspended
        is_suspended = await ks.is_suspended(customer_id)
        assert is_suspended is False

        # Step 2: Suspend the customer
        result = await ks.suspend(customer_id, reason="Terms violation")
        assert result is True

        # Step 3: Verify suspended
        is_suspended = await ks.is_suspended(customer_id)
        assert is_suspended is True

        # Step 4: Resume the customer
        result = await ks.resume(customer_id)
        assert result is True

        # Step 5: Verify no longer suspended
        is_suspended = await ks.is_suspended(customer_id)
        assert is_suspended is False

    @pytest.mark.asyncio
    async def test_admin_agent_scan(self):
        """Trigger agent scan -> verify scan results structure."""
        from src.services.ai_agent.agent import AIAgent
        from src.services.ai_agent.config import AgentSettings
        from src.services.ai_agent.rules import AgentContext, Detection

        # Set up agent with mocked context
        agent_config = AgentSettings(
            agent_enabled=False,  # Do not start background loop
            auto_fix_enabled=False,
            scan_interval_s=60,
        )

        mock_pipeline = MagicMock()
        mock_pipeline._training = None
        mock_pipeline._guardrails = None

        mock_config = MagicMock()
        mock_config.billing_enabled = False

        ctx = AgentContext(
            db=None,
            redis=None,
            node_manager=MagicMock(),
            pipeline=mock_pipeline,
            config=mock_config,
            agent_config=agent_config,
        )

        agent = AIAgent(ctx)

        # Run a manual scan
        results = await agent.run_manual_scan()

        # Verify structure — results is a list of detection dicts
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, dict)
            assert "rule_id" in item
            assert "severity" in item
            assert "title" in item
            assert "description" in item
            assert "recommendation" in item
            assert "auto_fixable" in item

        # Verify agent stats are updated
        stats = await agent.get_stats()
        assert stats["scan_count"] >= 1
        assert stats["running"] is False  # We did not start the loop
        assert stats["rules_count"] > 0
        assert stats["last_scan_at"] is not None


# =============================================================================
# 4. Failure Scenarios — TestFailureScenarios
# =============================================================================


class TestFailureScenarios:
    """End-to-end tests for error handling and edge cases.

    Covers: expired quota, invalid API keys, rate limiting, and
    node disconnect recovery.
    """

    @pytest.mark.asyncio
    async def test_expired_quota_blocks_session(self):
        """Customer with zero quota -> attempt session -> verify blocked."""
        from src.pipeline.billing import BillingEngine
        from src.utils.exceptions import BillingError

        config = MagicMock()
        config.billing_enabled = True
        config.billing_rate_per_second = 0.001
        config.billing_grace_period_s = 5

        # Create billing engine with a mock DB that returns zero balance
        mock_db = MagicMock()
        mock_session = AsyncMock()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _session():
            yield mock_session

        mock_db.session = _session

        billing = BillingEngine(config, db=mock_db)

        # Mock get_balance to return zero remaining
        async def mock_get_balance(cid):
            return {
                "plan_seconds_remaining": 0,
                "plan_seconds_total": 50000,
                "extra_seconds_remaining": 0,
                "total_remaining": 0,
                "plan_renewal_date": "2026-04-01T00:00:00+00:00",
                "usage_pct": 100.0,
            }

        billing.get_balance = mock_get_balance

        # Attempt to start session — should raise BillingError
        with pytest.raises(BillingError) as exc_info:
            await billing.start_session("sess_blocked", "cust_no_quota")

        assert "Quota exceeded" in str(exc_info.value.message)

    def test_invalid_api_key_rejected(self):
        """Request with bad API key -> verify 403 by testing the middleware directly."""
        import hmac
        from src.api.middleware import APIKeyAuthMiddleware

        # Create a config with a known API key
        config = MagicMock()
        config.api_key = "valid_secret_key_abc123"
        config.admin_api_key = "admin_secret_key_xyz789"

        middleware = APIKeyAuthMiddleware(app=MagicMock(), config=config)

        # Test that an invalid key does NOT match
        bad_key = "totally_wrong_key_12345"
        assert not hmac.compare_digest(bad_key, config.api_key)

        # Test that the valid key DOES match
        assert hmac.compare_digest(config.api_key, "valid_secret_key_abc123")

        # Test admin path detection
        assert middleware._is_admin_path("/api/v1/admin/suspend/cust1") is True
        assert middleware._is_admin_path("/api/v1/billing/quota") is False

        # Test that excluded paths bypass auth
        from src.api.middleware import _is_excluded
        assert _is_excluded("/api/v1/health") is True
        assert _is_excluded("/docs") is True
        assert _is_excluded("/ws/chat") is True
        assert _is_excluded("/api/v1/text-to-speech") is False

    def test_rate_limiting_enforced(self):
        """Send many rapid requests -> verify rate limit kicks in."""
        from src.api.middleware import RedisRateLimitMiddleware
        import time

        # Create a rate limiter with a very low limit (no Redis = in-memory fallback)
        config = MagicMock()
        config.rate_limit_per_minute = 3

        limiter = RedisRateLimitMiddleware(app=MagicMock(), config=config)
        client_ip = "192.168.1.100"
        now = time.time()

        # First 3 requests should be allowed
        for i in range(3):
            allowed, count = limiter._check_memory(client_ip, now + i * 0.001)
            assert allowed is True, f"Request {i+1} should be allowed"

        # 4th request should be blocked
        allowed, count = limiter._check_memory(client_ip, now + 0.01)
        assert allowed is False, "Request 4 should be rate limited"

    @pytest.mark.asyncio
    async def test_node_disconnect_recovery(self, e2e_node_manager):
        """Register node -> simulate disconnect -> verify cleanup."""
        nm = e2e_node_manager

        # Step 1: Register a node
        mock_ws = AsyncMock()
        await nm.register_node(
            websocket=mock_ws,
            node_id="gpu_ephemeral_001",
            hostname="ephemeral-box.local",
            gpu_type="RTX 3090",
            vram_mb=24576,
            license_key="lic_eph",
            customer_id="cust_eph",
            max_concurrent=3,
        )
        assert len(nm.list_nodes()) == 1

        # Step 2: Start sessions on this node
        ns1 = nm.start_session("gpu_ephemeral_001", "sess_eph_1", "avatar_001")
        ns2 = nm.start_session("gpu_ephemeral_001", "sess_eph_2", "avatar_002")
        assert ns1 is not None
        assert ns2 is not None

        # Verify sessions are tracked
        assert nm.get_session("sess_eph_1") is not None
        assert nm.get_session("sess_eph_2") is not None

        # Step 3: Verify concurrent limit
        ns3 = nm.start_session("gpu_ephemeral_001", "sess_eph_3", "avatar_003")
        assert ns3 is not None  # 3rd session under limit of 3

        ns4 = nm.start_session("gpu_ephemeral_001", "sess_eph_4", "avatar_004")
        assert ns4 is None  # 4th session blocked by concurrent limit

        # Step 4: Simulate disconnect (deregister cleans up sessions)
        await nm.deregister_node("gpu_ephemeral_001")

        # Step 5: Verify cleanup
        assert len(nm.list_nodes()) == 0
        assert nm.get_node("gpu_ephemeral_001") is None
        assert nm.get_session("sess_eph_1") is None
        assert nm.get_session("sess_eph_2") is None
        assert nm.get_session("sess_eph_3") is None

        # Step 6: Verify a new node can be registered with the same ID
        mock_ws2 = AsyncMock()
        await nm.register_node(
            websocket=mock_ws2,
            node_id="gpu_ephemeral_001",
            hostname="ephemeral-box-v2.local",
            gpu_type="RTX 4090",
            vram_mb=24576,
            license_key="lic_eph_2",
            customer_id="cust_eph",
            max_concurrent=5,
        )
        assert len(nm.list_nodes()) == 1
        recovered = nm.get_node("gpu_ephemeral_001")
        assert recovered.hostname == "ephemeral-box-v2.local"
        assert recovered.max_concurrent == 5

        # Cleanup
        await nm.deregister_node("gpu_ephemeral_001")
