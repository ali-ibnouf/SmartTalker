"""Infrastructure monitoring rules (PostgreSQL, Redis, DashScope, RunPod, R2)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from src.services.ai_agent.rules import AgentContext, Detection
from src.utils.logger import setup_logger

logger = setup_logger("ai_agent.monitors.infrastructure")


class PostgreSQLConnectionsRule:
    """Detects high PostgreSQL connection count relative to max_connections."""

    rule_id = "infra.pg_connections"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        try:
            from sqlalchemy import text

            async with ctx.db.session_ctx() as session:
                # Get current connection count
                r = await session.execute(text("SELECT count(*) FROM pg_stat_activity"))
                current = r.scalar() or 0

                # Get max connections
                r2 = await session.execute(text("SHOW max_connections"))
                max_conn = int(r2.scalar() or 100)

            usage_pct = (current / max_conn) * 100 if max_conn > 0 else 0
            threshold = ctx.agent_config.pg_connections_warn_pct

            if usage_pct >= threshold:
                severity = "critical" if usage_pct >= 90 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"PostgreSQL connections: {current}/{max_conn} ({usage_pct:.0f}%)",
                    description=(
                        f"Database has {current} active connections out of {max_conn} max "
                        f"({usage_pct:.0f}%). Risk of connection exhaustion."
                    ),
                    details={
                        "current_connections": current,
                        "max_connections": max_conn,
                        "usage_pct": round(usage_pct, 1),
                    },
                    recommendation="Kill idle connections or increase max_connections in PostgreSQL config.",
                    auto_fixable=True,
                )]
        except Exception as exc:
            logger.debug(f"PostgreSQLConnectionsRule skipped: {exc}")

        return []


class RedisMemoryRule:
    """Detects high Redis memory usage."""

    rule_id = "infra.redis_memory"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.redis is None:
            return []

        try:
            info = await ctx.redis.info("memory")
            used = info.get("used_memory", 0)
            maxmem = info.get("maxmemory", 0)

            # If maxmemory not set, use 1GB as reference
            if maxmem == 0:
                maxmem = 1024 * 1024 * 1024  # 1GB

            usage_pct = (used / maxmem) * 100 if maxmem > 0 else 0
            threshold = ctx.agent_config.redis_memory_warn_pct

            if usage_pct >= threshold:
                severity = "critical" if usage_pct >= 90 else "warning"
                used_mb = used / (1024 * 1024)
                max_mb = maxmem / (1024 * 1024)
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"Redis memory: {used_mb:.0f}MB / {max_mb:.0f}MB ({usage_pct:.0f}%)",
                    description=(
                        f"Redis is using {used_mb:.0f}MB of {max_mb:.0f}MB "
                        f"({usage_pct:.0f}%). Risk of evictions or OOM."
                    ),
                    details={
                        "used_mb": round(used_mb, 1),
                        "max_mb": round(max_mb, 1),
                        "usage_pct": round(usage_pct, 1),
                    },
                    recommendation="Clear expired cache keys or increase Redis maxmemory.",
                    auto_fixable=True,
                )]
        except Exception as exc:
            logger.debug(f"RedisMemoryRule skipped: {exc}")

        return []


class DashScopeLatencyRule:
    """Detects high DashScope API response times."""

    rule_id = "infra.dashscope_latency"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            llm = getattr(ctx.pipeline, "_llm", None)
            if llm is None:
                return []

            # Read from latency ring buffer (deque) on LLM instance
            latencies = getattr(llm, "_recent_latencies", None)
            if not latencies or len(latencies) == 0:
                return []

            avg_latency = sum(latencies) / len(latencies)
            threshold = ctx.agent_config.dashscope_latency_warn_s

            if avg_latency >= threshold:
                severity = "critical" if avg_latency >= 5.0 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"DashScope latency: {avg_latency:.1f}s avg",
                    description=(
                        f"Average DashScope API response time is {avg_latency:.1f}s "
                        f"over last {len(latencies)} calls (threshold: {threshold}s). "
                        f"Visitor response times will be degraded."
                    ),
                    details={
                        "avg_latency_s": round(avg_latency, 2),
                        "sample_count": len(latencies),
                        "max_latency_s": round(max(latencies), 2),
                        "min_latency_s": round(min(latencies), 2),
                    },
                    recommendation=(
                        "Check DashScope API status. Consider switching to a faster "
                        "model variant or increasing timeout."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"DashScopeLatencyRule skipped: {exc}")

        return []


class ASRLatencyRule:
    """Detects high ASR (speech recognition) processing times."""

    rule_id = "infra.asr_latency"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            asr = getattr(ctx.pipeline, "_asr", None)
            if asr is None:
                return []

            latencies = getattr(asr, "_recent_latencies", None)
            if not latencies or len(latencies) == 0:
                return []

            avg_latency = sum(latencies) / len(latencies)
            threshold = ctx.agent_config.asr_latency_warn_s

            if avg_latency >= threshold:
                severity = "critical" if avg_latency >= 1.0 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"ASR latency: {avg_latency*1000:.0f}ms avg",
                    description=(
                        f"Average ASR recognition time is {avg_latency*1000:.0f}ms "
                        f"over last {len(latencies)} calls (threshold: {threshold*1000:.0f}ms). "
                        f"Visitor turn-around times will be degraded."
                    ),
                    details={
                        "avg_latency_s": round(avg_latency, 3),
                        "avg_latency_ms": round(avg_latency * 1000),
                        "sample_count": len(latencies),
                        "max_latency_s": round(max(latencies), 3),
                        "min_latency_s": round(min(latencies), 3),
                    },
                    recommendation=(
                        "Check DashScope ASR service status. Consider switching to "
                        "a faster ASR model or reducing audio chunk sizes."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"ASRLatencyRule skipped: {exc}")

        return []


class TTSLatencyRule:
    """Detects TTS synthesis times exceeding 2x realtime."""

    rule_id = "infra.tts_latency"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            tts = getattr(ctx.pipeline, "_tts", None)
            if tts is None:
                return []

            latencies = getattr(tts, "_recent_latencies", None)
            if not latencies or len(latencies) == 0:
                return []

            avg_latency = sum(latencies) / len(latencies)
            threshold = ctx.agent_config.tts_latency_warn_s

            if avg_latency >= threshold:
                severity = "critical" if avg_latency >= threshold * 2 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"TTS latency: {avg_latency:.1f}s avg",
                    description=(
                        f"Average TTS synthesis time is {avg_latency:.1f}s "
                        f"over last {len(latencies)} calls (threshold: {threshold}s). "
                        f"Audio responses will be delayed."
                    ),
                    details={
                        "avg_latency_s": round(avg_latency, 2),
                        "sample_count": len(latencies),
                        "max_latency_s": round(max(latencies), 2),
                        "min_latency_s": round(min(latencies), 2),
                    },
                    recommendation=(
                        "Check DashScope TTS service status. Consider reducing "
                        "text length per synthesis call or using a faster voice model."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"TTSLatencyRule skipped: {exc}")

        return []


class ASRConnectionRule:
    """Detects DashScope ASR WebSocket connection failures."""

    rule_id = "infra.asr_connection"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            asr = getattr(ctx.pipeline, "_asr", None)
            if asr is None:
                return []

            errors = getattr(asr, "_recent_errors", None)
            if not errors or len(errors) == 0:
                return []

            # Count errors in the last 5 minutes
            cutoff = time.time() - 300
            recent_count = sum(1 for t in errors if t >= cutoff)
            threshold = ctx.agent_config.asr_connection_fail_threshold

            if recent_count >= threshold:
                severity = "critical" if recent_count >= threshold * 2 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"ASR connection failures: {recent_count} in 5min",
                    description=(
                        f"DashScope ASR WebSocket had {recent_count} connection failures "
                        f"in the last 5 minutes (threshold: {threshold}). "
                        f"Speech recognition is degraded."
                    ),
                    details={
                        "failures_5min": recent_count,
                        "threshold": threshold,
                    },
                    recommendation=(
                        "Check DashScope ASR service status and API key validity. "
                        "Verify WebSocket URL is correct."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"ASRConnectionRule skipped: {exc}")

        return []


class TTSConnectionRule:
    """Detects DashScope TTS WebSocket connection failures."""

    rule_id = "infra.tts_connection"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            tts = getattr(ctx.pipeline, "_tts", None)
            if tts is None:
                return []

            errors = getattr(tts, "_recent_errors", None)
            if not errors or len(errors) == 0:
                return []

            cutoff = time.time() - 300
            recent_count = sum(1 for t in errors if t >= cutoff)
            threshold = ctx.agent_config.tts_connection_fail_threshold

            if recent_count >= threshold:
                severity = "critical" if recent_count >= threshold * 2 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"TTS connection failures: {recent_count} in 5min",
                    description=(
                        f"DashScope TTS WebSocket had {recent_count} connection failures "
                        f"in the last 5 minutes (threshold: {threshold}). "
                        f"Voice synthesis is degraded."
                    ),
                    details={
                        "failures_5min": recent_count,
                        "threshold": threshold,
                    },
                    recommendation=(
                        "Check DashScope TTS service status and API key validity. "
                        "Verify WebSocket URL is correct."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"TTSConnectionRule skipped: {exc}")

        return []


class DashScopeQuotaRule:
    """Detects when DashScope API usage is nearing monthly quota."""

    rule_id = "infra.dashscope_quota"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        try:
            from datetime import datetime

            from sqlalchemy import func, select
            from src.db.models import APICostRecord

            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            async with ctx.db.session_ctx() as session:
                # Sum DashScope costs this month (ASR + LLM + TTS)
                stmt = (
                    select(func.coalesce(func.sum(APICostRecord.cost_usd), 0))
                    .where(APICostRecord.created_at >= month_start)
                    .where(APICostRecord.service.in_(["asr", "llm", "tts"]))
                )
                result = await session.execute(stmt)
                total_cost = result.scalar() or 0.0

            monthly_budget = ctx.agent_config.dashscope_monthly_budget_usd
            if monthly_budget <= 0:
                return []

            usage_pct = (total_cost / monthly_budget) * 100
            warn_pct = ctx.agent_config.dashscope_quota_warn_pct

            if usage_pct >= warn_pct:
                remaining = monthly_budget - total_cost
                severity = "critical" if usage_pct >= 100 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"DashScope quota: {usage_pct:.0f}% used (${total_cost:.2f}/${monthly_budget:.0f})",
                    description=(
                        f"DashScope API costs are ${total_cost:.2f} this month "
                        f"({usage_pct:.0f}% of ${monthly_budget:.0f} budget). "
                        f"{'Budget exceeded!' if usage_pct >= 100 else f'${remaining:.2f} remaining.'}"
                    ),
                    details={
                        "total_cost_usd": round(total_cost, 2),
                        "monthly_budget_usd": monthly_budget,
                        "usage_pct": round(usage_pct, 1),
                        "remaining_usd": round(max(0, remaining), 2),
                    },
                    recommendation=(
                        "Review per-customer usage. Consider rate-limiting heavy users "
                        "or upgrading DashScope plan."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"DashScopeQuotaRule skipped: {exc}")

        return []


class RunPodQueueDepthRule:
    """Detects high RunPod serverless queue depth (pending jobs)."""

    rule_id = "infra.runpod_queue"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        config = ctx.config
        if config is None:
            return []

        endpoint_url = getattr(config, "runpod_endpoint_musetalk", "") or ""
        api_key = getattr(config, "runpod_api_key", "") or ""

        if not endpoint_url or not api_key:
            return []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{endpoint_url}/health",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    return []

                data = resp.json()
                if not isinstance(data, dict):
                    return []
                # RunPod health returns: jobs.inQueue, jobs.inProgress, workers.idle, etc.
                jobs = data.get("jobs", {})
                in_queue = jobs.get("inQueue", 0)
                in_progress = jobs.get("inProgress", 0)
                threshold = ctx.agent_config.runpod_queue_depth_warn

                if in_queue >= threshold:
                    severity = "critical" if in_queue >= threshold * 3 else "warning"
                    return [Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"RunPod queue depth: {in_queue} pending, {in_progress} running",
                        description=(
                            f"RunPod has {in_queue} jobs queued and {in_progress} running. "
                            f"Threshold: {threshold}. Rendering latency will increase."
                        ),
                        details={
                            "in_queue": in_queue,
                            "in_progress": in_progress,
                            "workers": data.get("workers", {}),
                        },
                        recommendation=(
                            "Scale up RunPod workers or throttle incoming render requests."
                        ),
                    )]
        except Exception as exc:
            logger.debug(f"RunPodQueueDepthRule skipped: {exc}")

        return []


class RunPodRenderTimeRule:
    """Detects high average RunPod render times."""

    rule_id = "infra.runpod_render_time"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        detections: list[Detection] = []

        # Check from RunPodServerless client's ring buffer (if pipeline available)
        if ctx.pipeline is not None:
            try:
                runpod = getattr(ctx.pipeline, "_runpod", None)
                if runpod is None:
                    # Try alternate attribute names
                    runpod = getattr(ctx.pipeline, "_runpod_client", None)
                if runpod is not None:
                    render_times = getattr(runpod, "_recent_render_times", None)
                    if render_times and len(render_times) > 0:
                        avg_time = sum(render_times) / len(render_times)
                        warn_s = ctx.agent_config.runpod_render_time_warn_s
                        cold_s = ctx.agent_config.runpod_render_time_cold_s

                        if avg_time >= cold_s:
                            detections.append(Detection(
                                rule_id=self.rule_id,
                                severity="critical",
                                title=f"RunPod render time: {avg_time:.1f}s avg (cold starts)",
                                description=(
                                    f"Average RunPod render time is {avg_time:.1f}s "
                                    f"over last {len(render_times)} jobs "
                                    f"(cold start threshold: {cold_s}s). "
                                    f"Workers may be scaling up from zero."
                                ),
                                details={
                                    "avg_time_s": round(avg_time, 1),
                                    "sample_count": len(render_times),
                                    "max_time_s": round(max(render_times), 1),
                                    "min_time_s": round(min(render_times), 1),
                                },
                                recommendation=(
                                    "Set RunPod min workers > 0 to avoid cold starts. "
                                    "Or increase the active workers flashboot setting."
                                ),
                            ))
                        elif avg_time >= warn_s:
                            detections.append(Detection(
                                rule_id=self.rule_id,
                                severity="warning",
                                title=f"RunPod render time: {avg_time:.1f}s avg",
                                description=(
                                    f"Average RunPod render time is {avg_time:.1f}s "
                                    f"over last {len(render_times)} jobs "
                                    f"(warm threshold: {warn_s}s). "
                                    f"Video responses will be delayed."
                                ),
                                details={
                                    "avg_time_s": round(avg_time, 1),
                                    "sample_count": len(render_times),
                                    "max_time_s": round(max(render_times), 1),
                                    "min_time_s": round(min(render_times), 1),
                                },
                                recommendation=(
                                    "Check RunPod worker utilization. Consider upgrading "
                                    "GPU tier or optimizing the render pipeline."
                                ),
                            ))
            except Exception as exc:
                logger.debug(f"RunPodRenderTimeRule (client) skipped: {exc}")

        # Fallback: check from DB APICostRecord
        if not detections and ctx.db is not None:
            try:
                from datetime import datetime, timedelta

                from sqlalchemy import func, select
                from src.db.models import APICostRecord

                cutoff = datetime.utcnow() - timedelta(hours=1)

                async with ctx.db.session_ctx() as session:
                    stmt = (
                        select(func.avg(APICostRecord.duration_ms))
                        .where(
                            APICostRecord.service == "runpod",
                            APICostRecord.created_at >= cutoff,
                            APICostRecord.duration_ms > 0,
                        )
                    )
                    result = await session.execute(stmt)
                    avg_ms = result.scalar()

                if avg_ms is not None:
                    avg_s = avg_ms / 1000.0
                    warn_s = ctx.agent_config.runpod_render_time_warn_s

                    if avg_s >= warn_s:
                        severity = "critical" if avg_s >= ctx.agent_config.runpod_render_time_cold_s else "warning"
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity=severity,
                            title=f"RunPod render time: {avg_s:.1f}s avg (DB)",
                            description=(
                                f"Average RunPod render time from DB is {avg_s:.1f}s "
                                f"in the last hour (threshold: {warn_s}s)."
                            ),
                            details={"avg_time_s": round(avg_s, 1), "source": "db"},
                            recommendation=(
                                "Check RunPod worker utilization and queue depth."
                            ),
                        ))
            except Exception as exc:
                logger.debug(f"RunPodRenderTimeRule (DB) skipped: {exc}")

        return detections


class RunPodHealthRule:
    """Detects RunPod endpoint failures or high error rates."""

    rule_id = "infra.runpod_health"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        detections: list[Detection] = []
        config = ctx.config
        if config is None:
            return []

        # Check RunPod endpoint reachability
        endpoint_url = getattr(config, "runpod_endpoint_musetalk", "") or ""
        api_key = getattr(config, "runpod_api_key", "") or ""

        if endpoint_url and api_key:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{endpoint_url}/health",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if resp.status_code >= 500:
                        detections.append(Detection(
                            rule_id=self.rule_id,
                            severity="critical",
                            title="RunPod endpoint unhealthy",
                            description=(
                                f"RunPod endpoint returned HTTP {resp.status_code}. "
                                f"GPU rendering is unavailable."
                            ),
                            details={
                                "endpoint": endpoint_url,
                                "status_code": resp.status_code,
                            },
                            recommendation="Check RunPod dashboard for endpoint status and worker health.",
                        ))
            except Exception as exc:
                detections.append(Detection(
                    rule_id=self.rule_id,
                    severity="critical",
                    title="RunPod endpoint unreachable",
                    description=f"Cannot connect to RunPod endpoint: {exc}",
                    details={"endpoint": endpoint_url, "error": str(exc)},
                    recommendation="Check network connectivity and RunPod service status.",
                ))

        # Check recent job error rate from APICostRecord
        if ctx.db is not None:
            try:
                from datetime import datetime, timedelta

                from sqlalchemy import func, select
                from src.db.models import APICostRecord

                cutoff = datetime.utcnow() - timedelta(hours=1)

                async with ctx.db.session_ctx() as session:
                    total_r = await session.execute(
                        select(func.count(APICostRecord.id))
                        .where(
                            APICostRecord.service == "runpod",
                            APICostRecord.created_at >= cutoff,
                        )
                    )
                    total = total_r.scalar() or 0

                    if total >= 5:  # Need minimum sample
                        error_r = await session.execute(
                            select(func.count(APICostRecord.id))
                            .where(
                                APICostRecord.service == "runpod",
                                APICostRecord.created_at >= cutoff,
                                APICostRecord.details.like('%"error"%'),
                            )
                        )
                        errors = error_r.scalar() or 0
                        error_rate = errors / total
                        threshold = ctx.agent_config.runpod_error_rate_warn

                        if error_rate >= threshold:
                            detections.append(Detection(
                                rule_id=self.rule_id,
                                severity="warning",
                                title=f"RunPod error rate: {error_rate*100:.0f}%",
                                description=(
                                    f"{errors}/{total} RunPod jobs failed in the last hour "
                                    f"({error_rate*100:.0f}%). Threshold: {threshold*100:.0f}%."
                                ),
                                details={
                                    "total_jobs": total,
                                    "failed_jobs": errors,
                                    "error_rate": round(error_rate, 3),
                                },
                                recommendation="Check RunPod worker logs for errors. May need to redeploy worker.",
                            ))
            except Exception as exc:
                logger.debug(f"RunPod error rate check skipped: {exc}")

        return detections


class R2ConnectivityRule:
    """Detects Cloudflare R2 connectivity failures."""

    rule_id = "infra.r2_connectivity"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        config = ctx.config
        if config is None:
            return []

        bucket = getattr(config, "r2_bucket", "") or ""
        account_id = getattr(config, "r2_account_id", "") or ""
        if not bucket or not account_id:
            return []

        try:
            import boto3
            from botocore.config import Config as BotoConfig

            endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=getattr(config, "r2_access_key_id", ""),
                aws_secret_access_key=getattr(config, "r2_secret_access_key", ""),
                config=BotoConfig(
                    connect_timeout=5,
                    read_timeout=5,
                    retries={"max_attempts": 1},
                ),
            )
            client.list_objects_v2(Bucket=bucket, Prefix="__healthcheck__", MaxKeys=1)
        except ImportError:
            return []  # boto3 not available
        except Exception as exc:
            return [Detection(
                rule_id=self.rule_id,
                severity="critical",
                title="R2 storage unreachable",
                description=(
                    f"Cannot connect to Cloudflare R2 bucket '{bucket}': {exc}. "
                    f"Photo uploads and media storage will fail."
                ),
                details={"bucket": bucket, "endpoint": endpoint, "error": str(exc)},
                recommendation="Check R2 credentials and endpoint URL. Verify bucket exists.",
            )]

        return []


class RunPodWorkersRule:
    """Monitors RunPod active/idle worker counts and detects warm pool issues."""

    rule_id = "infra.runpod_workers"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        config = ctx.config
        if config is None:
            return []

        endpoint_url = getattr(config, "runpod_endpoint_musetalk", "") or ""
        api_key = getattr(config, "runpod_api_key", "") or ""
        if not endpoint_url or not api_key:
            return []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{endpoint_url}/health",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    return []

                data = resp.json()
                if not isinstance(data, dict):
                    return []
                workers = data.get("workers", {})
                idle = workers.get("idle", 0)
                running = workers.get("running", 0)
                ready = workers.get("ready", 0)
                throttled = workers.get("throttled", 0)

                detections: list[Detection] = []
                min_idle = ctx.agent_config.runpod_min_idle_workers

                # Check if we need warm pool but have no idle workers
                active_customers = 0
                if ctx.db is not None:
                    try:
                        from sqlalchemy import func, select
                        from src.db.models import Customer

                        async with ctx.db.session_ctx() as session:
                            r = await session.execute(
                                select(func.count(Customer.id))
                                .where(Customer.is_active == True)  # noqa: E712
                            )
                            active_customers = r.scalar() or 0
                    except Exception:
                        pass

                threshold_customers = ctx.agent_config.runpod_active_customers_for_warm

                if idle < min_idle and active_customers >= threshold_customers:
                    severity = "critical" if idle == 0 and running > 0 else "warning"
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"RunPod workers: {idle} idle, {running} running ({active_customers} active customers)",
                        description=(
                            f"RunPod has {idle} idle workers with {active_customers} active customers "
                            f"(threshold: {threshold_customers}). New requests will hit cold starts. "
                            f"Minimum idle workers: {min_idle}."
                        ),
                        details={
                            "idle": idle,
                            "running": running,
                            "ready": ready,
                            "throttled": throttled,
                            "active_customers": active_customers,
                        },
                        recommendation=(
                            "Increase RunPod min workers setting to maintain a warm pool. "
                            "Set min_workers >= 1 to avoid cold start latency."
                        ),
                        auto_fixable=idle == 0,
                    ))

                # Log scaling event if throttled workers detected
                if throttled > 0:
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="warning",
                        title=f"RunPod scaling: {throttled} workers throttled",
                        description=(
                            f"RunPod has {throttled} throttled workers. "
                            f"The endpoint may be hitting its max workers limit."
                        ),
                        details={
                            "idle": idle,
                            "running": running,
                            "throttled": throttled,
                        },
                        recommendation="Increase max workers on RunPod endpoint to handle load.",
                    ))

                return detections
        except Exception as exc:
            logger.debug(f"RunPodWorkersRule skipped: {exc}")

        return []


class RunPodColdStartRule:
    """Detects high cold start frequency in RunPod jobs."""

    rule_id = "infra.runpod_cold_starts"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
            if runpod is None:
                return []

            # Use lipsync times for cold start detection (render jobs are the main concern)
            render_times = getattr(runpod, "_recent_lipsync_times", None)
            if not render_times or len(render_times) < 3:
                # Fallback to all render times
                render_times = getattr(runpod, "_recent_render_times", None)
                if not render_times or len(render_times) < 3:
                    return []

            cold_threshold = ctx.agent_config.runpod_lipsync_cold_warn_s
            total = len(render_times)
            cold_count = sum(1 for t in render_times if t >= cold_threshold)
            cold_pct = (cold_count / total) * 100

            warn_pct = ctx.agent_config.runpod_cold_start_pct_warn

            if cold_pct >= warn_pct:
                severity = "critical" if cold_pct >= 60 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"RunPod cold starts: {cold_pct:.0f}% ({cold_count}/{total} jobs)",
                    description=(
                        f"{cold_count} of {total} recent RunPod jobs exceeded {cold_threshold}s "
                        f"({cold_pct:.0f}% cold starts, threshold: {warn_pct:.0f}%). "
                        f"Workers are frequently scaling from zero."
                    ),
                    details={
                        "cold_count": cold_count,
                        "total_jobs": total,
                        "cold_pct": round(cold_pct, 1),
                        "cold_threshold_s": cold_threshold,
                    },
                    recommendation=(
                        "Set RunPod min workers > 0 to keep workers warm. "
                        "Enable FlashBoot for faster cold starts."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"RunPodColdStartRule skipped: {exc}")

        return []


class RunPodTaskTimeRule:
    """Monitors per-task-type execution times (preprocess_face vs render_lipsync)."""

    rule_id = "infra.runpod_task_time"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        detections: list[Detection] = []

        try:
            runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
            if runpod is None:
                return []

            # Check preprocess_face times
            preprocess_times = getattr(runpod, "_recent_preprocess_times", None)
            if preprocess_times and len(preprocess_times) > 0:
                avg_time = sum(preprocess_times) / len(preprocess_times)
                threshold = ctx.agent_config.runpod_preprocess_time_warn_s

                if avg_time >= threshold:
                    severity = "critical" if avg_time >= threshold * 2 else "warning"
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"preprocess_face: {avg_time:.1f}s avg (threshold: {threshold:.0f}s)",
                        description=(
                            f"Average preprocess_face execution time is {avg_time:.1f}s "
                            f"over last {len(preprocess_times)} jobs (threshold: {threshold:.0f}s). "
                            f"Photo uploads will be slow for new avatars."
                        ),
                        details={
                            "task_type": "preprocess_face",
                            "avg_time_s": round(avg_time, 1),
                            "sample_count": len(preprocess_times),
                            "max_time_s": round(max(preprocess_times), 1),
                            "min_time_s": round(min(preprocess_times), 1),
                        },
                        recommendation=(
                            "Check RunPod worker logs for preprocessing bottlenecks. "
                            "May need to optimize image resolution or MediaPipe settings."
                        ),
                    ))

            # Check render_lipsync times (warm vs cold)
            lipsync_times = getattr(runpod, "_recent_lipsync_times", None)
            if lipsync_times and len(lipsync_times) > 0:
                avg_time = sum(lipsync_times) / len(lipsync_times)
                warm_threshold = ctx.agent_config.runpod_lipsync_warm_warn_s
                cold_threshold = ctx.agent_config.runpod_lipsync_cold_warn_s

                if avg_time >= cold_threshold:
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="critical",
                        title=f"render_lipsync: {avg_time:.1f}s avg (cold starts)",
                        description=(
                            f"Average render_lipsync execution time is {avg_time:.1f}s "
                            f"over last {len(lipsync_times)} jobs "
                            f"(cold threshold: {cold_threshold:.0f}s). "
                            f"Workers may be scaling up from zero."
                        ),
                        details={
                            "task_type": "render_lipsync",
                            "avg_time_s": round(avg_time, 1),
                            "sample_count": len(lipsync_times),
                            "max_time_s": round(max(lipsync_times), 1),
                            "min_time_s": round(min(lipsync_times), 1),
                        },
                        recommendation=(
                            "Set RunPod min workers > 0. Cold starts add 7-10s to render time."
                        ),
                    ))
                elif avg_time >= warm_threshold:
                    detections.append(Detection(
                        rule_id=self.rule_id,
                        severity="warning",
                        title=f"render_lipsync: {avg_time:.1f}s avg (warm threshold: {warm_threshold:.0f}s)",
                        description=(
                            f"Average render_lipsync execution time is {avg_time:.1f}s "
                            f"over last {len(lipsync_times)} jobs "
                            f"(warm threshold: {warm_threshold:.0f}s). "
                            f"Video responses may feel slow."
                        ),
                        details={
                            "task_type": "render_lipsync",
                            "avg_time_s": round(avg_time, 1),
                            "sample_count": len(lipsync_times),
                            "max_time_s": round(max(lipsync_times), 1),
                            "min_time_s": round(min(lipsync_times), 1),
                        },
                        recommendation=(
                            "Check RunPod worker GPU utilization. Consider upgrading "
                            "to faster GPU tier or optimizing MuseTalk pipeline."
                        ),
                    ))
        except Exception as exc:
            logger.debug(f"RunPodTaskTimeRule skipped: {exc}")

        return detections


class RunPodR2LatencyRule:
    """Detects high R2 upload latency from RunPod workers."""

    rule_id = "infra.runpod_r2_latency"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
            if runpod is None:
                return []

            r2_latencies = getattr(runpod, "_recent_r2_latencies", None)
            if not r2_latencies or len(r2_latencies) == 0:
                return []

            avg_latency = sum(r2_latencies) / len(r2_latencies)
            threshold = ctx.agent_config.runpod_r2_latency_warn_s

            if avg_latency >= threshold:
                severity = "critical" if avg_latency >= threshold * 2 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"RunPod→R2 upload: {avg_latency:.1f}s avg",
                    description=(
                        f"Average R2 upload latency from RunPod workers is {avg_latency:.1f}s "
                        f"over last {len(r2_latencies)} uploads (threshold: {threshold}s). "
                        f"Video delivery will be delayed."
                    ),
                    details={
                        "avg_latency_s": round(avg_latency, 2),
                        "sample_count": len(r2_latencies),
                        "max_latency_s": round(max(r2_latencies), 2),
                        "min_latency_s": round(min(r2_latencies), 2),
                    },
                    recommendation=(
                        "Check R2 endpoint connectivity from RunPod region. "
                        "Consider using a closer R2 region or optimizing upload size."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"RunPodR2LatencyRule skipped: {exc}")

        return []


class RunPodTimeoutRule:
    """Detects high RunPod job timeout rate."""

    rule_id = "infra.runpod_timeouts"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
            if runpod is None:
                return []

            timeouts = getattr(runpod, "_recent_timeouts", None)
            render_times = getattr(runpod, "_recent_render_times", None)

            if not timeouts or len(timeouts) == 0:
                return []

            # Count timeouts in last hour
            cutoff = time.time() - 3600
            recent_timeouts = sum(1 for t in timeouts if t >= cutoff)
            if recent_timeouts == 0:
                return []

            # Total jobs = successful renders + timeouts + failures
            total_renders = len(render_times) if render_times else 0
            failures = getattr(runpod, "_recent_failures", None)
            recent_failures = sum(1 for t in failures if t >= cutoff) if failures else 0
            total_jobs = total_renders + recent_timeouts + recent_failures

            if total_jobs < 3:
                return []

            timeout_rate = recent_timeouts / total_jobs
            threshold = ctx.agent_config.runpod_timeout_rate_warn

            if timeout_rate >= threshold:
                severity = "critical" if timeout_rate >= 0.25 else "warning"
                return [Detection(
                    rule_id=self.rule_id,
                    severity=severity,
                    title=f"RunPod timeouts: {recent_timeouts}/{total_jobs} ({timeout_rate*100:.0f}%)",
                    description=(
                        f"{recent_timeouts} of {total_jobs} RunPod jobs timed out in the last hour "
                        f"({timeout_rate*100:.0f}%). Threshold: {threshold*100:.0f}%. "
                        f"Workers may be overloaded or unresponsive."
                    ),
                    details={
                        "timeouts": recent_timeouts,
                        "total_jobs": total_jobs,
                        "timeout_rate": round(timeout_rate, 3),
                    },
                    recommendation=(
                        "Check RunPod worker health. Workers may be OOM or stuck. "
                        "Consider increasing job timeout or redeploying workers."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"RunPodTimeoutRule skipped: {exc}")

        return []


class RunPodConsecutiveFailureRule:
    """Detects consecutive RunPod job failures and triggers video disable escalation."""

    rule_id = "resilience.runpod_consecutive_failures"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            runpod = getattr(ctx.pipeline, "_runpod", None) or getattr(ctx.pipeline, "_runpod_client", None)
            if runpod is None:
                return []

            consecutive = getattr(runpod, "_consecutive_failures", 0)
            threshold = ctx.agent_config.runpod_consecutive_failure_threshold

            if consecutive >= threshold:
                video_disabled = getattr(runpod, "video_disabled", False)
                return [Detection(
                    rule_id=self.rule_id,
                    severity="critical",
                    title=f"RunPod: {consecutive} consecutive failures",
                    description=(
                        f"RunPod has had {consecutive} consecutive job failures "
                        f"(threshold: {threshold}). Video rendering should be disabled "
                        f"to protect visitor experience."
                    ),
                    details={
                        "consecutive_failures": consecutive,
                        "threshold": threshold,
                        "video_disabled": video_disabled,
                    },
                    recommendation=(
                        "Check RunPod worker logs. Redeploy workers if needed. "
                        "Video will be disabled until failures clear."
                    ),
                    auto_fixable=not video_disabled,
                )]
        except Exception as exc:
            logger.debug(f"RunPodConsecutiveFailureRule skipped: {exc}")

        return []


class DashScopeConsecutiveTimeoutRule:
    """Detects consecutive DashScope timeouts and triggers text-only mode."""

    rule_id = "resilience.dashscope_consecutive_timeouts"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            llm = getattr(ctx.pipeline, "_llm", None)
            if llm is None:
                return []

            consecutive = getattr(llm, "_consecutive_timeouts", 0)
            threshold = ctx.agent_config.dashscope_consecutive_timeout_threshold

            if consecutive >= threshold:
                text_only = getattr(llm, "text_only_mode", False)
                return [Detection(
                    rule_id=self.rule_id,
                    severity="critical",
                    title=f"DashScope: {consecutive} consecutive timeouts",
                    description=(
                        f"DashScope LLM has timed out {consecutive} times consecutively "
                        f"(threshold: {threshold}). TTS/rendering should be skipped "
                        f"to preserve basic text responses."
                    ),
                    details={
                        "consecutive_timeouts": consecutive,
                        "threshold": threshold,
                        "text_only_mode": text_only,
                    },
                    recommendation=(
                        "Check DashScope API status and network connectivity. "
                        "Text-only mode will be enabled until service recovers."
                    ),
                    auto_fixable=not text_only,
                )]
        except Exception as exc:
            logger.debug(f"DashScopeConsecutiveTimeoutRule skipped: {exc}")

        return []


class DashScopeQueueDepthRule:
    """Detects high pending request count on the DashScope LLM client."""

    rule_id = "resilience.dashscope_queue_depth"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            llm = getattr(ctx.pipeline, "_llm", None)
            if llm is None:
                return []

            pending = getattr(llm, "_pending_requests", 0)
            rate_limit_count = getattr(llm, "_rate_limit_429_count", 0)
            threshold = ctx.agent_config.dashscope_queue_depth_warn

            if pending >= threshold:
                return [Detection(
                    rule_id=self.rule_id,
                    severity="critical",
                    title=f"DashScope queue: {pending} pending requests",
                    description=(
                        f"DashScope LLM client has {pending} pending requests "
                        f"(threshold: {threshold}). Rate limit hits: {rate_limit_count}. "
                        f"Visitor responses will be severely delayed."
                    ),
                    details={
                        "pending_requests": pending,
                        "threshold": threshold,
                        "rate_limit_429_count": rate_limit_count,
                    },
                    recommendation=(
                        "DashScope is overloaded. Reduce concurrent sessions or "
                        "upgrade DashScope QPS quota."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"DashScopeQueueDepthRule skipped: {exc}")

        return []


class R2DowntimeRule:
    """Detects prolonged R2 upload failures (R2 down > 5 minutes)."""

    rule_id = "resilience.r2_downtime"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.pipeline is None:
            return []

        try:
            r2 = getattr(ctx.pipeline, "_r2", None) or getattr(ctx.pipeline, "_r2_storage", None)
            if r2 is None:
                return []

            consecutive = getattr(r2, "_consecutive_failures", 0)
            last_failure = getattr(r2, "_last_failure_time", 0.0)

            if consecutive == 0 or last_failure == 0.0:
                return []

            since_last_failure_s = time.time() - last_failure
            threshold_s = ctx.agent_config.r2_downtime_threshold_s

            # since_last_failure_s measures how long ago the last failure was.
            # If consecutive is still high, no successful upload has reset
            # the counter. Two cases:
            #   <= threshold: failures started recently, still developing → warning
            #   >  threshold: failures ongoing for extended period → critical
            if consecutive >= 3 and since_last_failure_s <= threshold_s:
                # Failures started recently — still accumulating
                return [Detection(
                    rule_id=self.rule_id,
                    severity="warning",
                    title=f"R2 storage: {consecutive} consecutive upload failures",
                    description=(
                        f"R2 has had {consecutive} consecutive upload failures. "
                        f"Media uploads are failing. Monitor for extended downtime."
                    ),
                    details={
                        "consecutive_failures": consecutive,
                        "last_failure_age_s": round(since_last_failure_s, 1),
                    },
                    recommendation="Check R2 credentials and endpoint connectivity.",
                )]
            elif consecutive >= 3 and since_last_failure_s > threshold_s:
                # Extended downtime — failures ongoing for > threshold
                return [Detection(
                    rule_id=self.rule_id,
                    severity="critical",
                    title=f"R2 storage DOWN: {consecutive} failures over {since_last_failure_s:.0f}s",
                    description=(
                        f"R2 storage has been failing for {since_last_failure_s:.0f}s "
                        f"with {consecutive} consecutive failures "
                        f"(threshold: {threshold_s:.0f}s). "
                        f"All media uploads are failing."
                    ),
                    details={
                        "consecutive_failures": consecutive,
                        "since_last_failure_s": round(since_last_failure_s, 1),
                        "threshold_s": threshold_s,
                    },
                    recommendation=(
                        "R2 has been down for an extended period. "
                        "Check Cloudflare status page and R2 credentials. "
                        "Consider switching to local tmp fallback."
                    ),
                )]
        except Exception as exc:
            logger.debug(f"R2DowntimeRule skipped: {exc}")

        return []


class MarginSqueezeRule:
    """Detects when a customer's API cost exceeds a percentage of their revenue."""

    rule_id = "resilience.margin_squeeze"

    async def evaluate(self, ctx: AgentContext) -> list[Detection]:
        if ctx.db is None:
            return []

        detections: list[Detection] = []
        threshold_pct = ctx.agent_config.margin_squeeze_pct

        try:
            from sqlalchemy import func, select
            from src.db.models import APICostRecord, Customer, Subscription

            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            async with ctx.db.session_ctx() as session:
                # Get active customers with subscriptions
                stmt = (
                    select(
                        Customer.id,
                        Customer.company,
                        Subscription.plan,
                        Subscription.price_monthly,
                    )
                    .join(Subscription, Subscription.customer_id == Customer.id)
                    .where(
                        Customer.is_active == True,  # noqa: E712
                        Subscription.is_active == True,  # noqa: E712
                        Subscription.price_monthly > 0,
                    )
                )
                customers = (await session.execute(stmt)).all()

                for cust_id, company, plan, revenue in customers:
                    # Sum API costs for this customer this month
                    cost_stmt = (
                        select(func.coalesce(func.sum(APICostRecord.cost_usd), 0))
                        .where(
                            APICostRecord.customer_id == cust_id,
                            APICostRecord.created_at >= month_start,
                        )
                    )
                    cost_result = await session.execute(cost_stmt)
                    total_cost = cost_result.scalar() or 0.0

                    if revenue > 0:
                        cost_pct = (total_cost / revenue) * 100
                        if cost_pct >= threshold_pct:
                            severity = "critical" if cost_pct >= 80 else "warning"
                            detections.append(Detection(
                                rule_id=self.rule_id,
                                severity=severity,
                                title=f"Margin squeeze: {company} cost {cost_pct:.0f}% of revenue",
                                description=(
                                    f"Customer '{company}' ({plan}) has API costs of "
                                    f"${total_cost:.2f} against ${revenue:.2f}/mo revenue "
                                    f"({cost_pct:.0f}%). Margin is being squeezed."
                                ),
                                details={
                                    "customer_id": cust_id,
                                    "company": company,
                                    "plan": plan,
                                    "cost_usd": round(total_cost, 2),
                                    "revenue_usd": round(revenue, 2),
                                    "cost_pct": round(cost_pct, 1),
                                },
                                recommendation=(
                                    f"Suggest plan upgrade for '{company}'. "
                                    f"Current {plan} plan may be insufficient for their usage."
                                ),
                            ))
        except Exception as exc:
            logger.debug(f"MarginSqueezeRule skipped: {exc}")

        return detections
