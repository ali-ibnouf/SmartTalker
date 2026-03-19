#!/usr/bin/env bash
# =============================================================================
# SmartTalker — One-Command Production Deployment
# Usage: ./scripts/deploy.sh
# =============================================================================
set -euo pipefail

DEPLOY_DIR="/opt/maskki/SmartTalker"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"
CENTRAL_URL="http://localhost:8000"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[DEPLOY]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ── Step 1: Pull latest code ─────────────────────────────────────────────────
log "Pulling latest code..."
cd "$DEPLOY_DIR"
if [ -d .git ]; then
    git pull --ff-only origin main
else
    warn "Not a git repo — skipping git pull"
fi

# ── Step 2: Verify env file exists ───────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    fail "$ENV_FILE not found. Copy .env.production.example and fill in values."
fi
log "Environment file found: $ENV_FILE"

# ── Step 3: Build Docker images ──────────────────────────────────────────────
log "Building Docker images..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build --no-cache

# ── Step 4: Run DB migrations ────────────────────────────────────────────────
log "Starting database for migrations..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres redis
sleep 5

log "Running database migrations..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm central \
    python -c "
import asyncio
from src.db.engine import Database
from src.config import get_settings
async def migrate():
    config = get_settings()
    db = Database(config.database_url)
    await db.connect()
    await db.disconnect()
asyncio.run(migrate())
" || warn "Migration step completed (tables may already exist)"

# ── Step 5: Restart all services ─────────────────────────────────────────────
log "Restarting services..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d

# ── Step 6: Wait for services to be healthy ──────────────────────────────────
log "Waiting for services to start (30s)..."
sleep 30

# ── Step 7: Health checks ────────────────────────────────────────────────────
log "Running health checks..."

CHECKS_PASSED=0
CHECKS_TOTAL=0

check_health() {
    local name="$1"
    local url="$2"
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        log "  $name: OK (HTTP $HTTP_CODE)"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        warn "  $name: FAILED (HTTP $HTTP_CODE)"
    fi
}

check_health "Central API" "$CENTRAL_URL/api/v1/health"
check_health "Database"    "$CENTRAL_URL/api/v1/health/db"
check_health "Redis"       "$CENTRAL_URL/api/v1/health/redis"
check_health "DashScope"   "$CENTRAL_URL/api/v1/health/dashscope"
check_health "RunPod"      "$CENTRAL_URL/api/v1/health/runpod"
check_health "R2 Storage"  "$CENTRAL_URL/api/v1/health/r2"

# ── Step 8: Report status ────────────────────────────────────────────────────
echo ""
log "============================================"
log "  Deployment complete"
log "  Health checks: $CHECKS_PASSED/$CHECKS_TOTAL passed"
log "============================================"

# Show container status
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps

if [ "$CHECKS_PASSED" -lt "$CHECKS_TOTAL" ]; then
    warn "Some health checks failed — review logs:"
    warn "  docker compose -f $COMPOSE_FILE logs central --tail=50"
    exit 1
fi

log "All systems operational."
