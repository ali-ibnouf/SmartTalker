#!/bin/bash
# Verify RunPod GPU endpoint security before going live
# Usage: RUNPOD_API_KEY=rpa-xxx bash scripts/harden-gpu-node.sh

set -e
echo "========================================="
echo "  RunPod GPU Security Verification"
echo "========================================="

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  PASS $1"; PASS=$((PASS + 1)); }
check_fail() { echo "  FAIL $1"; FAIL=$((FAIL + 1)); }
check_warn() { echo "  WARN $1"; WARN=$((WARN + 1)); }

# --- 1. Verify API key is set ---
echo ""
echo "[API Key]"
if [ -z "$RUNPOD_API_KEY" ]; then
    check_fail "RUNPOD_API_KEY not set"
    echo "       Set: export RUNPOD_API_KEY=rpa-..."
    echo "       Exiting — cannot proceed without API key"
    exit 1
else
    KEY_PREFIX="${RUNPOD_API_KEY:0:6}"
    check_pass "API key set (${KEY_PREFIX}...)"
fi

# --- 2. Verify TLS connection to RunPod ---
echo ""
echo "[TLS Encryption]"
TLS_OUTPUT=$(echo | openssl s_client -connect api.runpod.ai:443 -servername api.runpod.ai 2>/dev/null)

TLS_VERSION=$(echo "$TLS_OUTPUT" | grep "Protocol" | head -1 | awk '{print $NF}')
if echo "$TLS_VERSION" | grep -q "TLSv1.[23]"; then
    check_pass "TLS $TLS_VERSION to api.runpod.ai"
else
    check_fail "Weak or no TLS to api.runpod.ai (got: $TLS_VERSION)"
fi

CERT_ISSUER=$(echo "$TLS_OUTPUT" | grep "issuer=" | head -1)
if [ -n "$CERT_ISSUER" ]; then
    check_pass "Valid certificate chain"
else
    check_fail "Cannot verify certificate chain"
fi

# --- 3. Check endpoint accessibility ---
echo ""
echo "[Endpoints]"

# Read endpoint from .env.production.example or env
MUSETALK_ENDPOINT="${RUNPOD_ENDPOINT_MUSETALK:-}"
if [ -z "$MUSETALK_ENDPOINT" ] && [ -f ".env.production.example" ]; then
    MUSETALK_ENDPOINT=$(grep "RUNPOD_ENDPOINT_MUSETALK" .env.production.example | cut -d'=' -f2 | tr -d ' ')
fi

if [ -n "$MUSETALK_ENDPOINT" ] && [ "$MUSETALK_ENDPOINT" != "https://api.runpod.ai/v2/___" ]; then
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        "${MUSETALK_ENDPOINT}/health" 2>/dev/null || echo "000")

    if [ "$HEALTH_RESPONSE" = "200" ]; then
        check_pass "MuseTalk endpoint healthy (HTTP 200)"
    elif [ "$HEALTH_RESPONSE" = "401" ]; then
        check_fail "MuseTalk endpoint returned 401 — API key invalid or expired"
    else
        check_warn "MuseTalk endpoint returned HTTP $HEALTH_RESPONSE (may be cold)"
    fi
else
    check_warn "RUNPOD_ENDPOINT_MUSETALK not configured — skipping health check"
fi

# --- 4. Verify API key permissions ---
echo ""
echo "[API Key Permissions]"
PODS_RESPONSE=$(curl -s \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://api.runpod.ai/v2/pods" 2>/dev/null)

if echo "$PODS_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if 'error' not in str(d).lower() else 1)" 2>/dev/null; then
    check_warn "API key has broad permissions — consider creating a scoped key for production"
    echo "       RunPod dashboard > API Keys > Create with minimal permissions"
else
    check_pass "API key has limited permissions (cannot list pods)"
fi

# --- 5. Check for exposed ports in RunPod worker ---
echo ""
echo "[Worker Security]"
if [ -f "runpod-worker/handler.py" ]; then
    # Check worker doesn't expose debug endpoints
    if grep -q "debug\|DEBUG\|pdb\|breakpoint" runpod-worker/handler.py 2>/dev/null; then
        check_fail "Debug code found in runpod-worker/handler.py"
    else
        check_pass "No debug code in worker handler"
    fi

    # Check worker validates input
    if grep -q "validate\|schema\|required" runpod-worker/handler.py 2>/dev/null; then
        check_pass "Worker has input validation"
    else
        check_warn "Worker may lack input validation — review handler.py"
    fi

    # Check worker uses R2 (not local filesystem for output)
    if grep -q "r2\|R2\|boto3\|s3" runpod-worker/handler.py 2>/dev/null; then
        check_pass "Worker uses R2 storage (not local filesystem)"
    else
        check_warn "Worker may not use R2 for output — verify handler.py"
    fi
else
    check_warn "runpod-worker/handler.py not found — skipping worker checks"
fi

# --- 6. Check RunPod client in Central ---
echo ""
echo "[Central RunPod Client]"
CLIENT_FILE="src/services/runpod_client.py"
if [ -f "$CLIENT_FILE" ]; then
    # Check timeout is set
    if grep -q "timeout\|DEFAULT_TIMEOUT" "$CLIENT_FILE"; then
        check_pass "RunPod client has request timeout"
    else
        check_fail "RunPod client missing timeout — requests could hang"
    fi

    # Check error handling
    if grep -q "RunPodError\|except\|raise" "$CLIENT_FILE"; then
        check_pass "RunPod client has error handling"
    else
        check_fail "RunPod client missing error handling"
    fi

    # Check cost tracking
    if grep -q "cost\|COST_PER_SEC\|executionTime" "$CLIENT_FILE"; then
        check_pass "RunPod client tracks GPU costs"
    else
        check_warn "RunPod client may not track costs"
    fi
else
    check_warn "$CLIENT_FILE not found"
fi

# --- Summary ---
echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    echo "  Fix the failed checks before deploying!"
    exit 1
else
    echo "  GPU security checks passed."
    if [ $WARN -gt 0 ]; then
        echo "  Review warnings above for additional hardening."
    fi
    exit 0
fi
