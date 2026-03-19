#!/bin/bash
# Verify Cloudflare WAF is active before pointing DNS

echo "========================================="
echo "  Cloudflare WAF Verification"
echo "========================================="

PASS=0
FAIL=0

# 1. Check proxy is active (orange cloud — Cloudflare IPs)
echo ""
echo "[DNS Proxy]"
RESOLVED=$(dig +short ws.maskki.com 2>/dev/null)
PROXY_STATUS=$(echo "$RESOLVED" | grep -c '^104\.16\|^172\.64\|^104\.17\|^104\.18\|^104\.19\|^104\.20\|^104\.21\|^104\.22\|^104\.23\|^104\.24\|^104\.25\|^104\.26\|^104\.27\|^188\.114\|^141\.101')
if [ "$PROXY_STATUS" -gt 0 ]; then
    echo "  PASS Cloudflare proxy is active (orange cloud)"
    PASS=$((PASS + 1))
else
    echo "  FAIL Cloudflare proxy is NOT active! DNS points directly to server IP"
    echo "       Resolved to: $RESOLVED"
    echo "       Fix: Enable proxy (orange cloud) in Cloudflare DNS"
    FAIL=$((FAIL + 1))
fi

# 2. Check WAF rules via API (requires ZONE_ID + CLOUDFLARE_TOKEN)
echo ""
echo "[WAF Rules]"
if [ -z "$ZONE_ID" ] || [ -z "$CLOUDFLARE_TOKEN" ]; then
    echo "  SKIP ZONE_ID or CLOUDFLARE_TOKEN not set — cannot check WAF rules"
    echo "       Set: export ZONE_ID=... CLOUDFLARE_TOKEN=..."
else
    WAF_RESPONSE=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/firewall/rules" \
        -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
        -H "Content-Type: application/json")

    WAF_COUNT=$(echo "$WAF_RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('result',[])))" 2>/dev/null || echo "0")

    if [ "$WAF_COUNT" -gt 0 ]; then
        echo "  PASS $WAF_COUNT WAF rules configured"
        PASS=$((PASS + 1))
        echo "$WAF_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('result', []):
    status = 'active' if r.get('paused') == False else 'paused'
    print(f\"       - {r.get('description','(no name)')}: {status}\")
" 2>/dev/null
    else
        echo "  FAIL No WAF rules found — add rules in Cloudflare dashboard"
        FAIL=$((FAIL + 1))
    fi

    # 3. Check SSL mode
    SSL_RESPONSE=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/settings/ssl" \
        -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
        -H "Content-Type: application/json")

    SSL_MODE=$(echo "$SSL_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('value','unknown'))" 2>/dev/null || echo "unknown")

    if [ "$SSL_MODE" = "full" ] || [ "$SSL_MODE" = "strict" ]; then
        echo "  PASS SSL mode: $SSL_MODE"
        PASS=$((PASS + 1))
    else
        echo "  FAIL SSL mode is '$SSL_MODE' — should be 'full' or 'strict'"
        FAIL=$((FAIL + 1))
    fi

    # 4. Check Bot Fight Mode
    BOT_RESPONSE=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/bot_management" \
        -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
        -H "Content-Type: application/json")

    BOT_MODE=$(echo "$BOT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('fight_mode', False))" 2>/dev/null || echo "False")

    if [ "$BOT_MODE" = "True" ]; then
        echo "  PASS Bot Fight Mode enabled"
        PASS=$((PASS + 1))
    else
        echo "  WARN Bot Fight Mode not enabled — enable in Cloudflare Security > Bots"
    fi
fi

echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    echo "  Fix the failed checks before going live!"
    exit 1
else
    echo "  Cloudflare WAF verified."
    exit 0
fi
