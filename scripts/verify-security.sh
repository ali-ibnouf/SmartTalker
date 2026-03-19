#!/bin/bash
# Run AFTER deployment to verify security posture

echo "========================================="
echo "  Maskki Security Verification"
echo "========================================="

PASS=0
FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  PASS $1"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $1"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "[Files & Secrets]"
check "No .env in git" "! git ls-files --error-unmatch .env 2>/dev/null"
check "No .env.production in git" "! git ls-files --error-unmatch .env.production 2>/dev/null"
check ".env in .gitignore" "grep -q '^\.env$' .gitignore"
check ".env.production in .gitignore" "grep -q '\.env\.production' .gitignore"
check "No hardcoded API keys in src/" "! grep -rn 'sk-[a-zA-Z0-9]\{20,\}' src/ --include='*.py' | grep -v environ | grep -v example"
check "Pre-commit hook exists" "test -x .git/hooks/pre-commit"

echo ""
echo "[Docker Security]"
check "docker-compose.prod.yml exists" "test -f docker-compose.prod.yml"
check "No hardcoded passwords in docker-compose" "! grep -i 'password.*=' docker-compose.prod.yml | grep -v '\$'"
check "Uses .env.production" "grep -q 'env.production' docker-compose.prod.yml"

echo ""
echo "[Nginx Security]"
check "nginx.prod.conf exists" "test -f nginx.prod.conf"
check "TLS 1.3 only" "grep -q 'TLSv1.3' nginx.prod.conf"
check "HSTS header" "grep -q 'Strict-Transport-Security' nginx.prod.conf"
check "X-Frame-Options" "grep -q 'X-Frame-Options' nginx.prod.conf"
check "Direct IP blocked" "grep -q 'return 444' nginx.prod.conf"
check "Block .git access" "grep -q '\.git' nginx.prod.conf"

echo ""
echo "[Application Security]"
check "SSRF protection exists" "grep -rq 'validate_tool_url\|BLOCKED_HOSTS' src/ --include='*.py'"
check "Rate limiter exists" "grep -rq 'RateLimiter\|rate_limit' src/ --include='*.py'"
check "Guardrails engine exists" "grep -rq 'GuardrailsEngine\|check_response' src/ --include='*.py'"
check "JWT auth exists" "grep -rq 'verify_jwt\|jwt_secret' src/ --include='*.py'"
check "CORS configured" "grep -rq 'ALLOWED_ORIGINS\|allow_origins' src/ --include='*.py'"


echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    echo "  Fix the failed checks before deploying!"
    exit 1
else
    echo "  All security checks passed. Safe to deploy."
    exit 0
fi
