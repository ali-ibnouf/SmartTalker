#!/bin/bash
echo "Installing pre-commit secret detection hook..."

cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash
# Pre-commit hook: block commits containing secrets

echo "Checking for secrets in staged files..."

# Patterns that should NEVER be in code
PATTERNS=(
    'sk-[a-zA-Z0-9]{20,}'
    'rpa-[a-zA-Z0-9]{20,}'
    'rp_[a-zA-Z0-9]{20,}'
    're_[a-zA-Z0-9]{20,}'
    'AKIA[0-9A-Z]{16}'
    'pdl_[a-zA-Z0-9]{20,}'
)

# Get staged files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|js|ts|tsx|jsx|json|yml|yaml|env|sh|conf|md)$')

if [ -z "$FILES" ]; then
    exit 0
fi

FOUND=0
for pattern in "${PATTERNS[@]}"; do
    MATCHES=$(echo "$FILES" | xargs grep -l -E "$pattern" 2>/dev/null | grep -v '.env.example' | grep -v 'scan_secrets.py' | grep -v 'pre-commit')
    if [ -n "$MATCHES" ]; then
        echo "BLOCKED: Possible secret ($pattern) found in:"
        echo "$MATCHES"
        FOUND=1
    fi
done

# Check for .env files being committed
ENV_FILES=$(git diff --cached --name-only | grep -E '^\.env$|\.env\.production$|\.env\.local$|\.env\.staging$')
if [ -n "$ENV_FILES" ]; then
    echo "BLOCKED: Attempting to commit .env file: $ENV_FILES"
    FOUND=1
fi

if [ $FOUND -eq 1 ]; then
    echo ""
    echo "Commit blocked. Remove secrets before committing."
    echo "   If this is a false positive, use: git commit --no-verify"
    exit 1
fi

echo "No secrets detected."
exit 0
HOOK

chmod +x .git/hooks/pre-commit
echo "Pre-commit hook installed."
