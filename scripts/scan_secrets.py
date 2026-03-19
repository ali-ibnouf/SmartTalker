"""
Scan entire codebase and git history for leaked secrets.
Run this BEFORE every deployment.
"""
import subprocess, re, sys, os

# Patterns to detect
SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{20,}', 'DashScope API Key'),
    (r'rpa-[a-zA-Z0-9]{20,}', 'RunPod API Key'),
    (r'rp_[a-zA-Z0-9]{20,}', 'RunPod API Key'),
    (r're_[a-zA-Z0-9]{20,}', 'Resend API Key'),
    (r'pdl_[a-zA-Z0-9]{20,}', 'Paddle Key'),
    (r'[0-9a-f]{64}', 'Possible Secret Key (64 hex)'),
    (r'AKIA[0-9A-Z]{16}', 'AWS/R2 Access Key'),
    (r'password\s*=\s*["\'][^"\']{3,}', 'Hardcoded Password'),
    (r'secret\s*=\s*["\'][^"\']{3,}', 'Hardcoded Secret'),
]

# Files/dirs to skip
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '_archive_RenderNode', 'venv', '.venv', 'models'}
SKIP_FILES = {'.env.production.example', 'scan_secrets.py', 'INCIDENT_LOG.md'}
# Paths containing test fixtures (fake secrets are expected)
SKIP_PATH_PARTS = {'tests', 'test_'}

def scan_file(filepath):
    """Scan a single file for secrets."""
    is_test_file = any(part in filepath for part in ('tests', 'test_'))
    findings = []
    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Skip if it's a variable reference, URL hash, or placeholder
                        lowline = line.lower()
                        if any(safe in lowline for safe in [
                            'environ', 'getenv', 'config.', '___', 'your_',
                            '<your', 'example', 'placeholder', 'xxx',
                            'http://', 'https://', 'test_secret', 'testpass',
                        ]):
                            continue
                        # Skip test fixtures (fake creds are expected)
                        if is_test_file:
                            continue
                        findings.append({
                            'file': filepath,
                            'line': line_num,
                            'type': name,
                            'content': line.strip()[:100],
                        })
    except Exception:
        pass
    return findings

def scan_codebase():
    """Walk entire project directory."""
    findings = []
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_FILES:
                continue
            if f.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.yml', '.yaml', '.env', '.md', '.sh', '.conf')):
                filepath = os.path.join(root, f)
                findings.extend(scan_file(filepath))
    return findings

def scan_git_history():
    """Scan git log for leaked secrets (last 100 commits)."""
    findings = []
    try:
        result = subprocess.run(
            ['git', 'log', '-100', '-p', '--diff-filter=A'],
            capture_output=True, text=True, timeout=60,
            encoding='utf-8', errors='ignore'
        )
        for pattern, name in SECRET_PATTERNS:
            matches = re.finditer(pattern, result.stdout)
            for match in matches:
                context = result.stdout[max(0, match.start()-50):match.end()+50]
                if not any(safe in context.lower() for safe in ['environ', 'getenv', 'example', 'placeholder', 'xxx']):
                    findings.append({
                        'source': 'git_history',
                        'type': name,
                        'content': match.group()[:20] + '...',
                    })
    except Exception as e:
        print(f"Git scan error: {e}")
    return findings

if __name__ == '__main__':
    print("=" * 60)
    print("SECRET SCANNER — Pre-Deployment Check")
    print("=" * 60)

    print("\n[1/2] Scanning codebase...")
    code_findings = scan_codebase()

    print("[2/2] Scanning git history...")
    git_findings = scan_git_history()

    all_findings = code_findings + git_findings

    if all_findings:
        print(f"\n FOUND {len(all_findings)} POTENTIAL SECRETS:\n")
        for f in all_findings:
            if 'file' in f:
                print(f"  WARNING  {f['type']} in {f['file']}:{f['line']}")
                print(f"      {f['content']}")
            else:
                print(f"  WARNING  {f['type']} in {f['source']}")
                print(f"      {f['content']}")
            print()
        print("DO NOT DEPLOY until all secrets are removed!")
        sys.exit(1)
    else:
        print("\nNo secrets found. Safe to deploy.")
        sys.exit(0)
