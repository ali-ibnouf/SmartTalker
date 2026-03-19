"""
Verify that critical config files have not been tampered with.
Run after deployment or on a schedule to detect unauthorized changes.

Usage:
    # First time — generate baseline:
    python scripts/verify-config-integrity.py --generate

    # Subsequent runs — verify against baseline:
    python scripts/verify-config-integrity.py
"""

import hashlib
import json
import os
import sys

BASELINE_FILE = os.path.join(os.path.dirname(__file__), ".config-hashes.json")

# Critical files to monitor
CONFIG_FILES = [
    "nginx.prod.conf",
    "docker-compose.prod.yml",
    "scripts/harden-server.sh",
    "scripts/deploy.sh",
    ".env.production.example",
    "scripts/install-hooks.sh",
    "scripts/verify-security.sh",
    "scripts/verify-cloudflare-waf.sh",
    "scripts/backup.sh",
    "scripts/verify-backup.sh",
    "scripts/harden-gpu-node.sh",
    "Dockerfile",
]


def hash_file(filepath: str) -> str | None:
    """Compute SHA-256 hash of a file."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def generate_baseline():
    """Generate baseline hashes for all config files."""
    hashes = {}
    for filepath in CONFIG_FILES:
        h = hash_file(filepath)
        if h:
            hashes[filepath] = h
            print(f"  {filepath}: {h[:16]}...")
        else:
            print(f"  {filepath}: NOT FOUND (skipped)")

    with open(BASELINE_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"\nBaseline saved to {BASELINE_FILE}")
    print(f"  {len(hashes)} files recorded.")


def verify():
    """Verify current hashes against baseline."""
    if not os.path.exists(BASELINE_FILE):
        print("No baseline found. Run with --generate first:")
        print("  python scripts/verify-config-integrity.py --generate")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    modified = []
    missing = []
    ok = 0

    for filepath, expected_hash in baseline.items():
        actual = hash_file(filepath)
        if actual is None:
            missing.append(filepath)
            print(f"  MISSING  {filepath}")
        elif actual != expected_hash:
            modified.append(filepath)
            print(f"  CHANGED  {filepath}")
            print(f"           expected: {expected_hash[:16]}...")
            print(f"           actual:   {actual[:16]}...")
        else:
            ok += 1
            print(f"  OK       {filepath}")

    print(f"\nResults: {ok} ok, {len(modified)} changed, {len(missing)} missing")

    if modified or missing:
        print("\nConfig files have been modified!")
        print("If intentional, re-run with --generate to update baseline.")
        sys.exit(1)
    else:
        print("\nAll config files match baseline.")
        sys.exit(0)


if __name__ == "__main__":
    print("=========================================")
    print("  Config Integrity Verification")
    print("=========================================\n")

    if "--generate" in sys.argv:
        print("[Generating baseline hashes]\n")
        generate_baseline()
    else:
        print("[Verifying against baseline]\n")
        verify()
