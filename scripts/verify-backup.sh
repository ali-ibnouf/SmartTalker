#!/bin/bash
# =============================================================================
# SmartTalker — Backup Configuration Verification
# Verifies backup script exists, is properly configured, and recent backups exist
# Usage: bash scripts/verify-backup.sh
# =============================================================================

echo "========================================="
echo "  Backup Configuration Verification"
echo "========================================="

PASS=0
FAIL=0
WARN=0
BACKUP_DIR="${BACKUP_DIR:-/opt/maskki/backups}"

check_pass() { echo "  PASS $1"; PASS=$((PASS + 1)); }
check_fail() { echo "  FAIL $1"; FAIL=$((FAIL + 1)); }
check_warn() { echo "  WARN $1"; WARN=$((WARN + 1)); }

# --- 1. Backup script exists ---
echo ""
echo "[Backup Script]"
if [ -f "scripts/backup.sh" ]; then
    check_pass "scripts/backup.sh exists"

    # Check it's executable or has bash shebang
    if head -1 scripts/backup.sh | grep -q "bash"; then
        check_pass "Has bash shebang"
    else
        check_warn "Missing bash shebang — run with: bash scripts/backup.sh"
    fi

    # Check set -e (fail on errors)
    if grep -q "set -e" scripts/backup.sh; then
        check_pass "Uses strict error handling (set -e)"
    else
        check_warn "No strict error handling — consider adding 'set -e'"
    fi
else
    check_fail "scripts/backup.sh not found"
    echo "       Create a backup script before deploying to production"
fi

# --- 2. Encryption ---
echo ""
echo "[Encryption]"
if [ -f "scripts/backup.sh" ]; then
    if grep -q "gpg\|openssl\|encrypt\|AES" scripts/backup.sh; then
        check_pass "Backup script uses encryption"

        # Check encryption algorithm
        if grep -q "AES256\|aes-256\|AES-256" scripts/backup.sh; then
            check_pass "Uses AES-256 encryption"
        else
            check_warn "Encryption found but AES-256 not explicitly specified"
        fi

        # Check passphrase is from environment (not hardcoded)
        if grep -q "BACKUP_PASSPHRASE\|ENCRYPTION_KEY" scripts/backup.sh; then
            check_pass "Encryption key loaded from environment variable"
        else
            check_warn "Encryption key source unclear — ensure it's not hardcoded"
        fi
    else
        check_fail "Backups are NOT encrypted — add gpg or openssl encryption"
    fi
else
    check_fail "Cannot verify encryption — backup script not found"
fi

# --- 3. Retention policy ---
echo ""
echo "[Retention Policy]"
if [ -f "scripts/backup.sh" ]; then
    if grep -q "find.*-mtime\|RETENTION" scripts/backup.sh; then
        check_pass "Retention policy configured"

        # Extract retention days
        RETENTION=$(grep -oP 'RETENTION_DAYS=\K\d+' scripts/backup.sh 2>/dev/null || \
                    grep -oP 'mtime \+\K\d+' scripts/backup.sh 2>/dev/null || echo "")
        if [ -n "$RETENTION" ]; then
            if [ "$RETENTION" -ge 7 ] && [ "$RETENTION" -le 90 ]; then
                check_pass "Retention period: ${RETENTION} days"
            else
                check_warn "Retention period ($RETENTION days) is outside recommended range (7-90)"
            fi
        fi
    else
        check_warn "No retention policy found — old backups may accumulate"
    fi
else
    check_fail "Cannot verify retention — backup script not found"
fi

# --- 4. PostgreSQL backup ---
echo ""
echo "[Database Backup]"
if [ -f "scripts/backup.sh" ]; then
    if grep -q "pg_dump\|pgdump\|postgres" scripts/backup.sh; then
        check_pass "PostgreSQL backup configured"
    else
        check_fail "No PostgreSQL backup — database will be lost on failure"
    fi

    if grep -q "gzip\|gz\|compress\|zstd" scripts/backup.sh; then
        check_pass "Backup compression enabled"
    else
        check_warn "No compression — backups may use excessive disk space"
    fi
else
    check_fail "Cannot verify database backup — backup script not found"
fi

# --- 5. Redis backup ---
echo ""
echo "[Redis Backup]"
if [ -f "scripts/backup.sh" ]; then
    if grep -q "redis\|BGSAVE\|dump.rdb" scripts/backup.sh; then
        check_pass "Redis backup configured"
    else
        check_warn "No Redis backup — session/cache data will be lost"
    fi
else
    check_fail "Cannot verify Redis backup — backup script not found"
fi

# --- 6. Offsite copy ---
echo ""
echo "[Offsite Storage]"
if [ -f "scripts/backup.sh" ]; then
    if grep -q "scp\|rsync\|s3\|r2\|HETZNER\|offsite\|remote" scripts/backup.sh; then
        check_pass "Offsite backup copy configured"
    else
        check_warn "No offsite copy — all backups are local only"
        echo "       Consider adding scp/rsync to a remote storage box"
    fi
else
    check_fail "Cannot verify offsite — backup script not found"
fi

# --- 7. Cron schedule ---
echo ""
echo "[Cron Schedule]"
if command -v crontab > /dev/null 2>&1; then
    if crontab -l 2>/dev/null | grep -q "backup"; then
        SCHEDULE=$(crontab -l 2>/dev/null | grep "backup" | head -1)
        check_pass "Backup cron job found"
        echo "       Schedule: $SCHEDULE"
    else
        check_warn "No backup cron job — backups are manual only"
        echo "       Add: 0 3 * * * cd /opt/maskki/SmartTalker && bash scripts/backup.sh"
    fi
else
    check_warn "crontab not available — cannot verify schedule"
fi

# --- 8. Recent backup existence ---
echo ""
echo "[Recent Backups]"
if [ -d "$BACKUP_DIR" ]; then
    BACKUP_COUNT=$(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$BACKUP_COUNT" -gt 0 ]; then
        check_pass "$BACKUP_COUNT backup(s) found in $BACKUP_DIR"

        # Check most recent backup age
        LATEST=$(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | \
                 sort -n -r | head -1 | cut -d' ' -f2)

        if [ -n "$LATEST" ]; then
            LATEST_NAME=$(basename "$LATEST")
            echo "       Latest: $LATEST_NAME"

            # Check if backup is less than 48 hours old
            LATEST_AGE_HOURS=$(( ($(date +%s) - $(stat -c%Y "$LATEST" 2>/dev/null || echo 0)) / 3600 ))
            if [ "$LATEST_AGE_HOURS" -lt 48 ]; then
                check_pass "Latest backup is ${LATEST_AGE_HOURS}h old (within 48h)"
            else
                check_warn "Latest backup is ${LATEST_AGE_HOURS}h old (older than 48h)"
            fi

            # Verify encrypted files exist
            GPG_COUNT=$(find "$LATEST" -name "*.gpg" 2>/dev/null | wc -l)
            if [ "$GPG_COUNT" -gt 0 ]; then
                check_pass "Latest backup has $GPG_COUNT encrypted files"
            else
                check_warn "Latest backup has no encrypted (.gpg) files"
            fi

            # Verify manifest exists
            if [ -f "$LATEST/manifest.json" ]; then
                check_pass "Backup manifest found"
            else
                check_warn "No manifest.json in latest backup"
            fi
        fi
    else
        check_warn "No backups found in $BACKUP_DIR — run backup.sh first"
    fi
else
    check_warn "$BACKUP_DIR does not exist — no backups taken yet"
fi

# --- 9. Docker volume backup ---
echo ""
echo "[Docker Volumes]"
if command -v docker > /dev/null 2>&1; then
    PG_VOLUME=$(docker volume ls -q 2>/dev/null | grep -c "postgres-data" || echo "0")
    REDIS_VOLUME=$(docker volume ls -q 2>/dev/null | grep -c "redis-data" || echo "0")

    if [ "$PG_VOLUME" -gt 0 ]; then
        check_pass "PostgreSQL data volume exists"
    else
        check_warn "PostgreSQL data volume not found (service may not be running)"
    fi

    if [ "$REDIS_VOLUME" -gt 0 ]; then
        check_pass "Redis data volume exists"
    else
        check_warn "Redis data volume not found (service may not be running)"
    fi
else
    check_warn "Docker not available — cannot verify volumes"
fi

# --- Summary ---
echo ""
echo "========================================="
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    echo "  Fix the failed checks before going live!"
    exit 1
else
    echo "  Backup configuration verified."
    if [ $WARN -gt 0 ]; then
        echo "  Review warnings above for additional hardening."
    fi
    exit 0
fi
