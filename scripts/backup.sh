#!/bin/bash
# =============================================================================
# SmartTalker — Automated Production Backup
# Backs up PostgreSQL, Redis, and critical config files
# Usage: BACKUP_PASSPHRASE=xxx bash scripts/backup.sh
#
# Recommended cron (daily at 3 AM):
#   0 3 * * * cd /opt/maskki/SmartTalker && BACKUP_PASSPHRASE=xxx bash scripts/backup.sh >> /var/log/maskki-backup.log 2>&1
# =============================================================================
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/opt/maskki/backups}"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BACKUP]${NC} $(date +%H:%M:%S) $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $(date +%H:%M:%S) $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $(date +%H:%M:%S) $*"; exit 1; }

echo "========================================="
echo "  SmartTalker Production Backup"
echo "  $(date)"
echo "========================================="

# --- 1. Verify prerequisites ---
if [ -z "${BACKUP_PASSPHRASE:-}" ]; then
    fail "BACKUP_PASSPHRASE not set. Required for encryption."
fi

if [ ! -f "$ENV_FILE" ]; then
    # Try loading from env
    if [ -z "${DB_PASSWORD:-}" ]; then
        fail "No $ENV_FILE and DB_PASSWORD not set"
    fi
else
    set -a
    source "$ENV_FILE"
    set +a
fi

mkdir -p "$BACKUP_DIR"
BACKUP_SUBDIR="$BACKUP_DIR/$DATE"
mkdir -p "$BACKUP_SUBDIR"
log "Backup directory: $BACKUP_SUBDIR"

PASS=0
FAIL_COUNT=0

record_pass() { log "  OK $1"; PASS=$((PASS + 1)); }
record_fail() { warn "  FAIL $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

# --- 2. PostgreSQL backup ---
echo ""
log "[PostgreSQL Backup]"
PG_DUMP_FILE="$BACKUP_SUBDIR/postgres_${DATE}.sql.gz"

if docker compose -f "$COMPOSE_FILE" exec -T postgres pg_dump \
    -U smarttalker \
    -d smarttalker \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    2>/dev/null | gzip > "$PG_DUMP_FILE"; then

    PG_SIZE=$(du -sh "$PG_DUMP_FILE" 2>/dev/null | cut -f1)
    record_pass "PostgreSQL dump ($PG_SIZE)"
else
    record_fail "PostgreSQL dump failed"
    rm -f "$PG_DUMP_FILE"
fi

# --- 3. Redis backup ---
echo ""
log "[Redis Backup]"
REDIS_DUMP_FILE="$BACKUP_SUBDIR/redis_${DATE}.rdb"

# Trigger BGSAVE and wait for completion
if docker compose -f "$COMPOSE_FILE" exec -T redis \
    redis-cli -a "${REDIS_PASSWORD}" BGSAVE 2>/dev/null; then

    sleep 5  # Wait for background save

    if docker compose -f "$COMPOSE_FILE" cp \
        redis:/data/dump.rdb "$REDIS_DUMP_FILE" 2>/dev/null; then
        REDIS_SIZE=$(du -sh "$REDIS_DUMP_FILE" 2>/dev/null | cut -f1)
        record_pass "Redis snapshot ($REDIS_SIZE)"
    else
        record_fail "Redis snapshot copy failed"
        rm -f "$REDIS_DUMP_FILE"
    fi
else
    record_fail "Redis BGSAVE failed"
fi

# --- 4. Config files backup ---
echo ""
log "[Config Files Backup]"
CONFIG_TAR="$BACKUP_SUBDIR/config_${DATE}.tar.gz"

CONFIG_FILES=(
    "nginx.prod.conf"
    "docker-compose.prod.yml"
    "Dockerfile"
    "Dockerfile.agent"
    ".env.production.example"
    "scripts/deploy.sh"
    "scripts/harden-server.sh"
    "scripts/install-hooks.sh"
)

EXISTING_FILES=()
for cf in "${CONFIG_FILES[@]}"; do
    if [ -f "$cf" ]; then
        EXISTING_FILES+=("$cf")
    fi
done

if [ ${#EXISTING_FILES[@]} -gt 0 ]; then
    if tar -czf "$CONFIG_TAR" "${EXISTING_FILES[@]}" 2>/dev/null; then
        record_pass "Config archive (${#EXISTING_FILES[@]} files)"
    else
        record_fail "Config archive creation failed"
    fi
else
    record_fail "No config files found to back up"
fi

# --- 5. Encrypt all backup files ---
echo ""
log "[Encryption]"
ENCRYPTED_COUNT=0

for backup_file in "$BACKUP_SUBDIR"/*; do
    if [ -f "$backup_file" ] && [[ "$backup_file" != *.gpg ]]; then
        if gpg --batch --yes --symmetric \
            --cipher-algo AES256 \
            --passphrase "$BACKUP_PASSPHRASE" \
            --output "${backup_file}.gpg" \
            "$backup_file" 2>/dev/null; then
            rm -f "$backup_file"
            ENCRYPTED_COUNT=$((ENCRYPTED_COUNT + 1))
        else
            record_fail "Encryption failed for $(basename "$backup_file")"
        fi
    fi
done

if [ $ENCRYPTED_COUNT -gt 0 ]; then
    record_pass "Encrypted $ENCRYPTED_COUNT files (AES-256)"
else
    record_fail "No files encrypted"
fi

# --- 6. Create manifest ---
echo ""
log "[Manifest]"
MANIFEST="$BACKUP_SUBDIR/manifest.json"
cat > "$MANIFEST" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "backup_dir": "$BACKUP_SUBDIR",
  "retention_days": $RETENTION_DAYS,
  "files": [
$(ls -1 "$BACKUP_SUBDIR"/*.gpg 2>/dev/null | while read f; do
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
    hash=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$f" 2>/dev/null | cut -d' ' -f1)
    echo "    {\"name\": \"$(basename "$f")\", \"size\": $size, \"sha256\": \"${hash:0:16}...\"},"
done | sed '$ s/,$//')
  ]
}
EOF
record_pass "Manifest created"

# --- 7. Offsite copy (optional — Hetzner Storage Box) ---
echo ""
log "[Offsite Copy]"
if [ -n "${HETZNER_STORAGE_USER:-}" ] && [ -n "${HETZNER_STORAGE_HOST:-}" ]; then
    REMOTE_DIR="/home/${HETZNER_STORAGE_USER}/backups/smarttalker/${DATE}"
    if ssh -o StrictHostKeyChecking=accept-new \
        -p "${HETZNER_STORAGE_PORT:-23}" \
        "${HETZNER_STORAGE_USER}@${HETZNER_STORAGE_HOST}" \
        "mkdir -p $REMOTE_DIR" 2>/dev/null; then

        if scp -P "${HETZNER_STORAGE_PORT:-23}" \
            "$BACKUP_SUBDIR"/* \
            "${HETZNER_STORAGE_USER}@${HETZNER_STORAGE_HOST}:${REMOTE_DIR}/" 2>/dev/null; then
            record_pass "Offsite copy to Hetzner Storage Box"
        else
            record_fail "Offsite SCP transfer failed"
        fi
    else
        record_fail "Cannot connect to Hetzner Storage Box"
    fi
else
    warn "  SKIP Offsite copy — HETZNER_STORAGE_USER/HOST not set"
fi

# --- 8. Retention policy: delete backups older than 30 days ---
echo ""
log "[Retention Policy]"
OLD_BACKUPS=$(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +${RETENTION_DAYS} 2>/dev/null | wc -l)
if [ "$OLD_BACKUPS" -gt 0 ]; then
    find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} +
    record_pass "Cleaned $OLD_BACKUPS backups older than ${RETENTION_DAYS} days"
else
    log "  No backups older than ${RETENTION_DAYS} days to clean"
fi

# --- 9. Calculate total backup size ---
TOTAL_SIZE=$(du -sh "$BACKUP_SUBDIR" 2>/dev/null | cut -f1)

# --- Summary ---
echo ""
echo "========================================="
echo "  Backup complete: $PASS ok, $FAIL_COUNT failed"
echo "  Total size: $TOTAL_SIZE"
echo "  Location: $BACKUP_SUBDIR"
echo "========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    fail "Some backup steps failed — review output above"
else
    log "All backup steps completed successfully."
    exit 0
fi
