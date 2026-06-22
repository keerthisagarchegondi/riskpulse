#!/usr/bin/env bash
# Backup PostgreSQL database
set -euo pipefail

DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-riskpulse}"
DB_USER="${POSTGRES_USER:-riskpulse}"
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "Backing up ${DB_NAME} to ${BACKUP_DIR}/..."
PGPASSWORD="${POSTGRES_PASSWORD:-riskpulse_dev_password}" pg_dump \
    -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
    --format=custom --compress=9 \
    -f "${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.dump"

echo "Backup complete: ${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.dump"
