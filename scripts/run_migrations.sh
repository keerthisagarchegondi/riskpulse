#!/usr/bin/env bash
# Run PostgreSQL migrations in order
set -euo pipefail

DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-riskpulse}"
DB_USER="${POSTGRES_USER:-riskpulse}"

echo "Running migrations against ${DB_HOST}:${DB_PORT}/${DB_NAME}..."

for migration in database/migrations/*.sql; do
    echo "  Applying: $(basename "$migration")"
    PGPASSWORD="${POSTGRES_PASSWORD:-riskpulse_dev_password}" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -f "$migration" -v ON_ERROR_STOP=1
done

echo "All migrations applied successfully."
