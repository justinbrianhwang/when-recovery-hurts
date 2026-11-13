#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?Usage: bash scripts/backup_results.sh /path/outside_project/results.tar.gz}
[[ ! -e "$OUT" ]] || { echo 'Backup path exists; refusing overwrite'; exit 2; }
# Deliberately excludes raw demos, virtualenvs, .env files, credentials, and upstream sources.
items=(configs docs prompts templates src scripts pyproject.toml upstream.lock.json)
for x in runs artifacts frozen; do [[ ! -e "$x" ]] || items+=("$x"); done
tar --exclude='__pycache__' --exclude='*.h5' -czf "$OUT" "${items[@]}"
sha256sum "$OUT" > "$OUT.sha256"
echo 'This is only a local archive. Transfer it off the rented instance and verify the hash before destroying the instance.'
