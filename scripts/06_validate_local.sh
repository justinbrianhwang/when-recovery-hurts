#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-validation/local}
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT"; exit 2; }
mkdir -p "$OUT"
python -m compileall -q src scripts tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --junitxml="$OUT/pytest.xml" | tee "$OUT/pytest.txt"
python -m hci --help > "$OUT/cli-help.txt"
python -m hci sample-size --epsilon .05 --delta .05 --candidates 19 > "$OUT/sample-size-example.json"
echo 'These are CPU/core/mock tests. They do not validate ManiSkill physics, RGB rendering, CUDA or policy success.'
