#!/usr/bin/env bash
# Prints commands only. No training, cloud rental, or final Test is launched here.
set -euo pipefail
cat <<'COMMANDS'
# Execute from the extracted repository root, after activating .venv.
python -m hci doctor --out runs/doctor_001
python -m hci smoke --config configs/pickcube.yaml --out runs/smoke_pick_001
python -m hci smoke --config configs/stackcube.yaml --out runs/smoke_stack_001
bash scripts/02_prepare_demos.sh StackCube-v1 20
# Follow QUICKSTART.md to train and export a policy, then:
python -m hci audit --config configs/stackcube.yaml --policy artifacts/policies/stack_s1 --out runs/audit_stack_001
python -m hci collect --config configs/stackcube.yaml --policy artifacts/policies/stack_s1 --audit runs/audit_stack_001/audit.json --episodes 2 --out runs/branches_smoke_001
# STOP: inspect state replay and actual branch behavior before increasing episodes.
COMMANDS
