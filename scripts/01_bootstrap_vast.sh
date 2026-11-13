#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
"$PYTHON" - <<'CHECK'
import sys
if sys.version_info[:2] not in [(3,10),(3,11)]:
    raise SystemExit('Use a fresh Python 3.10/3.11 environment for this pinned starter, preferably 3.11.')
CHECK
if [[ -e .venv ]]; then
  echo '.venv already exists. Activate it and inspect the existing install; this script will not overwrite it.' >&2
  exit 2
fi
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/constraints-linux-py311.txt -r requirements/runtime.txt
mkdir -p third_party
REV=a4a4f9272ad64b1564035874b605ceb687b63ed8
if [[ ! -d third_party/ManiSkill ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/mani-skill/ManiSkill.git third_party/ManiSkill
  git -C third_party/ManiSkill checkout --detach "$REV"
fi
[[ $(git -C third_party/ManiSkill rev-parse HEAD) == "$REV" ]] || { echo 'Unexpected upstream commit'; exit 2; }
python -m pip install -c requirements/constraints-linux-py311.txt -e third_party/ManiSkill
python -m pip install -c requirements/constraints-linux-py311.txt -e third_party/ManiSkill/examples/baselines/diffusion_policy
python -m pip install -e '.[test]'
python -m pip check
mkdir -p artifacts/environment
python -m pip freeze > artifacts/environment/installed-pip-freeze.txt
printf '\nInstallation finished, but GPU/RGB runtime is NOT yet validated.\n'
printf 'Next: source .venv/bin/activate && python -m hci doctor --out runs/doctor_001\n'
