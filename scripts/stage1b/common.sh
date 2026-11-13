# Shared by the Stage 1B driver scripts. Every step is logged with its command, start/end time and exit code.
# A failing step is recorded and returned to the caller; nothing is retried or masked.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." && source .venv/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 WANDB_MODE=disabled
mkdir -p logs
ST=logs/stage1b_002.status
C=configs/stackcube.yaml
DEMO_DIR_STACK="$HOME/.maniskill/demos/StackCube-v1/motionplanning"
DEMO="$DEMO_DIR_STACK/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
SRC="$DEMO_DIR_STACK/trajectory.h5"
run(){
  local name=$1; shift
  if [[ -e logs/${name}.log ]]; then echo "$name REFUSED: logs/${name}.log exists (use a new step name)" >> $ST; return 3; fi
  echo "$name cmd=$*" >> $ST; echo "$name start=$(date -u +%FT%TZ)" >> $ST
  "$@" > logs/${name}.log 2>&1; local rc=$?
  echo "$name exit=$rc end=$(date -u +%FT%TZ)" >> $ST; return $rc
}
