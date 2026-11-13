#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:-StackCube-v1}
COUNT=${2:-20}
case "$TASK" in
  PickCube-v1|StackCube-v1) CONTROL=pd_ee_delta_pos ;;
  PegInsertionSide-v1) CONTROL=pd_ee_delta_pose ;;
  *) echo 'Only the three registered starter tasks are supported.' >&2; exit 2 ;;
esac
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'COUNT must be a positive integer'; exit 2; }
DEMO_ROOT=$(python -c 'from mani_skill import DEMO_DIR; print(DEMO_DIR)')
SRC="$DEMO_ROOT/$TASK/motionplanning/trajectory.h5"
DST="$DEMO_ROOT/$TASK/motionplanning/trajectory.rgb.$CONTROL.physx_cpu.h5"
if [[ -e "$DST" ]]; then echo "Refusing to replace existing RGB dataset: $DST" >&2; exit 2; fi
if [[ ! -f "$SRC" ]]; then python -m mani_skill.utils.download_demo "$TASK"; fi
[[ -f "$SRC" ]] || { echo "Expected demo missing: $SRC. Inspect the downloader output; do not fabricate a path."; exit 2; }
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$SRC" --use-first-env-state -c "$CONTROL" -o rgb \
  -b physx_cpu --num-envs 1 --count "$COUNT" --save-traj
[[ -f "$DST" ]] || { echo 'Replay output path differed; inspect files before training.'; exit 2; }
echo "RGB dataset: $DST"
echo 'Inspect actual replay success count. COUNT is an upper bound on requested demonstrations, not a success guarantee.'
