#!/usr/bin/env bash
# Function check with the qualified policy and the new retreat: restore audit first, then (only if it passes)
# the branches of two source episodes, plus three development videos. Not a pilot collection.
# Usage: 40_post.sh POLICY_DIR SUFFIX
source "$(dirname "$0")/common.sh"
P=$1; X=$2
run audit_stack_${X} python -m hci audit --config $C --policy $P --out runs/audit_stack_${X} || { echo "AUDIT FAILED: no collection" >> $ST; exit 1; }
run branches_smoke_${X} python -m hci collect --config $C --policy $P --audit runs/audit_stack_${X}/audit.json --episodes 2 --out runs/branches_smoke_${X} || exit 1
run inspect_branches_${X} python scripts/inspect_branches.py --data runs/branches_smoke_${X} || exit 1
run video_continue_${X} python scripts/record_episode.py --config $C --policy $P --seed 20000 --action continue --out runs/video_continue_${X} || exit 1
run video_refresh_${X} python scripts/record_episode.py --config $C --policy $P --seed 20000 --action refresh --at 12 --out runs/video_refresh_${X} || exit 1
run video_retreat_${X} python scripts/record_episode.py --config $C --policy $P --seed 20000 --action retreat_retry --at 12 --out runs/video_retreat_${X} || exit 1
echo "POST_DONE ${X}" >> $ST
