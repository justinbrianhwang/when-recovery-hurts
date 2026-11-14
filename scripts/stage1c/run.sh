#!/usr/bin/env bash
# Stage 1C driver (diagnosis only; no training, no qualification). Usage: run.sh probe|main|d5b|pack [SUFFIX]
# SUFFIX gives a failed phase a new step/run name on the next attempt; earlier logs and outputs are kept.
# Every step is logged with command, start/end time and exit code in logs/stage1c_001.status. Nothing is retried.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/stage1c_001.status
R=runs/stage1c_001
M=$R/manifest/diagnostic_manifest.json
P=artifacts/policies/stack_1b_s1_d100_i060000
CK=runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt
D=data_backup/demos_stack_002/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
S=data_backup/demos_stack_002/trajectory.h5
X=${2:-}
[[ -f $M ]] || { echo "manifest missing: $M" >> $ST; exit 1; }
case "$1" in
  probe)  # small bundle first: throughput, memory, and that every script runs end to end (outputs are not results)
    run s1c_doctor${X} python -m hci doctor --out $R/doctor${X} || exit 1
    run s1c_pytest${X} python -m pytest -q || exit 1
    run s1c_probe_rollouts${X} python scripts/14_scene_rollouts.py --config $C --manifest $M --demo $D --source $S --policy $P --limit 1 --out $R/probe_rollouts${X} || exit 1
    run s1c_probe_offline${X} python scripts/15_offline_action_diag.py --config $C --manifest $M --demo $D --policy $P --replay-obs $R/probe_rollouts${X}/replay_obs \
        --raw-checkpoint $CK --limit 1 --out $R/probe_offline${X} || exit 1
    echo "PROBE_DONE" >> $ST ;;
  main)
    run s1c_io_parity_60k${X} python scripts/08_policy_io_audit.py --config $C --policy $P --checkpoint $CK --demo $D --out $R/io_parity_60k${X} \
        || { echo "IO PARITY FAILED: later diagnostics not started" >> $ST; exit 1; }
    run s1c_scene_rollouts${X} python scripts/14_scene_rollouts.py --config $C --manifest $M --demo $D --source $S --policy $P --out $R/scene_rollouts${X} || exit 1
    run s1c_offline${X} python scripts/15_offline_action_diag.py --config $C --manifest $M --demo $D --policy $P --replay-obs $R/scene_rollouts${X}/replay_obs \
        --out $R/offline${X} || exit 1
    echo "MAIN_DONE" >> $ST ;;
  d5b)   # optional auxiliary diagnostic B: non-EMA weights of the same checkpoint, same histories and noise
    run s1c_d5b_raw_vs_ema${X} python scripts/15_offline_action_diag.py --config $C --manifest $M --demo $D --policy $P --raw-checkpoint $CK --out $R/d5b_raw_vs_ema${X} || exit 1
    echo "D5B_DONE" >> $ST ;;
  pack)  # new outputs only; the 2.46 GB Stage 1B backup is not re-archived. The archive hash is checked after download and the
         # per-file list is checked again after extraction on the receiving side.
    tar czf /root/stage1c_out.tgz $R logs/s1c_*.log logs/stage1c_001.status || exit 1
    tar tzf /root/stage1c_out.tgz | grep -v '/$' | sort | xargs sha256sum > /root/stage1c_out.sha256 || exit 1
    sha256sum /root/stage1c_out.tgz; wc -l < /root/stage1c_out.sha256 ;;
  *) echo "usage: run.sh probe|main|d5b|pack"; exit 2 ;;
esac
