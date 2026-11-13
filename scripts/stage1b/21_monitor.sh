#!/usr/bin/env bash
# Export the EMA of every 10k full checkpoint and evaluate it on the 20 monitor seeds (descriptive only, never
# used for selection). Runs beside the trainer. Usage: 21_monitor.sh RUN DEMO_MANIFEST
source "$(dirname "$0")/common.sh"
RUN=$1; MANIFEST=$2
for tag in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000; do
  f=runs/policy/$RUN/checkpoints/full_${tag}.pt
  until [[ -f $f ]]; do
    if grep -q "^train_${RUN} exit=" $ST && [[ ! -f $f ]]; then echo "monitor_${RUN}: training ended without $f" >> $ST; exit 1; fi
    sleep 30
  done
  P=artifacts/policies/${RUN}_i${tag}
  run export_${RUN}_i${tag} python scripts/04_export_policy.py --run runs/policy/$RUN --checkpoint $f --out $P || exit 1
  run monitor_${RUN}_i${tag} python scripts/09_eval_policy.py --config $C --policy $P --set policy_monitor --demo-manifest $MANIFEST \
      --out runs/monitor_${RUN}_i${tag} || exit 1
done
echo "MONITOR_DONE ${RUN}" >> $ST
