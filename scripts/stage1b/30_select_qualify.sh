#!/usr/bin/env bash
# Pre-declared candidates (iteration tags 60000, 80000, 100000) on the same 50 selection seeds, then ONE
# qualification of the selected candidate on an untouched 100-seed set.
# Usage: 30_select_qualify.sh RUN DEMO_MANIFEST QUALIFY_SET
source "$(dirname "$0")/common.sh"
RUN=$1; MANIFEST=$2; QSET=$3; pids=()
for tag in 060000 080000 100000; do
  [[ -f artifacts/policies/${RUN}_i${tag}/weights.pt ]] || { echo "select_${RUN}: missing exported candidate $tag" >> $ST; exit 1; }
  run select_${RUN}_i${tag} python scripts/09_eval_policy.py --config $C --policy artifacts/policies/${RUN}_i${tag} --set policy_select \
      --success-at-end --demo-manifest $MANIFEST --out runs/select_${RUN}_i${tag} & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail -eq 0 ]] || { echo "select_${RUN}: a candidate evaluation failed" >> $ST; exit 1; }
run choose_${RUN} python scripts/11_select_candidate.py --candidate 60000=runs/select_${RUN}_i060000 --candidate 80000=runs/select_${RUN}_i080000 \
    --candidate 100000=runs/select_${RUN}_i100000 --out runs/select_${RUN}/selection.json || exit 1
SEL=$(python -c "import json;print(json.load(open('runs/select_${RUN}/selection.json'))['selected_policy'])")
run qualify_${RUN} python scripts/09_eval_policy.py --config $C --policy $SEL --set $QSET --success-at-end --demo-manifest $MANIFEST \
    --out runs/qualify_${RUN} || exit 1
echo "QUALIFY_DONE ${RUN} ${SEL}" >> $ST
