#!/usr/bin/env bash
# Temporal Vision-CENTER v2 driver (vision_center_v2_001). PREPARED, NOT RUN.
# Usage: run.sh env|scenes|audit|shadow|pilot|pack
# Every phase needs runs/vision_center_v2_001/authorization.json (written by the user side: approval, cap, plan hash).
# shadow and pilot need USD_PER_HOUR and SPENT_USD; each starts with its own budget gate.
source "$(dirname "$0")/../stage1b/common.sh"
# one thread per numerical library, before every Python process (the first v1 attempt was stopped by thread oversubscription on a shared host)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
ST=logs/vision_center_v2_001.status
R=runs/vision_center_v2_001
PLAN=$R/plan.json
PY=(python scripts/33_vision_center_v2.py)
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }

case "$1" in
  env)
    run vc2_doctor python -m hci doctor --out $R/doctor || exit 1
    run vc2_versions python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))" || exit 1
    run vc2_pytest python -m pytest -q || exit 1
    run vc2_dry "${PY[@]}" dry-run --plan $PLAN --run $R --out $R/dry_run_server.json || { echo "FILES MISSING OR CHANGED ON THE SERVER" >> $ST; exit 1; }
    echo "ENV_DONE" >> $ST ;;
  scenes)  # cube placements of the 200 new seeds against every placement on record; no policy is run
    grep -q "^ENV_DONE" $ST || { echo "scenes: environment checks have not passed" >> $ST; exit 1; }
    run vc2_scenes "${PY[@]}" scenes --plan $PLAN --run $R || { echo "PLACEMENT CHECK FAILED: no pilot on this set" >> $ST; exit 1; }
    echo "SCENES_DONE" >> $ST ;;
  audit)   # 3 already-used scenes, all four arms: technical checks and timing
    grep -q "^SCENES_DONE" $ST || { echo "audit: the placement check has not passed" >> $ST; exit 1; }
    run vc2_audit "${PY[@]}" audit --plan $PLAN --run $R || { echo "AUDIT FAILED: TECHNICAL_BLOCKED, no scene set is touched" >> $ST; exit 1; }
    echo "AUDIT_DONE" >> $ST ;;
  shadow)  # stage B1: the 300 USED scenes, C0, VISION_CENTER_V1, VISION_CENTER_V2; fixed criteria decide whether the new scenes are run
    grep -q "^AUDIT_DONE" $ST || { echo "shadow: no audit" >> $ST; exit 1; }
    run vc2_budget_shadow "${PY[@]}" budget --stage shadow --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "BUDGET_BLOCKED: shadow pass not started" >> $ST; exit 1; }
    run vc2_shadow "${PY[@]}" shadow --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "V2_SHADOW_BLOCKED OR INCOMPLETE: the new scenes are not touched; nothing is tuned and re-tried" >> $ST; exit 1; }
    echo "V2_SHADOW_PASS" >> $ST ;;
  pilot)   # stage B2: the one pilot pass on the 200 new scenes; ledger written first; never resumed
    grep -q "^V2_SHADOW_PASS" $ST || { echo "pilot: the shadow pass has not passed" >> $ST; exit 1; }
    mkdir -p artifacts/ledger
    run vc2_budget_pilot "${PY[@]}" budget --stage pilot --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "BUDGET_BLOCKED: pilot not started, the set is unused" >> $ST; exit 1; }
    run vc2_pilot "${PY[@]}" run --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "PILOT PASS INCOMPLETE: the set is used; no second pass" >> $ST; exit 1; }
    echo "PILOT_DONE" >> $ST ;;
  pack)
    A=/root/vision_center_v2_out
    mkdir -p artifacts/ledger
    tar czf $A.tgz $R artifacts/ledger logs/vc2_* logs/vision_center_v2_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|audit|shadow|pilot|pack"; exit 2 ;;
esac
