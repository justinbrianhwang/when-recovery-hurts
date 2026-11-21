#!/usr/bin/env bash
# Vision-CENTER v2 Phase C1 driver (vision_center_v2_c1_001). PREPARED, NOT RUN.
# Usage: run.sh env|scenes|probecheck|audit|run|pack
# Every phase needs runs/vision_center_v2_c1_001/authorization.json (written by the user side: approval, total cap, plan hash).
# env needs USD_PER_HOUR (all-in hourly price of this instance) and SPENT_BEFORE_USD (spend of this work before this instance);
# it writes the session clock once. Later steps compute the spend from that clock.
source "$(dirname "$0")/../stage1b/common.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
ST=logs/vision_center_v2_c1_001.status
R=runs/vision_center_v2_c1_001
PLAN=$R/plan.json
PY=(python scripts/34_vision_center_c1.py)
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }
"${PY[@]}" check-auth --plan $PLAN --run $R >> $ST 2>&1 || { echo "REFUSED: authorization.json does not name this plan" >> $ST; exit 1; }

case "$1" in
  env)
    run c1_clock "${PY[@]}" clock --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-before-usd "${SPENT_BEFORE_USD:?}" || { echo "SESSION CLOCK NOT WRITTEN" >> $ST; exit 1; }
    run c1_doctor python -m hci doctor --out $R/doctor || exit 1
    run c1_versions python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))" || exit 1
    run c1_pipfreeze python -m pip freeze || exit 1
    run c1_pytest python -m pytest -q || exit 1
    run c1_dry "${PY[@]}" dry-run --plan $PLAN --run $R --out $R/dry_run_server.json || { echo "FILES MISSING OR CHANGED ON THE SERVER" >> $ST; exit 1; }
    echo "ENV_DONE" >> $ST ;;
  scenes)  # cube placements of the 200 new seeds against every placement on record; no policy is run
    grep -q "^ENV_DONE" $ST || { echo "scenes: environment checks have not passed" >> $ST; exit 1; }
    run c1_scenes "${PY[@]}" scenes --plan $PLAN --run $R || { echo "PLACEMENT CHECK FAILED: no run on this set" >> $ST; exit 1; }
    echo "SCENES_DONE" >> $ST ;;
  probecheck)  # logging ON/OFF on the 3 used audit scenes with train seed 1: bitwise, decides nothing
    grep -q "^SCENES_DONE" $ST || { echo "probecheck: the placement check has not passed" >> $ST; exit 1; }
    run c1_probe_check "${PY[@]}" probe-check --plan $PLAN --run $R || { echo "PROBE CHECK FAILED: TECHNICAL_BLOCKED, no new scene is touched" >> $ST; exit 1; }
    echo "PROBE_CHECK_DONE" >> $ST ;;
  audit)   # 3 already-used scenes x both policies, every arm: technical checks and timing
    grep -q "^PROBE_CHECK_DONE" $ST || { echo "audit: the logging ON/OFF check has not passed" >> $ST; exit 1; }
    run c1_audit "${PY[@]}" audit --plan $PLAN --run $R || { echo "AUDIT FAILED: TECHNICAL_BLOCKED, no new scene is touched" >> $ST; exit 1; }
    echo "AUDIT_DONE" >> $ST ;;
  run)     # budget gate, ledger, then the pass with train seed 1 and the pass with train seed 2; never resumed
    grep -q "^AUDIT_DONE" $ST || { echo "run: no audit" >> $ST; exit 1; }
    mkdir -p artifacts/ledger
    run c1_budget "${PY[@]}" budget --plan $PLAN --run $R || { echo "BUDGET_BLOCKED: run not started, the set is unused" >> $ST; exit 1; }
    run c1_run "${PY[@]}" run --plan $PLAN --run $R || { echo "RUN INCOMPLETE: the set is used; no second pass" >> $ST; exit 1; }
    echo "RUN_DONE" >> $ST ;;
  pack)
    A=/root/vision_center_v2_c1_out
    mkdir -p artifacts/ledger
    tar czf $A.tgz $R artifacts/ledger logs/c1_* logs/vision_center_v2_c1_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|probecheck|audit|run|pack"; exit 2 ;;
esac
