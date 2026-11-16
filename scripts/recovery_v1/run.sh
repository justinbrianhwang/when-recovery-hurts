#!/usr/bin/env bash
# Recovery-V1 diagnostic driver (recovery_v1_diagnostic_001): ONE pass of the frozen Vision-V1 policy without intervention on
# 100 new scenes with a per-step ground-truth trace. No training, no recovery action. PREPARED, NOT RUN.
# Usage: run.sh env|scenes|probe|budget|diag|pack
# Every phase needs runs/recovery_v1_diagnostic_001/authorization.json (written by the user side: approval, cap, plan hash).
# budget needs USD_PER_HOUR and SPENT_USD.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/recovery_v1_diagnostic_001.status
R=runs/recovery_v1_diagnostic_001
PLAN=$R/plan.json
PY=(python scripts/30_recovery_v1_diag.py)
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }

case "$1" in
  env)
    run rv1_doctor python -m hci doctor --out $R/doctor || exit 1
    run rv1_versions python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))" || exit 1
    run rv1_pytest python -m pytest -q || exit 1
    run rv1_dry "${PY[@]}" dry-run --plan $PLAN --run $R --out $R/dry_run_server.json || { echo "FILES MISSING OR CHANGED ON THE SERVER" >> $ST; exit 1; }
    echo "ENV_DONE" >> $ST ;;
  scenes)  # cube placements of the 100 diagnostic seeds against every placement on record; no policy is run
    grep -q "^ENV_DONE" $ST || { echo "scenes: environment checks have not passed" >> $ST; exit 1; }
    run rv1_scenes "${PY[@]}" scenes --plan $PLAN --run $R || { echo "PLACEMENT CHECK FAILED: no diagnostic pass on this set" >> $ST; exit 1; }
    echo "SCENES_DONE" >> $ST ;;
  probe)   # 3 already-used development scenes with the full recorder: timing, recorder check, comparison with the record
    grep -q "^SCENES_DONE" $ST || { echo "probe: the placement check has not passed" >> $ST; exit 1; }
    run rv1_probe "${PY[@]}" probe --plan $PLAN --run $R || { echo "PROBE FAILED: TECHNICAL_BLOCKED, the diagnostic set is untouched" >> $ST; exit 1; }
    echo "PROBE_DONE" >> $ST ;;
  budget)
    grep -q "^PROBE_DONE" $ST || { echo "budget: no probe" >> $ST; exit 1; }
    run rv1_budget "${PY[@]}" budget --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "BUDGET_BLOCKED: diagnostic pass not started" >> $ST; exit 1; }
    echo "BUDGET_OK" >> $ST ;;
  diag)    # the one diagnostic pass; ledger written first; never resumed
    grep -q "^BUDGET_OK" $ST || { echo "diag: the budget gate has not passed" >> $ST; exit 1; }
    mkdir -p artifacts/ledger
    run rv1_diag "${PY[@]}" run --plan $PLAN --run $R || { echo "DIAGNOSTIC PASS INCOMPLETE: the set is used; no second pass" >> $ST; exit 1; }
    run rv1_summary "${PY[@]}" summarize --diagnostic $R/diagnostic --out $R/diagnostic/summary.json || exit 1
    echo "DIAG_DONE" >> $ST ;;
  pack)
    A=/root/recovery_v1_out
    mkdir -p artifacts/ledger
    tar czf $A.tgz $R artifacts/ledger logs/rv1_* logs/recovery_v1_diagnostic_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|probe|budget|diag|pack"; exit 2 ;;
esac
