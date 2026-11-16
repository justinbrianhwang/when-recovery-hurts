#!/usr/bin/env bash
# Recovery-V1 oracle XY pilot driver (recovery_v1_oracle_xy_pilot_001): four arms (C0, R5, HOLD_MATCHED, CENTER_ORACLE) from the
# state just before the frozen Vision-V1 policy's first pending release command, on 100 new scenes. PREPARED, NOT RUN.
# Usage: run.sh env|scenes|audit|budget|pilot|pack
# Every phase needs runs/recovery_v1_oracle_xy_pilot_001/authorization.json (written by the user side: approval, cap, plan hash).
# budget and pilot need USD_PER_HOUR and SPENT_USD.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/recovery_v1_oracle_xy_pilot_001.status
R=runs/recovery_v1_oracle_xy_pilot_001
PLAN=$R/plan.json
PY=(python scripts/31_recovery_v1_oracle_xy.py)
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }

case "$1" in
  env)
    run oxy_doctor python -m hci doctor --out $R/doctor || exit 1
    run oxy_versions python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))" || exit 1
    run oxy_pytest python -m pytest -q || exit 1
    run oxy_dry "${PY[@]}" dry-run --plan $PLAN --run $R --out $R/dry_run_server.json || { echo "FILES MISSING OR CHANGED ON THE SERVER" >> $ST; exit 1; }
    echo "ENV_DONE" >> $ST ;;
  scenes)  # cube placements of the 100 pilot seeds against every placement on record; no policy is run
    grep -q "^ENV_DONE" $ST || { echo "scenes: environment checks have not passed" >> $ST; exit 1; }
    run oxy_scenes "${PY[@]}" scenes --plan $PLAN --run $R || { echo "PLACEMENT CHECK FAILED: no pilot on this set" >> $ST; exit 1; }
    echo "SCENES_DONE" >> $ST ;;
  audit)   # 5 already-used scenes: controller scaling, the full four-arm procedure, tracking of the bounded horizontal move
    grep -q "^SCENES_DONE" $ST || { echo "audit: the placement check has not passed" >> $ST; exit 1; }
    run oxy_audit "${PY[@]}" audit --plan $PLAN --run $R || { echo "AUDIT FAILED: TECHNICAL_BLOCKED, the pilot set is untouched, no bound is changed" >> $ST; exit 1; }
    echo "AUDIT_DONE" >> $ST ;;
  budget)
    grep -q "^AUDIT_DONE" $ST || { echo "budget: no audit" >> $ST; exit 1; }
    run oxy_budget "${PY[@]}" budget --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "BUDGET_BLOCKED: pilot not started" >> $ST; exit 1; }
    echo "BUDGET_OK" >> $ST ;;
  pilot)   # the one pilot pass; ledger written first; never resumed
    grep -q "^BUDGET_OK" $ST || { echo "pilot: the budget gate has not passed" >> $ST; exit 1; }
    mkdir -p artifacts/ledger
    run oxy_pilot "${PY[@]}" run --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" || { echo "PILOT PASS INCOMPLETE: the set is used; no second pass" >> $ST; exit 1; }
    echo "PILOT_DONE" >> $ST ;;
  pack)
    A=/root/recovery_v1_oracle_out
    mkdir -p artifacts/ledger
    tar czf $A.tgz $R artifacts/ledger logs/oxy_* logs/recovery_v1_oracle_xy_pilot_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|audit|budget|pilot|pack"; exit 2 ;;
esac
