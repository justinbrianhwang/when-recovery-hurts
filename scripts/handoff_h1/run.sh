#!/usr/bin/env bash
# Handoff H1 driver (handoff_h1_001): the fixed D500-clean 80k EMA policy, a restore audit on this host,
# continue / r_keep / r_pad_once on at most 12 new train scenes. PREPARED, NOT RUN.
# Every phase below calls an execution command of scripts/26_handoff_h1.py, which refuses unless
# runs/handoff_h1_001/authorization.json (written by the user side: approval, cap, plan hash) exists.
# No training, no predictor, no qualification, no calibration, no Test.
# Usage: run.sh env|scenes|audit|first2|scale|rest|final|pack
# Nothing here reads which arm succeeded in order to decide what runs next.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/handoff_h1_001.status
R=runs/handoff_h1_001
PLAN=$R/scope_plan.json
PY=(python scripts/26_handoff_h1.py)
W=${WORKERS:-2}
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }

case "$1" in
  env)
    run h1_doctor python -m hci doctor --out $R/doctor || exit 1
    run h1_pytest python -m pytest -q || exit 1
    echo "ENV_DONE" >> $ST ;;
  scenes)  # initial cube placement of the new scenes against every placement on record; no policy is run
    run h1_scenes "${PY[@]}" scenes --plan $PLAN --keys $R/reference_scene_keys.json --out $R/scenes.json || exit 1
    echo "SCENES_DONE" >> $ST ;;
  audit)   # the earlier audits belong to other hosts; collection needs a PASS for this policy, config and host
    run h1_audit "${PY[@]}" audit --plan $PLAN --out $R/audit || { echo "AUDIT FAILED: no collection" >> $ST; exit 1; }
    echo "AUDIT_DONE" >> $ST ;;
  first2)  # at most 96 continuations (+ trunk, R5 control references and the declared reruns); the check that follows is technical only
    run h1_collect_first2 "${PY[@]}" collect --plan $PLAN --run $R --start 0 --stop 2 --label first2 || exit 1
    run h1_techcheck "${PY[@]}" techcheck --plan $PLAN --run $R || { echo "TECHNICAL CHECK FAILED: no further collection" >> $ST; exit 1; }
    echo "FIRST2_DONE" >> $ST ;;
  scale)   # 12 -> 6 -> 2 from measured seconds and money only
    run h1_scale "${PY[@]}" scale --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" \
        --reserve-usd ${RESERVE_USD:-0.20} --workers $W --efficiency ${EFFICIENCY:-0.7} || exit 1
    echo "SCALE_DONE" >> $ST ;;
  rest)    # the remaining scenes of the decided prefix, split over W workers by position; each scene is written once
    N=$(python -c "import json;print(json.load(open('$R/scale_decision.json'))['chosen_scenes'])") || exit 1
    if [[ $N -le 2 ]]; then echo "REST NOT RUN: decided scale is $N scenes" >> $ST; exit 0; fi
    [[ "${COLLECT_LIMIT_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "rest: COLLECT_LIMIT_S is not set" >> $ST; exit 1; }
    pids=()
    for ((w=0; w<W; w++)); do
      # each worker keeps its own command/exit ledger; the ledgers are appended to the main one after all workers ended
      ( ST=logs/handoff_h1_001.w$w.status
        run h1_collect_w$w timeout $COLLECT_LIMIT_S "${PY[@]}" collect --plan $PLAN --run $R --start $((2+w)) --stop $N --step $W --label w$w ) & pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
    for ((w=0; w<W; w++)); do cat logs/handoff_h1_001.w$w.status >> $ST; done
    [[ $rc -eq 0 ]] || { echo "COLLECTION INCOMPLETE: a worker failed or reached COLLECT_LIMIT_S=$COLLECT_LIMIT_S" >> $ST; exit 1; }
    echo "REST_DONE scenes=$N workers=$W" >> $ST ;;
  final)   # dataset.json (completed only when every scheduled scene is valid), the descriptive tables and the event rows
    run h1_finalize "${PY[@]}" finalize --plan $PLAN --run $R --results $R/results || exit 1
    echo "FINAL_DONE" >> $ST ;;
  pack)    # new outputs only; the per-file list is computed from a fresh extraction of the archive
    A=/root/handoff_h1_out
    shopt -s nullglob
    tar czf $A.tgz $R logs/h1_* logs/handoff_h1_001.*status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|audit|first2|scale|rest|final|pack"; exit 2 ;;
esac
