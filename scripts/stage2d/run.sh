#!/usr/bin/env bash
# Stage 2D driver (stage2d_001): the fixed D500-clean 80k EMA policy, a restore audit on this host,
# continue / retreat_5cm / retreat_2cm_matched / hold_matched on at most 12 new train scenes.
# No training, no distance or duration search, no predictor, no qualification, no calibration, no Test.
# Usage: run.sh env|scenes|audit|first2|scale|rest|final|pack
# The plan is written locally before a GPU is rented; every server command re-checks its hashes and refuses on a difference.
# scale needs USD_PER_HOUR and SPENT_USD (credit-based); rest needs COLLECT_LIMIT_S. WORKERS must be the same for both.
# Nothing here reads which arm succeeded in order to decide what runs next.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/stage2d_001.status
R=runs/stage2d_001
PLAN=$R/plan.json
PY=(python scripts/23_stage2d_pilot.py)
W=${WORKERS:-2}
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }

case "$1" in
  env)
    run s2d_doctor python -m hci doctor --out $R/doctor || exit 1
    run s2d_pytest python -m pytest -q || exit 1
    echo "ENV_DONE" >> $ST ;;
  scenes)  # initial cube placement of the new scenes against every placement on record; no policy is run
    run s2d_scenes "${PY[@]}" scenes --plan $PLAN --keys $R/reference_scene_keys.json --out $R/scenes.json || exit 1
    echo "SCENES_DONE" >> $ST ;;
  audit)   # the earlier audits belong to other hosts; collection needs a PASS for this policy, config and host
    run s2d_audit "${PY[@]}" audit --plan $PLAN --out $R/audit || { echo "AUDIT FAILED: no collection" >> $ST; exit 1; }
    echo "AUDIT_DONE" >> $ST ;;
  first2)  # at most 128 continuations (+ trunk, T_ref references and the declared reruns); the check that follows is technical only
    run s2d_collect_first2 "${PY[@]}" collect --plan $PLAN --run $R --start 0 --stop 2 --label first2 || exit 1
    run s2d_techcheck "${PY[@]}" techcheck --plan $PLAN --run $R || { echo "TECHNICAL CHECK FAILED: no further collection" >> $ST; exit 1; }
    echo "FIRST2_DONE" >> $ST ;;
  scale)   # 12 -> 6 -> 2 from measured seconds and money only
    run s2d_scale "${PY[@]}" scale --plan $PLAN --run $R --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" \
        --reserve-usd ${RESERVE_USD:-0.20} --workers $W --efficiency ${EFFICIENCY:-0.7} || exit 1
    echo "SCALE_DONE" >> $ST ;;
  rest)    # the remaining scenes of the decided prefix, split over W workers by position; each scene is written once
    N=$(python -c "import json;print(json.load(open('$R/scale_decision.json'))['chosen_scenes'])") || exit 1
    if [[ $N -le 2 ]]; then echo "REST NOT RUN: decided scale is $N scenes" >> $ST; exit 0; fi
    [[ "${COLLECT_LIMIT_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "rest: COLLECT_LIMIT_S is not set" >> $ST; exit 1; }
    pids=()
    for ((w=0; w<W; w++)); do
      # each worker keeps its own command/exit ledger; the ledgers are appended to the main one after all workers ended
      ( ST=logs/stage2d_001.w$w.status
        run s2d_collect_w$w timeout $COLLECT_LIMIT_S "${PY[@]}" collect --plan $PLAN --run $R --start $((2+w)) --stop $N --step $W --label w$w ) & pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
    for ((w=0; w<W; w++)); do cat logs/stage2d_001.w$w.status >> $ST; done
    [[ $rc -eq 0 ]] || { echo "COLLECTION INCOMPLETE: a worker failed or reached COLLECT_LIMIT_S=$COLLECT_LIMIT_S" >> $ST; exit 1; }
    echo "REST_DONE scenes=$N workers=$W" >> $ST ;;
  final)   # dataset.json (completed only when every scheduled scene is valid), the descriptive tables and the event rows
    run s2d_finalize "${PY[@]}" finalize --plan $PLAN --run $R --results $R/results || exit 1
    echo "FINAL_DONE" >> $ST ;;
  pack)    # new outputs only; the per-file list is computed from a fresh extraction of the archive
    A=/root/stage2d_out
    shopt -s nullglob
    tar czf $A.tgz $R logs/s2d_* logs/stage2d_001.*status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|scenes|audit|first2|scale|rest|final|pack"; exit 2 ;;
esac
