#!/usr/bin/env bash
# Vision-V1 driver (vision_v1_001): two trainings of the Diffusion Policy with the pretrained ResNet-18 encoder on the
# same validated 500 demonstrations (train seeds 1 and 2), the fixed 80k EMA candidates on vision_v1_select, and ONE
# paired qualification of the selected candidate against the frozen legacy policy on vision_v1_qualify. PREPARED, NOT RUN.
# Usage: run.sh env|data|scenes|probe|budget|train SEED|export SEED|parity SEED|select SEED|choose|qualify|pack
# Every phase needs runs/vision_v1_001/authorization.json (written by the user side: approval, cap, plan hash).
# budget needs USD_PER_HOUR, SPENT_USD and PEAK_VRAM_MIB; train needs TRAIN_LIMIT_S. Nothing here reads an outcome to
# decide what is trained, which checkpoint is exported or how many scenes are evaluated.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/vision_v1_001.status
R=runs/vision_v1_001
PLAN=$R/plan.json
PY=(python scripts/29_vision_v1.py)
DATA=runs/stage1d_clean_002/data
TRAIN=$DATA/train_d500_clean_v1.h5
DM=$DATA/train_d500_clean_manifest.json
PRE=artifacts/pretrained/resnet18-f37072fd.pth
SEED=${2:-}
RUN=stack_v1_s${SEED}_d500_clean_resnet18
POL=artifacts/policies/${RUN}_i080000
[[ -f $R/authorization.json ]] || { echo "REFUSED: no authorization.json (paid compute not authorized)" >> $ST; exit 1; }
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }
need_seed(){ [[ "$SEED" == 1 || "$SEED" == 2 ]] || { echo "$1: train seed must be 1 or 2" >> $ST; exit 1; }; }
sampler(){ ( while sleep 5; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"; done > "$1" ) & SAMPLER=$!; }

case "$1" in
  env)
    run v1_doctor python -m hci doctor --out $R/doctor || exit 1
    run v1_versions python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name(0))" || exit 1
    run v1_pytest python -m pytest -q || exit 1
    echo "ENV_DONE" >> $ST ;;
  data)    # the uploaded bytes are re-hashed and re-read; the recorded expert actions of the sampled demonstrations are replayed on this host
    run v1_data_sha bash -c "echo '93416a4a8b7e2203d4cca1fe3c5d1e577830fa720ea4f3855021a168ea65f342  $TRAIN' | sha256sum -c - && echo 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  $PRE' | sha256sum -c -" \
        || { echo "DATA OR PRETRAINED FILE HASH MISMATCH: no training" >> $ST; exit 1; }
    run v1_gate_data python scripts/17_stage1d_clean.py gate --data $DATA --qualify-set vision_v1_qualify --out $R/gate_data_server.json || { echo "DATA GATE FAILED: no training" >> $ST; exit 1; }
    mkdir -p $R/data_recheck && ln -s "$(realpath $TRAIN)" $R/data_recheck/train_d500_clean_v1.h5 && cp $DM $DATA/train_d500_clean_v1.json $R/data_recheck/ || exit 1
    run v1_replay python scripts/17_stage1d_clean.py replay --config $C --data $R/data_recheck --out $R/replay || { echo "EXPERT REPLAY FAILED: no training" >> $ST; exit 1; }
    echo "DATA_DONE" >> $ST ;;
  scenes)  # cube placements of the 200 new evaluation seeds against every placement on record; no policy is run
    run v1_scenes "${PY[@]}" scenes --plan $PLAN --keys $R/reference_scene_keys.json --out $R/scenes.json || { echo "PLACEMENT CHECK FAILED: no evaluation on these sets" >> $ST; exit 1; }
    echo "SCENES_DONE" >> $ST ;;
  probe)   # non-candidate throughput probe: full data, batch 256, 300 updates; its weights are never exported
    sampler logs/v1_probe_gpu.csv
    run v1_probe python scripts/27_vision_v1_train.py --config $C --demo $TRAIN --run vision_v1_probe --pretrained $PRE --num-demos 500 --total-iters 300 --batch-size 256 \
        --eval-freq 300 --num-eval-episodes 1 --seed 0 --data-manifest $DM --probe --execute --plan $PLAN --auth-run $R; rc=$?
    kill $SAMPLER 2>/dev/null
    [[ $rc -eq 0 ]] || { echo "PROBE FAILED (e.g. out of memory at batch 256): TECHNICAL_BLOCKED, no training" >> $ST; exit 1; }
    echo "PROBE_DONE peak_vram_mib=$(cut -d, -f2 logs/v1_probe_gpu.csv | sort -n | tail -1)" >> $ST ;;
  budget)  # projected all-in cost of the two fixed trainings and four 100-scene passes against the approved cap
    run v1_budget "${PY[@]}" budget --plan $PLAN --run $R --probe-log logs/v1_probe.log --usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}" \
        --peak-vram-mib "${PEAK_VRAM_MIB:?}" --eval-seconds-per-episode ${EVAL_S_PER_EPISODE:-8} || { echo "BUDGET_BLOCKED: training not started" >> $ST; exit 1; }
    echo "BUDGET_OK" >> $ST ;;
  train)   # exactly the legacy command line plus the encoder; from scratch; a stopped run is not resumed
    need_seed train
    grep -q "^BUDGET_OK" $ST || { echo "train: the budget gate has not passed" >> $ST; exit 1; }
    [[ "${TRAIN_LIMIT_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "train: TRAIN_LIMIT_S is not set" >> $ST; exit 1; }
    [[ -e runs/policy/$RUN ]] && { echo "train: runs/policy/$RUN exists; a stopped run is not resumed" >> $ST; exit 1; }
    sampler logs/v1_train_s${SEED}_gpu.csv
    run v1_train_s${SEED} timeout $TRAIN_LIMIT_S python scripts/27_vision_v1_train.py --config $C --demo $TRAIN --run $RUN --pretrained $PRE --num-demos 500 --total-iters 100000 \
        --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed $SEED --data-manifest $DM --execute --plan $PLAN --auth-run $R; rc=$?
    kill $SAMPLER 2>/dev/null
    [[ $rc -eq 0 ]] || { echo "TRAIN s${SEED} INCOMPLETE exit=$rc (kept as it is; no retry, no other seed)" >> $ST; exit 1; }
    echo "TRAIN_DONE s${SEED}" >> $ST ;;
  export)  # the predeclared 80k EMA checkpoint, saved and hashed before any evaluation
    need_seed export
    grep -q "^TRAIN_DONE s${SEED}" $ST || { echo "export: training of seed $SEED did not complete" >> $ST; exit 1; }
    run v1_export_s${SEED} python scripts/28_vision_v1_export.py --run runs/policy/$RUN --checkpoint runs/policy/$RUN/checkpoints/full_080000.pt --iteration 80000 --pretrained $PRE --plan $PLAN --auth-run $R --out $POL || exit 1
    echo "EXPORT_DONE s${SEED} $(sha256sum $POL/weights.pt | cut -c1-64)" >> $ST ;;
  select)  # 100 scenes of vision_v1_select with the fixed candidate of this training seed
    need_seed select
    grep -q "^EXPORT_DONE s${SEED}" $ST || { echo "select: no exported candidate of seed $SEED" >> $ST; exit 1; }
    run v1_select_s${SEED} "${PY[@]}" eval-select --plan $PLAN --run $R --policy $POL --train-seed $SEED || exit 1
    echo "SELECT_DONE s${SEED}" >> $ST ;;
  choose)  # more successes wins, tie -> seed 1; both below 50/100 -> stop, the qualification set stays unused
    run v1_choose "${PY[@]}" choose --plan $PLAN --run $R; rc=$?
    [[ $rc -eq 3 ]] && { echo "NO_VIABLE_DEV_CANDIDATE: qualification not consumed" >> $ST; exit 3; }
    [[ $rc -eq 0 ]] || exit 1
    echo "CHOOSE_DONE" >> $ST ;;
  qualify) # ONE paired session: selected new policy and frozen legacy policy on the same 100 scenes; ledger written first; never resumed
    grep -q "^CHOOSE_DONE" $ST || { echo "qualify: no selection" >> $ST; exit 1; }
    run v1_qualify "${PY[@]}" qualify-paired --plan $PLAN --run $R || { echo "QUALIFICATION INCOMPLETE: the set is consumed; no second session" >> $ST; exit 1; }
    echo "QUALIFY_DONE" >> $ST ;;
  pack)    # new outputs only; the training file and the pretrained file were verified by hash and are not re-archived
           # of the full training checkpoints only tags 80000 (the candidate) and 100000 (end of schedule) are archived
    A=/root/vision_v1_out
    shopt -s nullglob
    find runs/policy -name "full_*.pt" ! -name "full_080000.pt" ! -name "full_100000.pt" > $A.exclude || exit 1      # explicit list of the checkpoints left on the server
    tar czf $A.tgz --exclude="$R/data_recheck/*.h5" -X $A.exclude $R runs/policy/stack_v1_s*_d500_clean_resnet18 runs/policy/vision_v1_probe/*.json artifacts/policies/stack_v1_s*_d500_clean_resnet18_i080000 \
        artifacts/ledger logs/v1_* logs/vision_v1_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    tar tzf $A.tgz | grep -c 'checkpoints/full_'; sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|data|scenes|probe|budget|train SEED|export SEED|select SEED|choose|qualify|pack"; exit 2 ;;
esac
