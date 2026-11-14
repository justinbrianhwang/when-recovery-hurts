#!/usr/bin/env bash
# Stage 1D clean rebuild driver (stage1d_clean_002): server-side data gate, ONE training run on train_d500_clean_v1,
# candidate selection, ONE qualification on policy_qualify_B, the Stage 1C comparison, and (only above the gate)
# the restore audit and a two-episode branch check.
# Usage: run_clean.sh env|gate|train|stop|select|diag|post|pack [SUFFIX]
# The data set is built and validated locally before a GPU is rented; this script never converts or rebuilds data.
# train needs TRAIN_LIMIT_S, SPENT_USD, USD_PER_HOUR (and optionally CAP_USD, RESERVE_USD): the gate refuses the start
# when a validation layer is missing, a file differs from the validated one, policy_qualify_B is used, or
# time limit x price + reserve does not fit the remaining budget.
# stop signals only the trainer's own process group (never the SSH session) and records the reason.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/stage1d_clean_002.status
X=${2:-}
R=runs/stage1d_clean_002
RUN=stack_1d_s1_d500_clean_v1
DATA=$R/data
TRAIN=$DATA/train_d500_clean_v1.h5
DM=$DATA/train_d500_clean_manifest.json
SAMPLE=$DATA/io_parity_sample_clean_v1.h5
H5=trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
OLD=data_backup/demos_stack_002/$H5
SRC=data_backup/demos_stack_002/trajectory.h5
DIAG=runs/stage1c_001/manifest/diagnostic_manifest.json
P60=artifacts/policies/stack_1b_s1_d100_i060000
CK60=runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt
QSET=policy_qualify_B
PGID_FILE=logs/train_${RUN}.pgid
[[ -f $R/plan.json ]] || { echo "plan missing: $R/plan.json" >> $ST; exit 1; }
sampler(){ ( while sleep 5; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits),$(free -m | awk '/Mem:/{print $3}')"; done > "$1" ) & SAMPLER=$!; }
selected(){ python -c "import json;print(json.load(open('runs/select_${RUN}/selection.json'))['$1'])"; }

case "$1" in
  env)
    run s1dc_doctor${X} python -m hci doctor --out $R/doctor${X} || exit 1
    run s1dc_pytest${X} python -m pytest -q || exit 1
    echo "ENV_DONE" >> $ST ;;
  gate)  # the uploaded files are re-hashed and re-read; then the two simulator layers are added
    run s1dc_gate_data${X} python scripts/17_stage1d_clean.py gate --data $DATA --out $R/gate_data_server${X}.json \
        || { echo "DATA GATE FAILED: no training" >> $ST; exit 1; }
    run s1dc_replay${X} python scripts/17_stage1d_clean.py replay --config $C --data $DATA --out $R/replay${X} \
        || { echo "EXPERT REPLAY FAILED: data gate closed, no training" >> $ST; exit 1; }
    # observation/target alignment of the new dataset path: all replacements + retained old + retained new, with the preserved 60k policy
    N=$(python -c "import json;print(len(json.load(open('$DM'))['io_parity_sample_train_ids']))") || exit 1
    run s1dc_io_dataset_path${X} python scripts/08_policy_io_audit.py --config $C --policy $P60 --checkpoint $CK60 --demo $SAMPLE --trajectories $N \
        --out $R/io_dataset_path_60k${X} || { echo "IO AUDIT OF THE NEW DATASET PATH FAILED: no training" >> $ST; exit 1; }
    [[ -e $DATA/policy_io_parity.json ]] && { echo "gate: $DATA/policy_io_parity.json exists (use a new data folder)" >> $ST; exit 1; }
    cp $R/io_dataset_path_60k${X}/policy_io_audit.json $DATA/policy_io_parity.json || exit 1
    echo "GATE_DONE" >> $ST ;;
  train)  # Stage 1B arguments (scripts/stage1b/20_train.sh) with the clean file and 500 demonstrations; from scratch
    [[ "${TRAIN_LIMIT_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "train: TRAIN_LIMIT_S is not set" >> $ST; exit 1; }
    run s1dc_gate_train_start${X} python scripts/17_stage1d_clean.py gate --data $DATA --train-start --cap-usd ${CAP_USD:-2.0} --spent-usd "${SPENT_USD:?}" \
        --usd-per-hour "${USD_PER_HOUR:?}" --train-limit-s $TRAIN_LIMIT_S --reserve-usd ${RESERVE_USD:-0.35} --out $R/gate_train_start${X}.json \
        || { echo "TRAIN START REFUSED BY THE GATE" >> $ST; exit 1; }
    [[ -e runs/policy/$RUN ]] && { echo "train: runs/policy/$RUN exists; a stopped run is not resumed" >> $ST; exit 1; }
    sampler logs/s1dc_train_gpu${X}.csv
    echo "train_${RUN} cmd=timeout $TRAIN_LIMIT_S python scripts/03_train_dp.py --config $C --demo $TRAIN --run $RUN --num-demos 500 --total-iters 100000 --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 --data-manifest $DM --execute" >> $ST
    echo "train_${RUN} start=$(date -u +%FT%TZ)" >> $ST
    setsid timeout $TRAIN_LIMIT_S python scripts/03_train_dp.py --config $C --demo $TRAIN --run $RUN --num-demos 500 --total-iters 100000 \
        --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 --data-manifest $DM --execute > logs/train_${RUN}.log 2>&1 & TP=$!
    echo "$TP $(date -u +%FT%TZ)" > $PGID_FILE          # setsid: the trainer leads its own process group, so this pid is the group id
    DEADLINE=$(( $(date +%s) + TRAIN_LIMIT_S + 900 ))     # the whole train-and-monitor phase ends by this time
    (  # descriptive monitor beside the trainer; it stops waiting as soon as the trainer is gone, and every step is time-limited
      for tag in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000; do
        f=runs/policy/$RUN/checkpoints/full_${tag}.pt
        until [[ -f $f ]]; do
          if ! kill -0 $TP 2>/dev/null; then sleep 2; [[ -f $f ]] && break; echo "monitor_${RUN}: training ended without $f" >> $ST; exit 1; fi
          sleep 30
        done
        [[ $(date +%s) -lt $DEADLINE ]] || { echo "monitor_${RUN}: phase deadline reached before $tag" >> $ST; exit 1; }
        P=artifacts/policies/${RUN}_i${tag}
        run export_${RUN}_i${tag} timeout 300 python scripts/04_export_policy.py --run runs/policy/$RUN --checkpoint $f --out $P || exit 1
        run monitor_${RUN}_i${tag} timeout 900 python scripts/09_eval_policy.py --config $C --policy $P --set policy_monitor --demo-manifest $DM --out runs/monitor_${RUN}_i${tag} || exit 1
      done
      echo "MONITOR_DONE ${RUN}" >> $ST
    ) & MON=$!
    wait $TP; rc=$?
    rm -f $PGID_FILE
    echo "train_${RUN} exit=$rc end=$(date -u +%FT%TZ)" >> $ST
    [[ $rc -eq 124 ]] && echo "train_${RUN}: stopped by TRAIN_LIMIT_S=$TRAIN_LIMIT_S" >> $ST
    wait $MON; mrc=$?
    kill $SAMPLER 2>/dev/null
    [[ $rc -eq 0 && $mrc -eq 0 ]] || { echo "TRAIN PHASE FAILED train=$rc monitor=$mrc" >> $ST; exit 1; }
    echo "TRAIN_DONE ${RUN}" >> $ST ;;
  stop)  # run_clean.sh stop "REASON": TERM to the trainer's process group only
    [[ -f $PGID_FILE ]] || { echo "stop: no trainer pid file" >> $ST; exit 1; }
    G=$(cut -d' ' -f1 $PGID_FILE)
    # the group must still be led by this run's trainer: same pid, group leader, and the command names this run
    INFO=$(ps -o pid=,pgid=,args= -p "$G" 2>/dev/null)
    if [[ -z $INFO || $(awk '{print $2}' <<< "$INFO") != "$G" || $INFO != *"03_train_dp.py"*"--run $RUN "* ]]; then
      echo "train_${RUN} STOP refused: pid $G is not this run's trainer group leader" >> $ST; exit 1
    fi
    echo "train_${RUN} STOP requested=$(date -u +%FT%TZ) pgid=$G reason=${2:-unspecified}" >> $ST
    kill -TERM -- -$G; rc=$?
    echo "train_${RUN} STOP signal_exit=$rc" >> $ST ;;
  select)  # three pre-declared candidates, one after another, on the same 50 seeds; parity of the selected one; then ONE qualification
    for tag in 060000 080000 100000; do
      [[ -f artifacts/policies/${RUN}_i${tag}/weights.pt ]] || { echo "select_${RUN}: missing exported candidate $tag" >> $ST; exit 1; }
      run select_${RUN}_i${tag} python scripts/09_eval_policy.py --config $C --policy artifacts/policies/${RUN}_i${tag} --set policy_select \
          --success-at-end --demo-manifest $DM --out runs/select_${RUN}_i${tag} || exit 1
    done
    run choose_${RUN} python scripts/11_select_candidate.py --candidate 60000=runs/select_${RUN}_i060000 --candidate 80000=runs/select_${RUN}_i080000 \
        --candidate 100000=runs/select_${RUN}_i100000 --out runs/select_${RUN}/selection.json || exit 1
    SEL=$(selected selected_policy) || exit 1
    CK=runs/policy/$RUN/checkpoints/full_$(printf %06d "$(selected selected_iteration)").pt
    N=$(python -c "import json;print(len(json.load(open('$DM'))['io_parity_sample_train_ids']))") || exit 1
    run s1dc_io_parity_selected${X} python scripts/08_policy_io_audit.py --config $C --policy $SEL --checkpoint $CK --demo $SAMPLE --trajectories $N \
        --out $R/io_parity_selected${X} || { echo "IO PARITY FAILED: qualification not started" >> $ST; exit 1; }
    run qualify_${RUN} python scripts/09_eval_policy.py --config $C --policy $SEL --set $QSET --success-at-end --demo-manifest $DM \
        --out runs/qualify_${RUN} || exit 1
    echo "QUALIFY_DONE ${RUN} ${SEL}" >> $ST ;;
  diag)  # the Stage 1C scenes (all 20 passed the content audit), histories and noise seeds with the selected policy; D4 is not repeated
    SEL=$(selected selected_policy) || exit 1
    K=$(python -c "import json;print(json.load(open('runs/qualify_${RUN}/policy_eval.json'))['success_once'])") || exit 1
    # spec table: 0-9/100 ends the stage without the comparison; from 10/100 the comparison is run
    if [[ $K -lt 10 ]]; then echo "DIAG NOT RUN: qualify=$K/100 is in the 0-9 band" >> $ST; exit 0; fi
    run s1dc_scene_rollouts${X} python scripts/14_scene_rollouts.py --config $C --manifest $DIAG --demo $OLD --source $SRC --policy $SEL --other-policy \
        --cells F_policy R_policy --out $R/scene_rollouts${X} || exit 1
    run s1dc_offline${X} python scripts/15_offline_action_diag.py --config $C --manifest $DIAG --demo $OLD --policy $SEL --other-policy --out $R/offline${X} || exit 1
    echo "DIAG_DONE" >> $ST ;;
  post)  # only at or above the gate: a new restore audit with the new weights, then the branches of two source episodes
    SEL=$(selected selected_policy) || exit 1
    K=$(python -c "import json;print(json.load(open('runs/qualify_${RUN}/policy_eval.json'))['success_once'])") || exit 1
    grep -q "^QUALIFY_DONE ${RUN} " $ST || { echo "POST NOT RUN: qualification did not complete" >> $ST; exit 1; }
    if [[ $K -lt 50 ]]; then echo "POST NOT RUN: qualify=$K/100 is below the gate (>=50)" >> $ST; exit 0; fi
    run audit_${RUN}${X} python -m hci audit --config $C --policy $SEL --out runs/audit_${RUN}${X} || { echo "AUDIT FAILED: no collection" >> $ST; exit 1; }
    run branches_smoke_${RUN}${X} python -m hci collect --config $C --policy $SEL --audit runs/audit_${RUN}${X}/audit.json --episodes 2 --out runs/branches_smoke_${RUN}${X} || exit 1
    run inspect_branches_${RUN}${X} python scripts/inspect_branches.py --data runs/branches_smoke_${RUN}${X} || exit 1
    echo "POST_DONE" >> $ST ;;
  pack)  # new outputs only; the training file and its sample were verified locally before upload and are not re-archived.
         # The per-file list is computed from a fresh extraction of the archive.
    A=/root/stage1d_clean_out${X}
    shopt -s nullglob
    tar czf $A.tgz --exclude="$TRAIN" --exclude="$SAMPLE" $R runs/policy/$RUN runs/*_${RUN}* artifacts/policies/${RUN}_i* artifacts/ledger \
        logs/s1dc_* logs/*${RUN}* $ST || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '\n' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run_clean.sh env|gate|train|stop|select|diag|post|pack [SUFFIX]"; exit 2 ;;
esac
