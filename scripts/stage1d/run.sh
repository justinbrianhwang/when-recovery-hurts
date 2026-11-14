#!/usr/bin/env bash
# Stage 1D driver: D500 data, ONE training run, candidate selection, ONE qualification on policy_qualify_B,
# the Stage 1C comparison, and (only above the gate) the restore audit and a two-episode branch check.
# Usage: run.sh env|data|probe|train|select|diag|post|pack [SUFFIX]
# DATA_SUFFIX selects the data build (set it when the data phase had to be repeated under a new name).
# TRAIN_LIMIT_S (required for train) is the wall-clock limit of the training command, set from the measured probe
# throughput and the remaining budget; the trainer is stopped when it is reached.
# SUFFIX gives a failed phase new step/run names on the next attempt; earlier logs and outputs are kept.
# Every step is logged with command, start/end time and exit code in logs/stage1d_001.status. Nothing is retried,
# no threshold is changed here, and a failed step stops the phase.
source "$(dirname "$0")/../stage1b/common.sh"
ST=logs/stage1d_001.status
X=${2:-}
R=runs/stage1d_001
PLAN=$R/plan/stage1d_plan.json
RUN=stack_1d_s1_d500
H5=trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
OLD=data_backup/demos_stack_002/$H5
SRC=data_backup/demos_stack_002/trajectory.h5
DS=${DATA_SUFFIX:-}
NEW=$R/new_source${DS}
DATA=$R/data${DS}
TRAIN=$DATA/train_d500/$H5
DM=$DATA/train_d500_manifest.json
SAMPLE=$DATA/io_parity_sample/$H5
DIAG=runs/stage1c_001/manifest/diagnostic_manifest.json
P60=artifacts/policies/stack_1b_s1_d100_i060000
CK60=runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt
QSET=policy_qualify_B
[[ -f $PLAN ]] || { echo "plan missing: $PLAN" >> $ST; exit 1; }
sampler(){ ( while sleep 5; do echo "$(date -u +%T),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits),$(free -m | awk '/Mem:/{print $3}')"; done > "$1" ) & SAMPLER=$!; }
data_ok(){ python -c "import json,sys;a=json.load(open('$DATA/subset_and_overlap_audit.json'));v=json.load(open('$DATA/demo_validation.json'));sys.exit(0 if a['pass'] and v['pass'] else 1)" || { echo "data build $DATA has no passing audit/validation" >> $ST; exit 1; }; }
selected(){ python -c "import json;print(json.load(open('runs/select_${RUN}/selection.json'))['$1'])"; }

case "$1" in
  env)
    run s1d_doctor${X} python -m hci doctor --out $R/doctor${X} || exit 1
    run s1d_pytest${X} python -m pytest -q || exit 1
    echo "ENV_DONE" >> $ST ;;
  data)  # new conversion with the Stage 1B command and flags, restricted to the source episodes declared in the plan
    N=$(python -c "import json;print(json.load(open('$PLAN'))['new_candidates']['window'])") || exit 1
    run s1d_subset${X} python scripts/16_stage1d_data.py subset --plan $PLAN --source $SRC --out-dir $NEW || exit 1
    run s1d_convert${X} python -m mani_skill.trajectory.replay_trajectory --traj-path $NEW/trajectory.h5 --use-first-env-state -c pd_ee_delta_pos -o rgb \
        -b physx_cpu --num-envs 1 --count $N --save-traj || exit 1
    run s1d_build${X} python scripts/16_stage1d_data.py build --config $C --plan $PLAN --old-demo $OLD --new-demo $NEW/$H5 --source $SRC --out $DATA \
        || { echo "DATA CHECK FAILED: no training" >> $ST; exit 1; }
    run s1d_validate${X} python scripts/16_stage1d_data.py validate --config $C --data $DATA --out $R/validate${X} \
        || { echo "DEMO VALIDATION FAILED: no training" >> $ST; exit 1; }
    # Observation/target alignment of the NEW dataset path (3 old + 3 new demonstrations), checked with the preserved 60k policy before any training.
    run s1d_io_dataset_path${X} python scripts/08_policy_io_audit.py --config $C --policy $P60 --checkpoint $CK60 --demo $SAMPLE \
        --out $R/io_dataset_path_60k${X} || { echo "IO AUDIT OF THE NEW DATASET PATH FAILED: no training" >> $ST; exit 1; }
    echo "DATA_DONE" >> $ST ;;
  probe)  # memory and throughput with all 500 demonstrations loaded; this model is discarded
    data_ok
    sampler logs/s1d_probe_gpu${X}.csv
    run s1d_probe_b256${X} python scripts/03_train_dp.py --config $C --demo $TRAIN --run stack_1d_probe_b256${X} --num-demos 500 --total-iters 1000 \
        --batch-size 256 --eval-freq 500 --num-eval-episodes 2 --data-manifest $DM --execute; rc=$?
    kill $SAMPLER
    [[ $rc -eq 0 ]] || exit 1
    echo "PROBE_DONE" >> $ST ;;
  train)  # same arguments as Stage 1B (scripts/stage1b/20_train.sh) except the demonstration file and count
    data_ok
    [[ "${TRAIN_LIMIT_S:-}" =~ ^[1-9][0-9]*$ ]] || { echo "train: TRAIN_LIMIT_S is not set" >> $ST; exit 1; }
    sampler logs/s1d_train_gpu${X}.csv
    trap 'kill $SAMPLER $MON $TP 2>/dev/null; pkill -f "03_train_dp.py.*--run $RUN " 2>/dev/null' EXIT
    run train_${RUN} timeout $TRAIN_LIMIT_S python scripts/03_train_dp.py --config $C --demo $TRAIN --run $RUN --num-demos 500 --total-iters 100000         --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 --data-manifest $DM --execute & TP=$!
    (  # descriptive monitor beside the trainer; it stops waiting as soon as the trainer is gone
      for tag in 010000 020000 030000 040000 050000 060000 070000 080000 090000 100000; do
        f=runs/policy/$RUN/checkpoints/full_${tag}.pt
        until [[ -f $f ]]; do
          if ! kill -0 $TP 2>/dev/null; then sleep 2; [[ -f $f ]] && break; echo "monitor_${RUN}: training ended without $f" >> $ST; exit 1; fi
          sleep 30
        done
        P=artifacts/policies/${RUN}_i${tag}
        run export_${RUN}_i${tag} python scripts/04_export_policy.py --run runs/policy/$RUN --checkpoint $f --out $P || exit 1
        run monitor_${RUN}_i${tag} python scripts/09_eval_policy.py --config $C --policy $P --set policy_monitor --demo-manifest $DM --out runs/monitor_${RUN}_i${tag} || exit 1
      done
      echo "MONITOR_DONE ${RUN}" >> $ST
    ) & MON=$!
    wait $TP; rc=$?
    [[ $rc -eq 124 ]] && echo "train_${RUN}: stopped by TRAIN_LIMIT_S=$TRAIN_LIMIT_S" >> $ST
    wait $MON; mrc=$?
    kill $SAMPLER 2>/dev/null; trap - EXIT
    [[ $rc -eq 0 && $mrc -eq 0 ]] || { echo "TRAIN PHASE FAILED train=$rc monitor=$mrc" >> $ST; exit 1; }
    echo "TRAIN_DONE ${RUN}" >> $ST ;;
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
    run s1d_io_parity_selected${X} python scripts/08_policy_io_audit.py --config $C --policy $SEL --checkpoint $CK --demo $SAMPLE --out $R/io_parity_selected${X} \
        || { echo "IO PARITY FAILED: qualification not started" >> $ST; exit 1; }
    run qualify_${RUN} python scripts/09_eval_policy.py --config $C --policy $SEL --set $QSET --success-at-end --demo-manifest $DM \
        --out runs/qualify_${RUN} || exit 1
    echo "QUALIFY_DONE ${RUN} ${SEL}" >> $ST ;;
  diag)  # the Stage 1C scenes, histories and noise seeds with the selected D500 policy (policy cells only; D4 is not repeated)
    SEL=$(selected selected_policy) || exit 1
    run s1d_scene_rollouts${X} python scripts/14_scene_rollouts.py --config $C --manifest $DIAG --demo $OLD --source $SRC --policy $SEL --other-policy \
        --cells F_policy R_policy --out $R/scene_rollouts${X} || exit 1
    run s1d_offline${X} python scripts/15_offline_action_diag.py --config $C --manifest $DIAG --demo $OLD --policy $SEL --other-policy --out $R/offline${X} || exit 1
    echo "DIAG_DONE" >> $ST ;;
  post)  # branch A only: gate reached and parity passed. A new audit with the new weights; never the Stage 1 audit.
    SEL=$(selected selected_policy) || exit 1
    K=$(python -c "import json;print(json.load(open('runs/qualify_${RUN}/policy_eval.json'))['success_once'])") || exit 1
    # QUALIFY_DONE is only written after the selected policy passed the I/O parity audit and the qualification finished.
    grep -q "^QUALIFY_DONE ${RUN} " $ST || { echo "POST NOT RUN: qualification did not complete" >> $ST; exit 1; }
    if [[ $K -lt 50 ]]; then echo "POST NOT RUN: qualify=$K/100 is below the gate (>=50)" >> $ST; exit 0; fi
    run audit_stack_1d${X} python -m hci audit --config $C --policy $SEL --out runs/audit_stack_1d${X} || { echo "AUDIT FAILED: no collection" >> $ST; exit 1; }
    run branches_smoke_1d${X} python -m hci collect --config $C --policy $SEL --audit runs/audit_stack_1d${X}/audit.json --episodes 2 --out runs/branches_smoke_1d${X} || exit 1
    run inspect_branches_1d${X} python scripts/inspect_branches.py --data runs/branches_smoke_1d${X} || exit 1
    echo "POST_DONE" >> $ST ;;
  pack)  # new outputs only. The per-file list is computed from a fresh extraction of the archive, not from the live files;
         # the archive hash is compared after download and the list is checked again after extraction on the receiving side.
    A=/root/stage1d_out${X}
    tar czf $A.tgz --exclude="$R/new_source*/trajectory.h5" $R runs/policy/$RUN runs/*_1d*         artifacts/policies/${RUN}_i* artifacts/ledger logs/s1d_* logs/*_1d* logs/stage1d_001.status || exit 1
    rm -rf $A.verify && mkdir $A.verify && tar xzf $A.tgz -C $A.verify || exit 1
    ( cd $A.verify && find . -type f | sed 's|^\./||' | sort | xargs -d '
' sha256sum ) > $A.sha256 || exit 1
    rm -rf $A.verify
    sha256sum $A.tgz; wc -l < $A.sha256; ls -la $A.tgz ;;
  *) echo "usage: run.sh env|data|probe|train|select|diag|post|pack [SUFFIX]"; exit 2 ;;
esac
