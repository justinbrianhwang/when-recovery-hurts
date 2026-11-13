#!/usr/bin/env bash
# Stage 1B, before any long training: environment checks, demonstrations, I/O parity audit, retreat diagnostic,
# and a short batch-256 probe for memory and throughput. Stops at the first failure.
source "$(dirname "$0")/common.sh"
OLD=artifacts/policies/stack_s1_d99
run doctor_002 python -m hci doctor --out runs/doctor_002 || exit 1
run pytest_002 python -m pytest -q || exit 1
run prepare_demos_stack_002 bash scripts/02_prepare_demos.sh StackCube-v1 130 || exit 1
run validate_demos_002 python scripts/07_validate_demos.py --config $C --demo "$DEMO" --source "$SRC" --out runs/demos_stack_002 \
    --num-train 100 --num-heldout 20 --replay 10 --record 3 --convert-command "bash scripts/02_prepare_demos.sh StackCube-v1 130" || exit 1
run retreat_diag_001 python scripts/10_retreat_diagnostic.py --config $C --demo "$DEMO" --out runs/retreat_diag_001 --demos 10 || exit 1
run policy_io_audit_001 python scripts/08_policy_io_audit.py --config $C --policy $OLD \
    --checkpoint runs/policy/stack_s1_d99/checkpoints/best_eval_success_once.pt --demo "$DEMO" --out runs/policy_io_audit_001 \
    || { echo "IO AUDIT FAILED: no training" >> $ST; exit 1; }
( while sleep 5; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; done > logs/probe_b256_gpu.csv ) & SAMPLER=$!
run probe_b256_001 python scripts/03_train_dp.py --config $C --demo "$DEMO" --run stack_1b_probe_b256 --num-demos 100 --total-iters 1000 \
    --batch-size 256 --eval-freq 500 --num-eval-episodes 2 --data-manifest runs/demos_stack_002/demo_manifest.json --execute; rc=$?
kill $SAMPLER
[[ $rc -eq 0 ]] || exit 1
echo "PRE_TRAINING_DONE" >> $ST
