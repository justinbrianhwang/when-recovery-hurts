#!/usr/bin/env bash
# Stage 1B training run. Usage: 20_train.sh RUN NUM_DEMOS DEMO_MANIFEST [DEMO_H5]
source "$(dirname "$0")/common.sh"
RUN=$1; N=$2; MANIFEST=$3; H5=${4:-$DEMO}
run train_${RUN} python scripts/03_train_dp.py --config $C --demo "$H5" --run $RUN --num-demos $N --total-iters 100000 \
    --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 --data-manifest $MANIFEST --execute
