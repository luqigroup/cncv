#!/bin/bash
# Run Gaussian CNCV at d=32, 64, 128 with per-dimension hyperparameter tuning,
# checking that variance reduction persists as the dimension grows.

set -e
cd "$(dirname "$0")/.."
# Activate your Python environment (with cncv installed) before running.

LOG_DIR=logs/rebuttal_higher_d
mkdir -p $LOG_DIR

run_one() {
    local d=$1
    local L=$2
    local depth=$3
    local hidden=$4
    local n_layers=$5
    local epochs=$6
    local num_train=$7
    local lr=$8
    local lr_final=$9
    local qofi=${10}

    echo "==============================================="
    echo "Starting d=$d qofi=$qofi: L=$L, depth=$depth, hidden=$hidden, n_layers=$n_layers, epochs=$epochs, num_train=$num_train"
    echo "==============================================="
    date

    python scripts/gaussian_ensemble_cv.py \
        --input_dim $d \
        --qofi $qofi \
        --layer_type shared_hierarchical \
        --depth $depth \
        --n_ensemble_members $L \
        --n_hidden $hidden \
        --n_layers $n_layers \
        --max_epochs $epochs \
        --num_train $num_train \
        --num_val 4096 \
        --num_samples 10000 \
        --batchsize 2048 \
        --lr $lr \
        --lr_final $lr_final \
        --sigma 0.3 \
        --gpu_id 0 \
        --seed 12 \
        --save_freq $epochs \
        --phase train \
        --testing_epoch -1 \
        --ddpm 0 \
        --grad_clip 1.0 \
        2>&1 | tee $LOG_DIR/d${d}_${qofi}.log

    echo "Finished d=$d qofi=$qofi at $(date)"
}

# Mimic the Darcy d=100 recipe (HANDOFF.md): depth=3 (not auto), L=16, mlp=5, lr=1e-4.
# Scale hidden width with d.
# d=32:  L=16, depth=3, hidden=128, n_layers=5, epochs=200, lr=1e-4 -> 1e-5
run_one 32 16 3 128 5 200 131072 0.0001 0.00001 mean

# d=64:  L=16, depth=3, hidden=192, n_layers=5, epochs=300, lr=1e-4 -> 1e-5
run_one 64 16 3 192 5 300 262144 0.0001 0.00001 mean

# d=128: L=16, depth=3, hidden=256, n_layers=5, epochs=400, lr=1e-4 -> 1e-5
run_one 128 16 3 256 5 400 262144 0.0001 0.00001 mean

echo ""
echo "==============================================="
echo "All runs complete at $(date)"
echo "==============================================="
