#!/bin/bash

# =========================
# Common config
# =========================
DATA_DIR=/root/autodl-tmp/data/imagenet100
RUNS_DIR=/root/autodl-tmp/runs_long

SEEDS="41 42 43"

EPOCHS=5
CONFIRM_EPOCHS=10
CONFIRM_N_SEEDS=12
BUDGET=40

MAX_PROCS=2
NUM_WORKERS=6
RMAX_PP=2.0
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

# =========================
# 1. SGM (MAIN EXPERIMENT)
# =========================
python outer_in100_long.py \
  --data_dir $DATA_DIR \
  --runs_dir $RUNS_DIR \
  --db $RUNS_DIR/long_sgm.db \
  --exp_name long_sgm \
  --seeds $SEEDS \
  --epochs $EPOCHS \
  --confirm_epochs $CONFIRM_EPOCHS \
  --confirm_n_seeds $CONFIRM_N_SEEDS \
  --budget $BUDGET \
  --max_procs $MAX_PROCS \
  --num_workers $NUM_WORKERS \
  --rmax_pp $RMAX_PP \
  --policy sgm \
  --no_early_stop


# =========================
# 2. Naive baseline
# =========================
# UNCOMMENT after SGM finishes
# python outer_in100_long.py \
#   --data_dir $DATA_DIR \
#   --runs_dir $RUNS_DIR \
#   --db $RUNS_DIR/long_naive.db \
#   --exp_name long_naive \
#   --seeds $SEEDS \
#   --epochs $EPOCHS \
#   --confirm_epochs $CONFIRM_EPOCHS \
#   --confirm_n_seeds $CONFIRM_N_SEEDS \
#   --budget $BUDGET \
#   --max_procs $MAX_PROCS \
#   --num_workers $NUM_WORKERS \
#   --rmax_pp $RMAX_PP \
#   --policy naive_screen \
#   --no_early_stop


# =========================
# 3. Best-screen baseline
# =========================
# UNCOMMENT last
# python outer_in100_long.py \
#   --data_dir $DATA_DIR \
#   --runs_dir $RUNS_DIR \
#   --db $RUNS_DIR/long_best.db \
#   --exp_name long_best \
#   --seeds $SEEDS \
#   --epochs $EPOCHS \
#   --confirm_epochs $CONFIRM_EPOCHS \
#   --confirm_n_seeds $CONFIRM_N_SEEDS \
#   --budget $BUDGET \
#   --max_procs $MAX_PROCS \
#   --num_workers $NUM_WORKERS \
#   --rmax_pp $RMAX_PP \
#   --policy best_screen \
#   --no_early_stop