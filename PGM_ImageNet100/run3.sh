#!/bin/bash

# ---------- CPU threading ----------
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# ---------- 启动 ----------
python outer_in100.py \
  --data_dir /root/autodl-tmp/data/imagenet100 \
  --runs_dir /root/autodl-tmp/runs_main3 \
  --db /root/autodl-tmp/runs_main3/pgm_main.db \
  --exp_name pgm_main3 \
  --seeds 81 82 83 \
  --epochs 5 \
  --confirm_epochs 10 \
  --confirm_n_seeds 10 \
  --budget 6 \
  --max_procs 2 \
  --num_workers 6 \
  --confirm_mean_pp 1.0 \
  --rmax_pp 2.0

