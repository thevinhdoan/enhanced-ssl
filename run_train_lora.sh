#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python3 train.py \
  --c config/lora/pet-ensembled/dtd/3-shot/dinov2/config.yaml \
  --lambda_1 0.01 \
  --lambda_2 0.1 \
  --log_partition_stats \
  --grouping_update_interval 5 \
  --uratio 12
python3 train.py \
  --c config/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/config.yaml \
  --lambda_1 0.01 \
  --lambda_2 0.1 \
  --log_partition_stats \
  --grouping_update_interval 5 \
  --uratio 12