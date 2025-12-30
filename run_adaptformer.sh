#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python3 train.py \
  --c config/adaptformer/pet-ensembled/dtd/3-shot/dinov2/config.yaml \
  --lambda_1 0.0 \
  --lambda_2 0.0 \
  --grouping_update_interval 2 \
  --uratio 12
python3 train.py \
  --c config/adaptformer/pet-ensembled-across-nets/dtd/3-shot/dinov2/config.yaml \
  --lambda_1 0.0 \
  --lambda_2 0.0 \
  --grouping_update_interval 2 \
  --uratio 12