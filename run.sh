#!/usr/bin/env bash

python3 train.py \
  --c config/lora/pet-ensembled/dtd/3-shot/dinov2/config.yaml \
  --lambda_1 0.01 \
  --lambda_2 0.05 \
  --grouping_update_interval 2 \
  --uratio 4
