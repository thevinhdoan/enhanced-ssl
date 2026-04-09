#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
set -euo pipefail

PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

DATASET="kitti"
COUNT=5
CLASS0=0
CLASS1=1
LAMBDA_1=0.01
LAMBDA_2=0.1
GROUPING_UPDATE_INTERVAL=5
URATIO=8

for RATIO in 6 8; do
  "${PYTHON_BIN}" run_two_class_imbalance_train.py \
    --base-config "config/adaptformer/pet-ensembled/${DATASET}/${COUNT}-shot/dinov2/config.yaml" \
    --classes "${CLASS0}" "${CLASS1}" \
    --ratio "${RATIO}" \
    --lambda-1 "${LAMBDA_1}" \
    --lambda-2 "${LAMBDA_2}" \
    --grouping-update-interval "${GROUPING_UPDATE_INTERVAL}" \
    --uratio "${URATIO}" \
    --log-partition-stats

  "${PYTHON_BIN}" run_two_class_imbalance_train.py \
    --base-config "config/adaptformer/pet-ensembled-across-nets/${DATASET}/${COUNT}-shot/dinov2/config.yaml" \
    --classes "${CLASS0}" "${CLASS1}" \
    --ratio "${RATIO}" \
    --lambda-1 "${LAMBDA_1}" \
    --lambda-2 "${LAMBDA_2}" \
    --grouping-update-interval "${GROUPING_UPDATE_INTERVAL}" \
    --uratio "${URATIO}" \
    --log-partition-stats
done
