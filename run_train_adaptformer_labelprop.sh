#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python3 train_labelprop.py --config config/adaptformer/pet-ensembled/dtd/3-shot/dinov2/config.yaml
python3 train_labelprop.py --config config/adaptformer/pet-ensembled/dtd/3-shot/clip/config.yaml
python3 train_labelprop.py --config config/adaptformer/pet-ensembled/kitti/5-shot/dinov2/config.yaml
python3 train_labelprop.py --config config/adaptformer/pet-ensembled/kitti/5-shot/clip/config.yaml
