#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_6/(v7)/lora/pet-ensembled/dtd/3-shot/dinov2/log/lora_pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list pgd \
    --eps_list 0.03137254901960784 \
    --attack_out lora_pet_pgd.csv
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled/dtd/3-shot/dinov2/log/lora_pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list square \
    --eps_list 0.03137254901960784 \
    --square_queries 1000 \
    --square_p_init 0.05 \
    --attack_out lora_pet_square.csv
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled/dtd/3-shot/dinov2/log/lora_pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list hopskipjump \
    --eps_list 0.03137254901960784 \
    --hsja_num_iterations 20 \
    --hsja_max_num_evals 200 \
    --hsja_init_num_evals 50 \
    --hsja_init_trials 100 \
    --attack_out lora_pet_hsja.csv
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_6/(v7)/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/log/lora_v-pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list pgd \
    --eps_list 0.03137254901960784 \
    --attack_out lora_v-pet_pgd.csv
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/log/lora_v-pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list square \
    --eps_list 0.03137254901960784 \
    --square_queries 1000 \
    --square_p_init 0.05 \
    --attack_out lora_v-pet_square.csv
python3 infer_adversarial.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/log/lora_v-pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_attacks \
    --attack_list hopskipjump \
    --eps_list 0.03137254901960784 \
    --hsja_num_iterations 20 \
    --hsja_max_num_evals 200 \
    --hsja_init_num_evals 50 \
    --hsja_init_trials 100 \
    --attack_out lora_v-pet_hsja.csv
