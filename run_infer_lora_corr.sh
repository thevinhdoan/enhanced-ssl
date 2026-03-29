export CUDA_VISIBLE_DEVICES=1
python3 infer_corrupted.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled/dtd/3-shot/dinov2/log/lora_pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_corruptions \
    --corruption_out lora_pet.csv
python3 infer_corrupted.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/log/lora_v-pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_corruptions \
    --corruption_out lora_v-pet.csv