export CUDA_VISIBLE_DEVICES=0
python3 infer_corrupted.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/adaptformer/pet-ensembled/dtd/3-shot/dinov2/log/adaptformer_pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_corruptions \
    --corruption_out adaptformer_pet.csv
python3 infer_corrupted.py \
    --checkpoint "/mnt/extra_storage/users/vinhdt/thesis/archive/.saved_models_7/(v2)/adaptformer/pet-ensembled-across-nets/dtd/3-shot/dinov2/log/adaptformer_v-pet.pth" \
    --dataset dtd \
    --num_classes 47 \
    --num_labels 141 \
    --net vit_large_patch14_reg4_dinov2.lvd142m \
    --data_zip "/mnt/extra_storage/users/vinhdt/thesis/zip/data.zip" \
    --eval_corruptions \
    --corruption_out adaptformer_v-pet.csv