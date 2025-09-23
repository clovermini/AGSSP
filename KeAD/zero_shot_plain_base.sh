#!/bin/bash

few_shots=(0 1 2 4)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD 
    des_path=./data/text_description/datasets_des_info.json 
    meta_path=./data/dataset_to_public.json
    surgery_type=vv_res
    dataset_name=metal_own
    data_path=/data/datasets/pub/public/own_anomaly_detect

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_base_16_240_few_shot_${few_shots[few_num]}_${surgery_type}/

    CUDA_VISIBLE_DEVICES=2 python -u main/get_anomaly_map_base.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path}\
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-B-16-plus-240 --pretrained laion400m_e32 --k_shot ${few_shots[few_num]} \
    --image_size 240 --patch_size 16 --feature_list 3 6 9 12 --dpam_layer 10 \
    --surgery_type ${surgery_type} --use_detailed --visualize --save_anomaly_map
    wait
done
