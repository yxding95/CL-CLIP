#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "ft" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 2 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2

