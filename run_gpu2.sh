#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "ft" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "lwf" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.5

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "geodl" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 10

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "imm" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.6

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "rkr" \
    --part "io" \
    --epochs 15 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.6

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "agem" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 1

CUDA_VISIBLE_DEVICES=2 python main_EuroSAT.py --mode "ost" \
    --method "vrd" \
    --part "io" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 1

