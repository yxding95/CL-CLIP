#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "ft" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "lwf" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.1

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "geodl" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.3

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "imm" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.7

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "rkr" \
    --part "to" \
    --epochs 15 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.6

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "agem" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 1

CUDA_VISIBLE_DEVICES=1 python main_EuroSAT.py --mode "ost" \
    --method "vrd" \
    --part "to" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 1

