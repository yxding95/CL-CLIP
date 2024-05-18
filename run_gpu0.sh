#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "ft" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "lwf" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.5

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "geodl" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 10

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "imm" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.6

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "rkr" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 0.6

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "agem" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2

CUDA_VISIBLE_DEVICES=0 python main_EuroSAT.py --mode "ost" \
    --method "vrd" \
    --part "wm" \
    --epochs 15 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --pretrained_path "/home/yxding/.cache/clip/ViT-B-32.pt" \
    --update_img "/home/shared/EuroSAT/" \
    --seed 10 \
    --logging_dir "../CL-CLIP_results/sat_results/sat_logs/" \
    --alpha 2