#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py --mode "mst" \
    --method "workr" \
    --part "to" \
    --epochs 10 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids_phase.json" \
    --seed 10 \
    --logging_dir "./results/logs_0/" \
    --alpha 1

CUDA_VISIBLE_DEVICES=2 python main.py --mode "mst" \
    --method "workr" \
    --part "to" \
    --epochs 10 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids_phase.json" \
    --seed 20 \
    --logging_dir "./results/logs_1/" \
    --alpha 1

CUDA_VISIBLE_DEVICES=2 python main.py --mode "mst" \
    --method "workr" \
    --part "to" \
    --epochs 10 \
    --lr 1e-5 \
    --batch 100 \
    --workers 4 \
    --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids_phase.json" \
    --seed 30 \
    --logging_dir "./results/logs_2/" \
    --alpha 1
