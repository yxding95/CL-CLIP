#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
    --method "agem" \
    --part "io" \
    --epochs 1 \
    --lr 1e-6 \
    --batch 100 \
    --workers 4 \
    --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
    --seed 10 \
    --logging_dir "./results/logs_0/" \
