#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --mode "mst" \
    --method "rkr" \
    --part "wm" \
    --epochs 1 \
    --batch 100 \
    --workers 4 \
    --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids_phase.json" \
    --seed 10 \
    --logging_dir "./results/logs_0/"

# CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
#     --method "geodl" \
#     --part "wm" \
#     --epochs 2 \
#     --batch 100 \
#     --workers 4 \
#     --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
#     --seed 10 \
#     --logging_dir "./results/logs_0/"

# CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
#     --method "ft" \
#     --part "wm" \
#     --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
#     --seed 20 \
#     --logging_dir "./results/logs_ost_1/" 

# CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
#     --method "ft" \
#     --part "wm" \
#     --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
#     --seed 30 \
#     --logging_dir "./results/logs_ost_2/" 

# CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
#     --method "ft" \
#     --part "wm" \
#     --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
#     --seed 40 \
#     --logging_dir "./results/logs_ost_3/" 

# CUDA_VISIBLE_DEVICES=1 python main.py --mode "ost" \
#     --method "ft" \
#     --part "wm" \
#     --update_data "/home/yxding/MSCOCO/annotations/crop_obj_ids.json" \
#     --seed 50 \
#     --logging_dir "./results/logs_ost_4/" 