import argparse
import os
import random
import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn

from models.clip import CLIP
from models.clip_rkr import CLIP_RKR, _find_modules_v2
from data import OST, RetriTask, ZSTask, MST, Replayed, Replayed_MST
from methods import FT, LwF, IMM, GeoDL, VRD, AGEM

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda")

def random_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    random_seed(args.seed)
    os.makedirs(args.logging_dir, exist_ok=True)

    model_path = args.pretrained_path
    state_dict = torch.jit.load(model_path).state_dict()
    input_resolution = state_dict['input_resolution']
    context_length = state_dict['context_length']
    vocab_size = state_dict['vocab_size']

    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    if args.method not in ["rkr", "workr"]:
        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        ).to(device)
            
        ###load_pretrain###
        for key in ["input_resolution", "context_length", "vocab_size"]:
            del state_dict[key]
        model.load_state_dict(state_dict)
        model.train()

        text_trans = []
        text_pre = ["token_embedding.weight", "positional_embedding"]
        text_post = ["ln_final.weight", "ln_final.bias", "text_projection", "logit_scale"]
        visual_trans = []
        visual_pre = ["visual.class_embedding", "visual.positional_embedding", "visual.conv1.weight", "visual.ln_pre.weight", "visual.ln_pre.bias"]
        visual_post =  ["visual.ln_post.weight", "visual.ln_post.bias", "visual.proj", "logit_scale"]   
        
        if args.only_part_layers > 0:
            for name, param in model.named_parameters():
                split_name = name.split('.')
                if len(split_name)>3 and split_name[0]=='visual' and int(split_name[3])>args.only_part_layers:
                    visual_trans.append(name)
                elif len(split_name)>2 and split_name[0]=='transformer' and int(split_name[2])>args.only_part_layers:
                    text_trans.append(name)
            if args.part == "wm":
                learning_param = text_trans + visual_trans + text_post + visual_post
            elif args.part == "io":
                learning_param = visual_trans + visual_post
            elif args.part == "to":
                learning_param = text_trans + text_post
        else:
            for name, param in model.named_parameters():
                split_name = name.split('.')
                if len(split_name)>3 and split_name[0]=='visual' and int(split_name[3])>=0:
                    visual_trans.append(name)
                elif len(split_name)>2 and split_name[0]=='transformer' and int(split_name[2])>=0:
                    text_trans.append(name)
            if args.part == "wm":
                learning_param = text_trans + visual_trans + text_post + visual_post + text_pre + visual_pre
            elif args.part == "io":
                learning_param = visual_trans + visual_post + visual_pre
            elif args.part == "to":
                learning_param = text_trans + text_post + text_pre

        for name, param in model.named_parameters():
            if name not in learning_param:
                param.requires_grad_(False)

        teacher = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        ).to(device)
        ###load_pretrain###
        teacher.load_state_dict(state_dict)
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
    else:
        teacher = None
        model = CLIP_RKR(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        ).to(device)
        ###load_pretrain###
        for key in ["input_resolution", "context_length", "vocab_size"]:
            del state_dict[key]
        model.load_state_dict(state_dict)
        model.train()
        for name, param in model.named_parameters():
            param.requires_grad_(False)
        
        if args.part == "wm":
            target_replace_module = {"VisualTransformer", "Transformer"}
        elif args.part == "to":
            target_replace_module = {"Transformer",}
        elif args.part == "io":
            target_replace_module = {"VisualTransformer",}
            
        resume_ckpt = None
        for pt_file in os.listdir(args.logging_dir):
            if os.path.splitext(pt_file)[1] == ".pt":
                resume_ckpt = args.logging_dir + pt_file
        if resume_ckpt is not None:
            checkpoint = torch.load(resume_ckpt)
            loras = checkpoint["model"]
        else:
            loras = None
        if args.method == "rkr":
            require_grad_params, names = model.inject_trainable_rkr(target_replace_module=target_replace_module, loras=loras)
        elif args.method == "workr":
            require_grad_params, names = model.inject_trainable_rkr(target_replace_module=target_replace_module, loras=loras, woLora=True)
        
    batch_size = args.batch
    num_workers = args.workers

    # if args.mode == "ost":
    #     trainset = OST(args.update_data, args.update_img, args.pseudo_cls, args.pseudo_length, split='train')
    #     _, categories_name = trainset.get_ClsName()
    #     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    # else:
    #     pass
    #     #trainset = MST(args.update_data, args.update_img, phase=0, pseudo_cls=args.pseudo_cls, pseudo_length=args.pseudo_length)
    #     #train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    testset = OST(args.update_data, args.update_img, split='test')
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    coco_retri = RetriTask(args.retri_coco_data, args.retri_coco_img)
    coco_loader = DataLoader(coco_retri, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    flickr_retri = RetriTask(args.retri_flickr_data, args.retri_flickr_img)
    flickr_loader = DataLoader(flickr_retri, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    test_zs = ZSTask(args.zs_data, args.zs_img)
    zs_loader = DataLoader(test_zs, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    if args.method in ["gem", "agem"]:
        replayset = Replayed(args.replay_data)
        replay_loader = DataLoader(replayset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    else:
        replay_loader = None

    global_step = 0
    writer = SummaryWriter(args.logging_dir)
    if args.mode == "ost":
        trainset = OST(args.update_data, args.update_img, args.pseudo_cls, args.pseudo_length, split='train')
        _, categories_name = trainset.get_ClsName()
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

        input_params = {
            "train_epoch": args.epochs,
            "global_step": global_step,
            "train_loader": train_loader,
            "replay_loader": replay_loader,
            "test_loader": test_loader,
            "coco_loader": coco_loader,
            "flickr_loader": flickr_loader,
            "zs_loader": zs_loader,
            "writer": writer,
            "embed_dim": embed_dim,
            "model": model,
            "teacher": teacher,
            "lr": args.lr,
            "method": args.method,
            "mode": args.mode,
            "part": args.part,
            "logging_dir": args.logging_dir,
        }

        if args.method == "ft":
            method = FT(input_params)
            model = method.train()
        elif args.method == "lwf":
            method = LwF(input_params)
            model = method.train(args.alpha)
        elif args.method == "imm":
            method = IMM(input_params)
            model = method.train(args.alpha)
        elif args.method == "geodl":
            method = GeoDL(input_params)
            model = method.train(args.alpha)
        elif args.method == "rkr":
            method = FT(input_params)
            model = method.train(require_grad_params)
        elif args.method == "workr":
            method = FT(input_params)
            model = method.train(require_grad_params)
        elif args.method == "vrd":
            method = VRD(input_params)
            model = method.train(args.alpha)
        elif args.method == "agem":
            method = AGEM(input_params)
            model = method.train()
        
    else:
        phase_matrix = np.zeros((8, 8))
        resume_ckpt = None
        for pt_file in os.listdir(args.logging_dir):
            if os.path.splitext(pt_file)[1] == ".pt":
                resume_ckpt = args.logging_dir + pt_file
        if resume_ckpt is not None:
            checkpoint = torch.load(resume_ckpt)
            start_phase = checkpoint["phase"]
            phase_matrix = checkpoint["phase_matrix"]
        else:
            start_phase = -1
        for phase in range(start_phase+1, phase_matrix.shape[0]):
            trainset = MST(args.update_data, args.update_img, phase=phase, pseudo_cls=args.pseudo_cls, pseudo_length=args.pseudo_length)
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

            input_params = {
                "train_epoch": args.epochs,
                "global_step": global_step,
                "train_loader": train_loader,
                "replay_loader": replay_loader,
                "test_loader": test_loader,
                "coco_loader": coco_loader,
                "flickr_loader": flickr_loader,
                "zs_loader": zs_loader,
                "writer": writer,
                "embed_dim": embed_dim,
                "model": model,
                "teacher": teacher,
                "phase": phase,
                "phase_matrix": phase_matrix,
                "lr": args.lr,
                "method": args.method,
                "mode": args.mode,
                "part": args.part,
                "logging_dir": args.logging_dir,
            }

            if args.method == "ft":
                method = FT(input_params)
                model, global_step, _ = method.train()
                continue
            elif args.method == "lwf":
                method = LwF(input_params)
                model, global_step, _ = method.train(args.alpha)
            elif args.method == "imm":
                method = IMM(input_params)
                model, global_step, _ = method.train(args.alpha)
            elif args.method == "geodl":
                method = GeoDL(input_params)
                model, global_step, _ = method.train(args.alpha)
            elif args.method == "rkr":
                method = FT(input_params)
                model, global_step, _ = method.train(require_grad_params)
                continue
            elif args.method == "workr":
                method = FT(input_params)
                model, global_step, _ = method.train(require_grad_params)
                continue
            elif args.method == "vrd":
                method = VRD(input_params)
                model, global_step, _ = method.train(args.alpha)
            elif args.method == "agem":
                method = AGEM(input_params)
                model, global_step, _ = method.train()
            
            teacher = copy.deepcopy(model)
            for param in teacher.parameters():
                param.requires_grad_(False)
            teacher.eval()
            if replay_loader is not None:
                add_data = Replayed_MST(args.update_data, args.update_img, phase=phase)
                replayset = ConcatDataset((replayset, add_data))
                replay_loader = DataLoader(replayset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=["ost", "mst"])
    parser.add_argument('--method', required=True, choices=["ft", "lwf", "geodl", "imm", "rkr", "vrd", "agem", "workr"])
    parser.add_argument('--part', required=True, choices=["wm", "io", "to"])
    parser.add_argument('--update_img', type=str, default='/home/shared/coco/')
    parser.add_argument('--update_data', type=str, default='/home/yxding/MSCOCO/annotations/crop_obj_ids_phase.json')
    parser.add_argument('--zs_img', type=str, default='/media/disk1/yxding/tiered-imagenet-tools-master/')
    parser.add_argument('--zs_data', type=str, default='../FS_json/ZS_train_test.json')
    parser.add_argument('--retri_coco_img', type=str, default='/home/shared/coco/')
    parser.add_argument('--retri_coco_data', type=str, default='../FS_json/retri_test.json')
    parser.add_argument('--retri_flickr_img', type=str, default='/home/shared/flickr30k-images/')
    parser.add_argument('--retri_flickr_data', type=str, default='../FS_json/flickr_retri_test.json')
    parser.add_argument('--replay_data', type=str, default='/home/shared/LAION_sub/')

    parser.add_argument('--pretrained_path', type=str, default='/home/yxding/.cache/clip/ViT-B-32.pt')

    parser.add_argument('--only_part_layers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=10, help="A seed for reproducible training.")
    parser.add_argument("--logging_dir", type=str, default="./logs/")
    args = parser.parse_args()

    if args.method == "vrd":
        args.pseudo_cls = 100
        args.pseudo_length = 10
    else:
        args.pseudo_cls = None
        args.pseudo_length = None

    dir_splits = args.logging_dir.split("/")
    if dir_splits[-1] != "":
        dir_splits[-1] = f"{args.method}_{args.part}_{args.mode}_{dir_splits[-1]}"
    else:
        dir_splits[-2] = f"{args.method}_{args.part}_{args.mode}_{dir_splits[-2]}"
    args.logging_dir = "/".join(dir_splits)
    
    main(args)
    print("Finish training ...")


