import warnings

# Must run before importing datasets/SAM/timm. DataLoader workers import this
# module independently, so a late filter prints the same deprecation repeatedly.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"timm(\..*)?",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.utils\.checkpoint: the use_reentrant parameter should be passed explicitly.*",
)

import transformers
import importlib.util
from pathlib import Path
from datasets.dataset_refavs_ALS3P import REFAVSALS3P

_config_path = (
    Path(__file__).resolve().parent
    / "configs"
    / "config_ALS3P.py"
)
_config_spec = importlib.util.spec_from_file_location(
    "config_ALS3P", _config_path
)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Unable to load ALS3P config: {_config_path}")
_config_module = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config_module)
args = _config_module.args
from torch.utils.data import DataLoader
from functools import partial
from models.ALS3P import ALS3PForCausalLM

import torch
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from utils import utility
import random
import numpy as np
import re
import os
import gc
import math


from transformers import logging
logging.set_verbosity_error()


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

AUDIO_TOKEN_INDEX = -300
GROUNDING_TOKEN_INDEX = -400

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
                isinstance(input_dict[k], list)
                and len(input_dict[k]) > 0
                and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

def tokenizer_image_audio_token(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        audio_token_index=AUDIO_TOKEN_INDEX,
        grounding_token_index=GROUNDING_TOKEN_INDEX,
        num_frames=10,
        return_tensors=None,
):

    prompt_chunks = re.split(r'(<grounding>|<image>|<audio>|<video>)', prompt)


    prompt_chunks = [chunk for chunk in prompt_chunks if chunk]

    text_chunks = []
    token_types = []
    for chunk in prompt_chunks:
        if chunk == "<image>":
            token_types.append("image")
        elif chunk == "<audio>":
            token_types.append("audio")
        elif chunk == "<video>":
            token_types.append("video")
        elif chunk == "<grounding>":
            token_types.append("grounding")
        else:
            text_chunks.append(chunk)

    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in text_chunks]

    def insert_separators(
            text_chunks,
            tokenized_chunks,
            token_types,
            image_token_index,
            audio_token_index,
            grounding_token_index,
            num_frames,
    ):
        input_ids = []
        offset = 0
        if (
                len(tokenized_chunks) > 0
                and len(tokenized_chunks[0]) > 0
                and tokenized_chunks[0][0] == tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(tokenized_chunks[0][0])

        min_length = min(len(text_chunks), len(token_types))
        for i in range(min_length):

            input_ids.extend(tokenized_chunks[i][offset:])

            if token_types[i] == "image":
                input_ids.append(image_token_index)
            elif token_types[i] == "audio":
                input_ids.append(audio_token_index)
            elif token_types[i] == "video":
                input_ids.extend([image_token_index] * num_frames)
            elif token_types[i] == "grounding":
                input_ids.append(grounding_token_index)


        if len(text_chunks) > min_length:
            input_ids.extend(tokenized_chunks[min_length][offset:])

        return input_ids

    input_ids = insert_separators(
        text_chunks,
        tokenized_chunks,
        token_types,
        image_token_index,
        audio_token_index,
        grounding_token_index,
        num_frames,
    )

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

def collate_fn(batch, tokenizer=None):
    vids = []
    images = []
    image_clips = []
    image_grounding = []
    grounding_valid_sizes = []
    masks = []
    conversations = []
    audio_feats = []
    image_feats = []
    resizes = []
    orgsizes = []
    refs = []
    refs_num = []
    fids = []


    for data in batch:
        vids.append(data['vid'])
        images.append(data['image'])
        image_clips.append(data['img_clip'])
        image_grounding.append(data['img_grounding'])
        grounding_valid_sizes.append(data['grounding_valid_sizes'])
        masks.append(data['mask'])
        conversations.append(data['conversation'])
        audio_feats.append(data['feat_aud'])
        resizes.append(data['resize'])
        orgsizes.append(data['orgsize'])
        image_feats.append(data['feat_sam'])
        refs_num.append(len(data['ref']))
        fids.append(data['fids'])

        refs.append(data['ref'][0])


    input_ids = [tokenizer_image_audio_token(conv, tokenizer, return_tensors="pt") for conv in conversations]  # list
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    ref_ids = [tokenizer_image_audio_token(ref, tokenizer, return_tensors="pt") for ref in refs]

    labels = input_ids.clone()

    sep = 'Sure, It is [SEG]'

    for conversation, target in zip(conversations, labels):

        parts = conversation.split(sep)
        # print(parts)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        sep_len = len(tokenizer_image_audio_token(sep, tokenizer)) - 1


        for i in range(len(parts)-1):
            part_len = len(tokenizer_image_audio_token(parts[i], tokenizer)) - 2
            target[cur_len: cur_len + part_len] = IGNORE_INDEX
            cur_len += part_len + sep_len

        target[cur_len:] = IGNORE_INDEX


    return {"vids": vids,
            "images": images,  # list[B]:[T, 3, 1024, 1024]
            "images_clip": image_clips,  # list[B]:[T, 3, 224, 224]
            "images_grounding": image_grounding,
            "grounding_valid_sizes": grounding_valid_sizes,
            "masks": masks,  # list[B]:[num_ref, T, H, W]
            "convs": conversations,  # list[B]: str
            "input_ids": input_ids,  # list[B]:[max_len]
            "attention_masks": attention_masks,  # list[B]:[max_len]
            "labels": labels,  # list[B]:[max_len]
            "audio_feats": audio_feats,  # list[B]:[10, 128]
            "resizes": resizes,  # list[B]
            "orgsizes": orgsizes,  # list[B]
            "image_feats": image_feats,
            "ref_ids": ref_ids,  # list[B]: [ref_id_len]
            "refs_num": refs_num,
            "fids": fids
    }


import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn")
    set_seed(42)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mllm,
        cache_dir=None,
        model_max_length=2048,  # 2048
        padding_side="right",
        use_fast=False,
    )   #加载一个预训练的语言模型的分词器（Tokenizer）

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]  # 32000
    print("seg_token_idx: ", seg_token_idx)

    train_dataset = REFAVSALS3P('train', args, tokenizer, input_type='refer')
    val_dataset_refer = REFAVSALS3P('val', args, tokenizer, input_type='refer')
    val_dataset_s_refer = REFAVSALS3P('test_s', args, tokenizer, input_type='refer')
    val_dataset_u_refer = REFAVSALS3P('test_u', args, tokenizer, input_type='refer')
    val_dataset_n_refer = REFAVSALS3P('test_n', args, tokenizer, input_type='refer')


    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker,collate_fn=partial(collate_fn, tokenizer=tokenizer), generator=g)

    eval_batch_size = args.eval_batch_size
    val_dataloader_refer = DataLoader(val_dataset_refer, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_s_refer = DataLoader(val_dataset_s_refer, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_u_refer = DataLoader(val_dataset_u_refer, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_n_refer = DataLoader(val_dataset_n_refer, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))


    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,  # 256
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,  # sam_vit_h_xxx.pth
        "vision_tower": args.vision_tower,
        "use_im_start_end": False,
        "compress": args.compress,
        "start": args.start,
        "uground_layer_strategy": args.layer_strategy,
        "uground_candidate_layers": args.candidate_layers,
        "uground_eval_layer_strategy": args.eval_layer_strategy,
        "uground_selector_temperature": args.selector_temperature_start,
        "uground_selector_target_temperature": args.selector_target_temperature,
        "uground_selector_standardize_targets": False,
        "uground_selector_warmup_epochs": args.selector_warmup_epochs,
        "uground_selector_loss_weight": args.selector_loss_weight,
        "uground_map_loss_weight": args.map_loss_weight,
        "uground_map_bce_weight": args.map_bce_weight,
        "uground_map_dice_weight": args.map_dice_weight,
        "uground_gaussian_kernel": args.gaussian_kernel,
        "uground_gaussian_sigma": args.gaussian_sigma,
        "uground_temperature": args.masp_temperature,
        "uground_grounding_grid_size": args.grounding_grid_size,
        "uground_max_sequence_length": args.max_sequence_length,
        "uground_return_similarity_maps": args.return_similarity_maps,
    }

    print(
        "ALS3P configuration: "
        f"grounding_image_size={args.grounding_image_size}, "
        f"grid={args.grounding_grid_size}, "
        f"layer_strategy={args.layer_strategy}, "
        f"map_loss_weight={args.map_loss_weight}, "
        f"selector_loss_weight={args.selector_loss_weight}, "
        f"selector_warmup={args.selector_warmup_epochs}, "
        f"selector_lr={args.selector_lr}, "
        f"target_temperature={args.selector_target_temperature}, "
        "all_candidate_supervision=True, reinforce=False, "
        "valid_normalization=True, grounding_attention_mask=True, "
        "separate_grounding_projector=True, train_prompt_encoder=True"
    )

    model = ALS3PForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print("\nmodel loaded")

    model.layer_selector.reset_parameters()
    model.mask_as_prompt.reset_parameters(args.masp_temperature)
    nonfinite_uground = model.layer_selector.nonfinite_parameter_names()
    if nonfinite_uground:
        raise RuntimeError(
            "ALS3P initialization produced invalid parameters: "
            f"{nonfinite_uground}"
        )
    print("ALS3P parameters explicitly initialized: all finite")

    model.config.eos_token_id = tokenizer.eos_token_id   # end of sentence token id
    model.config.bos_token_id = tokenizer.bos_token_id   # beginning of sentence token id
    model.config.pad_token_id = tokenizer.pad_token_id   # padding token id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.float32, device="cuda")

    model_args_from_pt = AutoConfig.from_pretrained(args.mllm)
    model_args_from_pt.use_cluster = True
    model_args_from_pt.freeze = False
    model_args_from_pt.mm_tune = True
    model_args_from_pt.spatial_cluster_rate0 = 64
    model_args_from_pt.spatial_cluster_rate1 = 32
    model_args_from_pt.spatial_cluster_rate2 = 16
    model_args_from_pt.temporal_cluster_rate = 0.0625
    model_args_from_pt.use_cluster = True
    model_args_from_pt.vision_tune = False
    model.get_model().initialize_cluster_modules(model_args_from_pt)

    model.get_model().initialize_lisa_modules(model.get_model().config)
    model.get_model().initialize_grounding_projector_from_mm()
    print("Grounding projector initialized from the pretrained mm_projector")

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_r = 8
    target_modules = "q_proj,v_proj"
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()

            for name, module in model.named_modules():
                if (

                        isinstance(module, cls)

                        and all(
                    [
                        x not in name
                        for x in [
                        "visual_model",
                        "vision_tower",
                        "mm_projector",
                        "text_hidden_fcs",
                            "audio_feature_layer",
                            "layer_selector",
                            "mask_as_prompt",
                    ]
                    ]
                )

                        and any([x in name for x in lora_target_modules])
                ):

                    lora_module_names.add(name)
            return sorted(list(lora_module_names))


        lora_alpha = 16
        lora_dropout = 0.05

        lora_target_modules = find_linear_layers(
            model, target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print("\nLora deployed")

        model.print_trainable_parameters()

        # LoRA 权重默认是 float32，需要手动转
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)


    model.resize_token_embeddings(len(tokenizer))

    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            raise FileNotFoundError(args.init_checkpoint)
        print(f"Loading baseline initialization: {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        incompatible = model.load_state_dict(checkpoint, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        selector_markers = [
            "layer_selector",
            "mask_as_prompt",
            "similarity_map_loss",
            "grounding_mm_projector",
        ]
        invalid_missing = [
            name for name in missing
            if not any(
                marker in name
                for marker in selector_markers
            )
        ]
        invalid_unexpected = [
            name for name in unexpected if "ppm" not in name
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Baseline checkpoint is incompatible. "
                f"Non-ALS3P missing keys: {invalid_missing}; "
                f"unexpected keys: {invalid_unexpected}"
            )
        if any("grounding_mm_projector" in name for name in missing):
            model.get_model().initialize_grounding_projector_from_mm()
            print(
                "Grounding projector reinitialized from the checkpoint's "
                "mm_projector"
            )
        print(
            f"Baseline loaded; {len(missing)} expected ALS3P parameters "
            "will be trained from initialization."
        )
        del checkpoint
        gc.collect()

    model = model.to("cuda")
    
    model = model.to(torch.bfloat16)

    model.layer_selector.to(device="cuda", dtype=torch.float32)
    model.mask_as_prompt.to(device="cuda", dtype=torch.float32)
    model.similarity_map_loss.to(device="cuda", dtype=torch.float32)

    # visual_model 和 vision_tower 保持 float32
    for name, module in model.named_modules():
        if "visual_model" in name or "vision_tower" in name:
            module.to(torch.float32)

    for name, param in model.audio_feature_layer.named_parameters():
        param.requires_grad = True
        # print(name, param.requires_grad)


    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            p.requires_grad = True

    for parameter in model.layer_selector.parameters():
        parameter.requires_grad = True
    for parameter in model.mask_as_prompt.parameters():
        parameter.requires_grad = True
    for parameter in model.get_model().grounding_mm_projector.parameters():
        parameter.requires_grad = True
    for parameter in model.get_model().visual_model.prompt_encoder.parameters():
        parameter.requires_grad = True

    def trainable_parameter_count(module):
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    print(
        "ALS3P trainable parameters: "
        f"grounding_projector={trainable_parameter_count(model.get_model().grounding_mm_projector)}, "
        f"prompt_encoder={trainable_parameter_count(model.get_model().visual_model.prompt_encoder)}, "
        f"mask_decoder={trainable_parameter_count(model.get_model().visual_model.mask_decoder)}"
    )


    print("will save train model")

    def evaluate_segmentation(model, dataloader, args, name):
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        total_iou = 0
        total_fscore = 0
        count = 0

        for batch in tqdm(dataloader, desc=f"Evaluating on {name}"):
            input_dict = dict_to_cuda(batch)
            with torch.inference_mode():
                output_dict = model.forward(images=input_dict["images"],
                                            images_clip=input_dict["images_clip"],
                                            images_grounding=input_dict["images_grounding"],
                                            grounding_valid_sizes=input_dict["grounding_valid_sizes"],
                                            audio_features=input_dict["audio_feats"],
                                            image_features=input_dict["image_feats"],
                                            input_ids=input_dict["input_ids"],
                                            labels=input_dict["labels"],
                                            attention_masks=input_dict["attention_masks"],
                                            masks_list=input_dict["masks"],
                                            resize_list=input_dict["resizes"],
                                            orgsize_list=input_dict["orgsizes"],
                                            conversation_list=input_dict["convs"],
                                            refs_num=input_dict["refs_num"],
                                            fids=input_dict["fids"],
                                            vids=input_dict["vids"],
                                            contrast=args.ct_weight,
                                            ref_ids=input_dict["ref_ids"],
                                            inference=True)
            pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
            gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
            for i in range(len(pred_masks)):
                num_seg = pred_masks[i].shape[0]
                T = pred_masks[i].shape[1]
                iou = utility.mask_iou(pred_masks[i], gt_masks[i])
                fscore = utility.Eval_Fmeasure(pred_masks[i], gt_masks[i], None)

                total_iou += iou * num_seg * T
                total_fscore += fscore * num_seg * T
                count += num_seg * T

            del input_dict, output_dict, pred_masks, gt_masks

        miou = total_iou / count
        fscore = total_fscore / count
        jf = (miou + fscore) / 2.0
        print(f"\n  evaluate on {name}: J: {miou}  F: {fscore}  J&F: {jf}")

        with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
            f.write(f"evaluate on {name}: J {miou}  F {fscore}  J&F {jf}\n")

        gc.collect()
        torch.cuda.empty_cache()
        return {"J": miou, "F": fscore, "J&F": jf}

    def evaluate_null(model, dataloader, args, name):
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        total_metric = 0
        count = 0

        for batch in tqdm(dataloader, desc=f"Evaluating on {name}"):
            input_dict = dict_to_cuda(batch)
            with torch.inference_mode():
                output_dict = model.forward(images=input_dict["images"],
                                            images_clip=input_dict["images_clip"],
                                            images_grounding=input_dict["images_grounding"],
                                            grounding_valid_sizes=input_dict["grounding_valid_sizes"],
                                            audio_features=input_dict["audio_feats"],
                                            image_features=input_dict["image_feats"],
                                            input_ids=input_dict["input_ids"],
                                            labels=input_dict["labels"],
                                            attention_masks=input_dict["attention_masks"],
                                            masks_list=input_dict["masks"],
                                            resize_list=input_dict["resizes"],
                                            orgsize_list=input_dict["orgsizes"],
                                            conversation_list=input_dict["convs"],
                                            refs_num=input_dict["refs_num"],
                                            fids=input_dict["fids"],
                                            vids=input_dict["vids"],
                                            contrast=args.ct_weight,
                                            ref_ids=input_dict["ref_ids"],
                                            inference=True)

            for pred_mask in output_dict["pred_masks"]:
                num_seg, num_frames = pred_mask.shape[:2]
                null_metric = utility.metric_s_for_null(pred_mask)
                total_metric += null_metric * num_seg * num_frames
                count += num_seg * num_frames

            del input_dict, output_dict

        metric_s = total_metric / count
        print(f"\n  evaluate on {name}: S: {metric_s}")
        with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
            f.write(f"evaluate on {name}: S {metric_s}\n")
        gc.collect()
        torch.cuda.empty_cache()
        return metric_s


    # ---------------train------------------------------------------

    model.train()
    epochs = args.epochs
    print("init lr:", args.lr)
    intended_trainable = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    selector_parameter_names = {
        name
        for name in intended_trainable
        if "layer_selector.layer_gate" in name
    }
    uground_parameter_names = {
        name
        for name in intended_trainable
        if any(
            marker in name
            for marker in [
                "mask_as_prompt",
                "grounding_mm_projector",
                "visual_model.prompt_encoder",
            ]
        )
        and name not in selector_parameter_names
    }
    base_parameters = [
        parameter for name, parameter in model.named_parameters()
        if intended_trainable[name]
        and name not in uground_parameter_names
        and name not in selector_parameter_names
    ]
    uground_parameters = [
        parameter for name, parameter in model.named_parameters()
        if intended_trainable[name] and name in uground_parameter_names
    ]
    selector_parameters = [
        parameter for name, parameter in model.named_parameters()
        if intended_trainable[name] and name in selector_parameter_names
    ]
    if not selector_parameters:
        raise RuntimeError(
            "Selector V3 found no trainable layer_selector.layer_gate parameters"
        )
    optimizer_groups = [
        {
            "params": base_parameters,
            "lr": args.lr,
            "weight_decay": 0.01,
            "group_name": "base",
        }
    ]
    if uground_parameters:
        optimizer_groups.append(
            {
                "params": uground_parameters,
                "lr": args.uground_lr,
                "weight_decay": 0.01,
                "group_name": "uground",
            }
        )
    optimizer_groups.append(
        {
            "params": selector_parameters,
            "lr": args.selector_lr,
            "weight_decay": 0.0,
            "group_name": "selector",
        }
    )
    optimizer = AdamW(
        optimizer_groups, betas=(0.9, 0.95), weight_decay=0.01
    )

    gradient_accumulation_steps = max(1, int(16 // args.batch_size))
    step_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if not step_per_epoch:
        raise ValueError("Training dataset is empty")
    total_steps = epochs * step_per_epoch
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)
    best_val_jf = float("-inf")
    best_epoch = -1
    best_checkpoint_path = os.path.join(
        args.checkpoint_root, f"{args.name}_best_val.pth"
    )

    for epoch in range(epochs):

        selector_progress = min(1.0, epoch / max(1, epochs - 1))
        selector_temperature = (
            args.selector_temperature_start
            + selector_progress
            * (args.selector_temperature_end - args.selector_temperature_start)
        )
        model.layer_selector.set_temperature(selector_temperature)

        uground_only = epoch < args.uground_warmup_epochs
        for name, parameter in model.named_parameters():
            if not intended_trainable[name]:
                parameter.requires_grad = False
            elif uground_only:
                parameter.requires_grad = (
                    name in uground_parameter_names
                    or name in selector_parameter_names
                )
            else:
                parameter.requires_grad = True
        if epoch < args.selector_warmup_epochs:
            stage = "fixed-last SAM warmup + all-layer selector supervision"
        else:
            stage = "all-layer soft-supervised selector fine-tuning"
        if uground_only:
            stage += " + UGround-only"
        print(
            f"Epoch {epoch + 1} stage: {stage}; "
            f"selector_temperature={selector_temperature:.4f}, "
            f"target_temperature={args.selector_target_temperature:.4f}"
        )
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        running_selected_layer = 0.0
        running_predicted_layer = 0.0
        running_oracle_layer = 0.0
        running_similarity_loss = 0.0
        running_selector_loss = 0.0
        running_selector_entropy = 0.0
        running_selector_max_probability = 0.0
        running_target_entropy = 0.0
        running_target_max_probability = 0.0
        running_selector_accuracy = 0.0
        running_selector_regret = 0.0
        running_oracle_loss = 0.0
        running_selector_selected_loss = 0.0
        running_valid_reference_ratio = 0.0
        selected_layer_histogram = torch.zeros(
            model.layer_selector.num_layers, dtype=torch.long
        )
        predicted_layer_histogram = torch.zeros(
            model.layer_selector.num_layers, dtype=torch.long
        )
        oracle_layer_histogram = torch.zeros(
            model.layer_selector.num_layers, dtype=torch.long
        )
        running_selector_gradient_norm = 0.0
        selector_gradient_steps = 0
        diagnostic_steps = 0

        loop = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(loop):
            input_dict = dict_to_cuda(batch)
            output_dict = model.forward(images=input_dict["images"],   #原始视频帧图，给SAM分割
                                        images_clip=input_dict["images_clip"],   #CLIP预处理后的图，给MLLM语义分割
                                        images_grounding=input_dict["images_grounding"],
                                        grounding_valid_sizes=input_dict["grounding_valid_sizes"],
                                        audio_features=input_dict["audio_feats"],  #VGGish音频特征
                                        image_features=input_dict["image_feats"],  #预先提取好的SAM图像特征
                                        input_ids=input_dict["input_ids"],  #文本token id序列，包括用户问题，模型回答模板，<image><audio>等特殊token,[SEG]这种分割标记  prepare_inputs_labels_for_multimodal根据这些token插入图像/音频token特征，组成inputs_embeds
                                        labels=input_dict["labels"],  #文本自回归损失
                                        attention_masks=input_dict["attention_masks"],  #标记哪些 token 是有效的，哪些是 padding
                                        masks_list=input_dict["masks"],   #这是真实分割掩码 GT
                                        resize_list=input_dict["resizes"],  #记录图像在送入模型前的 resize 后尺寸
                                        orgsize_list=input_dict["orgsizes"],  #原始图像尺寸,保证最终输出 mask 和原始帧对齐。
                                        conversation_list=input_dict["convs"],  #完整对话 prompt 字符串,没用
                                        refs_num=input_dict["refs_num"],  #表示一个样本里有多少个 reference expression / segmentation target，告诉模型：batch 里每个样本对应多少个 [SEG] 语义向量。
                                        fids=input_dict["fids"],  #目标对象的 id。
                                        vids=input_dict["vids"],  #视频ID
                                        contrast=args.ct_weight,   #对齐损失的权重
                                        ref_ids=input_dict["ref_ids"],   #参考文本ref的token id
                                        epoch=epoch,
                                        inference=False)

            loss = output_dict["loss"]
            if not torch.isfinite(loss):
                scalar_diagnostics = {}
                for key, value in output_dict.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        scalar_diagnostics[key] = value.detach().float().item()
                raise FloatingPointError(
                    "Non-finite training loss before backward. "
                    f"epoch={epoch} step={step} vids={input_dict['vids']} "
                    f"diagnostics={scalar_diagnostics}"
                )
            accumulation_size = min(
                gradient_accumulation_steps,
                len(train_dataloader) - (step // gradient_accumulation_steps) * gradient_accumulation_steps,
            )
            loss = loss / accumulation_size
            loss.backward()
            if epoch == 0 and step == 0:
                def gradient_norm(module):
                    gradients = [
                        parameter.grad.detach().float().norm().square()
                        for parameter in module.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    if not gradients:
                        return None
                    return torch.stack(gradients).sum().sqrt().item()

                projector_grad = gradient_norm(
                    model.get_model().grounding_mm_projector
                )
                prompt_grad = gradient_norm(
                    model.get_model().visual_model.prompt_encoder
                )
                selector_grad = gradient_norm(model.layer_selector)
                if projector_grad is None or prompt_grad is None or selector_grad is None:
                    raise RuntimeError(
                        "ALS3P is disconnected from the loss: "
                        f"grounding_projector_grad={projector_grad}, "
                        f"prompt_encoder_grad={prompt_grad}, "
                        f"selector_grad={selector_grad}"
                    )
                print(
                    "ALS3P first-step gradient norms: "
                    f"grounding_projector={projector_grad:.6g}, "
                    f"prompt_encoder={prompt_grad:.6g}, "
                    f"selector={selector_grad:.6g}"
                )
            running_loss += loss.item()
            if "selected_layer_mean" in output_dict:
                valid_references = output_dict[
                    "selector_valid_references"
                ].detach().bool()
                if bool(valid_references.any()):
                    running_selected_layer += output_dict[
                        "selected_layers"
                    ].detach()[valid_references].float().mean().item()
                    running_predicted_layer += output_dict[
                        "predicted_layers"
                    ].detach()[valid_references].float().mean().item()
                running_oracle_layer += output_dict[
                    "oracle_layer_mean"
                ].detach().float().item()
                running_similarity_loss += output_dict[
                    "similarity_loss"
                ].detach().float().item()
                running_selector_loss += output_dict[
                    "selector_loss"
                ].detach().float().item()
                running_selector_entropy += output_dict[
                    "selector_entropy"
                ].detach().float().item()
                running_selector_max_probability += output_dict[
                    "selector_max_probability"
                ].detach().float().item()
                running_target_entropy += output_dict[
                    "selector_target_entropy"
                ].detach().float().item()
                running_target_max_probability += output_dict[
                    "selector_target_max_probability"
                ].detach().float().item()
                running_selector_accuracy += output_dict[
                    "selector_top1_accuracy"
                ].detach().float().item()
                running_selector_regret += output_dict[
                    "selector_regret"
                ].detach().float().item()
                running_oracle_loss += output_dict[
                    "selector_oracle_loss"
                ].detach().float().item()
                running_selector_selected_loss += output_dict[
                    "selector_selected_loss"
                ].detach().float().item()
                running_valid_reference_ratio += output_dict[
                    "selector_valid_reference_ratio"
                ].detach().float().item()
                selected_layer_histogram += torch.bincount(
                    output_dict["selected_layers"].detach()[valid_references].cpu(),
                    minlength=model.layer_selector.num_layers,
                )
                predicted_layer_histogram += torch.bincount(
                    output_dict["predicted_layers"].detach()[valid_references].cpu(),
                    minlength=model.layer_selector.num_layers,
                )
                oracle_layer_histogram += torch.bincount(
                    output_dict["oracle_layers"].detach().cpu(),
                    minlength=model.layer_selector.num_layers,
                )
                diagnostic_steps += 1


            if (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                non_selector_parameters = [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name not in selector_parameter_names
                ]
                torch.nn.utils.clip_grad_norm_(
                    non_selector_parameters, args.max_grad_norm
                )
                selector_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    selector_parameters, args.selector_max_grad_norm
                )
                if torch.isfinite(selector_gradient_norm) and selector_gradient_norm > 0:
                    running_selector_gradient_norm += float(selector_gradient_norm.item())
                    selector_gradient_steps += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_lr = scheduler.get_last_lr()[0]
                postfix = {
                    "lr": current_lr,
                    "loss": running_loss
                    / ((step + 1) / gradient_accumulation_steps),
                }
                if diagnostic_steps:
                    postfix["layer"] = running_selected_layer / diagnostic_steps
                    postfix["map"] = running_similarity_loss / diagnostic_steps
                    postfix["selector"] = running_selector_loss / diagnostic_steps
                    postfix["regret"] = running_selector_regret / diagnostic_steps
                loop.set_postfix(**postfix)

        print(f"  Epoch {epoch + 1}, Loss:{running_loss / ((step + 1) / gradient_accumulation_steps) :.4f}, Learning Rate:{scheduler.get_last_lr()[0]:.6f}")

        # 在写入日志之前，确保目录存在
        os.makedirs(args.log_root, exist_ok=True)

        with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
            gate_log = (
                f" selected_layer {running_selected_layer / diagnostic_steps}"
                f" predicted_layer {running_predicted_layer / diagnostic_steps}"
                f" oracle_layer {running_oracle_layer / diagnostic_steps}"
                f" similarity_loss {running_similarity_loss / diagnostic_steps}"
                f" selector_loss {running_selector_loss / diagnostic_steps}"
                f" selector_entropy {running_selector_entropy / diagnostic_steps}"
                f" selector_max_probability {running_selector_max_probability / diagnostic_steps}"
                f" target_entropy {running_target_entropy / diagnostic_steps}"
                f" target_max_probability {running_target_max_probability / diagnostic_steps}"
                f" selector_top1_accuracy {running_selector_accuracy / diagnostic_steps}"
                f" selector_regret {running_selector_regret / diagnostic_steps}"
                f" oracle_loss {running_oracle_loss / diagnostic_steps}"
                f" selector_selected_loss {running_selector_selected_loss / diagnostic_steps}"
                f" valid_reference_ratio {running_valid_reference_ratio / diagnostic_steps}"
                f" selector_gradient_norm {running_selector_gradient_norm / max(1, selector_gradient_steps)}"
                f" selected_layer_histogram {selected_layer_histogram.tolist()}"
                f" predicted_layer_histogram {predicted_layer_histogram.tolist()}"
                f" oracle_layer_histogram {oracle_layer_histogram.tolist()}"
                if diagnostic_steps else ""
            )
            f.write(
                f"Epoch {epoch}: stage {stage}  running_loss "
                f"{running_loss / len(train_dataloader) * gradient_accumulation_steps}  "
                f"Learning Rate:{scheduler.get_last_lr()[0]:.6f}"
                f"{gate_log}\n"
            )

        # The loop variables keep the final training graph alive otherwise.
        del batch, input_dict, output_dict, loss
        gc.collect()
        torch.cuda.empty_cache()
        val_metrics = evaluate_segmentation(
            model, val_dataloader_refer, args, "val_refer"
        )
        if val_metrics["J&F"] > best_val_jf:
            best_val_jf = val_metrics["J&F"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            print(
                f"New best validation model at epoch {epoch}: "
                f"J&F={best_val_jf:.6f}; saved to {best_checkpoint_path}"
            )
            with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
                f.write(
                    f"best validation checkpoint: epoch {epoch}  "
                    f"J&F {best_val_jf}  path {best_checkpoint_path}\n"
                )

    last_checkpoint_path = os.path.join(
        args.checkpoint_root, f"{args.name}_last.pth"
    )
    torch.save(model.state_dict(), last_checkpoint_path)
    print(f"Last-epoch model saved to {last_checkpoint_path}")

    print(
        f"Loading best validation checkpoint from epoch {best_epoch}, "
        f"J&F={best_val_jf:.6f}: {best_checkpoint_path}"
    )
    best_state_dict = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_state_dict)
    del best_state_dict

    # Test sets are evaluated once, after model selection on validation data.
    model.eval()

    seen_metrics = evaluate_segmentation(
        model, val_dataloader_s_refer, args, "test_seen"
    )
    unseen_metrics = evaluate_segmentation(
        model, val_dataloader_u_refer, args, "test_unseen"
    )
    null_s = evaluate_null(model, val_dataloader_n_refer, args, "test_null")

    mix_j = (seen_metrics["J"] + unseen_metrics["J"]) / 2.0
    mix_f = (seen_metrics["F"] + unseen_metrics["F"]) / 2.0
    mix_jf = (mix_j + mix_f) / 2.0
    summary = (
        f"Final test results from best validation checkpoint (epoch {best_epoch}): "
        f"Seen J/F/J&F={seen_metrics['J']}/{seen_metrics['F']}/{seen_metrics['J&F']}; "
        f"Unseen J/F/J&F={unseen_metrics['J']}/{unseen_metrics['F']}/{unseen_metrics['J&F']}; "
        f"Mix J/F/J&F={mix_j}/{mix_f}/{mix_jf}; Null S={null_s}"
    )
    print(f"\n{summary}")
    with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
        f.write(summary + "\n")
