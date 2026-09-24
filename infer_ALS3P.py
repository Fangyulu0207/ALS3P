import warnings

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

import transformers
import importlib.util
from pathlib import Path


from datasets.dataset_refavs_ALS3P import REFAVSALS3P

_config_path = (
    Path(__file__).resolve().parent
    / "configs"
    / "config_infer_ALS3P.py"
)
_config_spec = importlib.util.spec_from_file_location(
    "config_infer_ALS3P", _config_path
)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Unable to load ALS3P config: {_config_path}")
_config_module = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config_module)
args = _config_module.args
from torch.utils.data import DataLoader
from functools import partial
# from  models.avs_model import VISAForCausalLM
from models.ALS3P import ALS3PForCausalLM
import torch
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from utils import utility
import random
import numpy as np
import re
import os
import csv
from PIL import Image



from transformers import logging
logging.set_verbosity_error()


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

AUDIO_TOKEN_INDEX = -300
GROUNDING_TOKEN_INDEX = -400

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dict_to_cuda(input_dict,dtype=torch.bfloat16):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True).to(dtype) if v.is_floating_point() else v.cuda(non_blocking=True)
        elif (
                isinstance(input_dict[k], list)
                and len(input_dict[k]) > 0
                and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [e.cuda(non_blocking=True).to(dtype) if e.is_floating_point() else e.cuda(non_blocking=True) for e in v]
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

    # divide prompt into two set
    text_chunks = []  # text
    token_types = []  # <image>/<audio>/<video>
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

    # Tokenize the text
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
    first_refs = []

    refs = []
    first_refs = []
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

        refs.append(data['ref'])
        first_refs.append(data['ref'][0])

    input_ids = [tokenizer_image_audio_token(conv, tokenizer, return_tensors="pt") for conv in conversations]  # list
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    ref_ids = [tokenizer_image_audio_token(ref, tokenizer, return_tensors="pt") for ref in first_refs]

    labels = input_ids.clone()

    sep = 'Sure, It is [SEG]'

    for conversation, target in zip(conversations, labels):
        parts = conversation.split(sep)
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
            "fids": fids,
            "refs": refs,
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
        local_files_only=True  # 强制使用本地文件
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]  # 32000
    print("seg_token_idx: ", seg_token_idx)


    val_dataset_s = REFAVSALS3P('test_s', args, tokenizer, input_type='refer')
    val_dataset_u = REFAVSALS3P('test_u', args, tokenizer, input_type='refer')
    val_dataset_n = REFAVSALS3P('test_n', args, tokenizer, input_type='refer')


    val_dataloader_s = DataLoader(val_dataset_s, batch_size=1, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_u = DataLoader(val_dataset_u, batch_size=1, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_n = DataLoader(val_dataset_n, batch_size=2, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))



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
        "uground_selector_loss_weight": args.selector_loss_weight,
        "uground_selector_lr": args.selector_lr,
        "uground_selector_warmup_epochs": args.selector_warmup_epochs,
        "uground_selector_temperature": args.selector_temperature_start,
        "uground_selector_target_temperature": args.selector_target_temperature,
        "uground_selector_standardize_targets": False,
        "uground_map_loss_weight": args.map_loss_weight,
        "uground_map_bce_weight": args.map_bce_weight,
        "uground_map_dice_weight": args.map_dice_weight,
        "uground_gaussian_kernel": args.gaussian_kernel,
        "uground_gaussian_sigma": args.gaussian_sigma,
        "uground_temperature": args.masp_temperature,
        "uground_grounding_grid_size": args.grounding_grid_size,
        "uground_max_sequence_length": args.max_sequence_length,
        # This entry point always needs the maps, regardless of CLI defaults.
        "uground_return_similarity_maps": True,
    }


    # model = ALS3PForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.float32, low_cpu_mem_usage=True, **model_args)
    model = ALS3PForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,local_files_only=True,  # 🔥 强制使用本地文件
                                                 **model_args)
    # model = ALS3PForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto", **model_args)


    print("\nmodel loaded")
    model.layer_selector.reset_parameters()
    model.mask_as_prompt.reset_parameters(args.masp_temperature)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # model.enable_input_require_grads()
    # model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.float32)

    model_args_from_pt = AutoConfig.from_pretrained(args.mllm,local_files_only=True)
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

        

    model = model.to("cuda")

    model = model.to(torch.bfloat16)

    model.resize_token_embeddings(len(tokenizer))

    checkpoint = torch.load(args.saved_model, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "ALS3P checkpoint is incompatible. "
            f"missing={list(incompatible.missing_keys)}; "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )

    
    # SAM 和 vision_tower 保持 float32
    for name, module in model.named_modules():
        if "visual_model" in name or "vision_tower" in name:
            module.to(torch.float32)
    model.layer_selector.to(device="cuda", dtype=torch.float32)
    model.mask_as_prompt.to(device="cuda", dtype=torch.float32)
    model.similarity_map_loss.to(device="cuda", dtype=torch.float32)
    print("saved model loaded")



    save_root = args.visualization_root

    def _safe_name(value):
        value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("._")
        return value[:120] or "unnamed"

    def _turbo_colormap(values):
        """Small dependency-free approximation of the Turbo color map."""
        x = np.clip(values.astype(np.float32), 0.0, 1.0)
        coefficients = np.asarray(
            [
                [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
                [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
            ],
            dtype=np.float32,
        )
        powers = np.stack([np.ones_like(x), x, x**2, x**3, x**4, x**5], axis=-1)
        rgb = np.einsum("...k,ck->...c", powers, coefficients)
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _resize_scalar(values, size, resample=Image.Resampling.BILINEAR):
        image = Image.fromarray(values.astype(np.float32), mode="F")
        return np.asarray(image.resize(size, resample=resample), dtype=np.float32)

    def _mask_overlay(original, mask, color, alpha=0.45):
        output = original.astype(np.float32).copy()
        active = mask.astype(bool)
        output[active] = (
            (1.0 - alpha) * output[active]
            + alpha * np.asarray(color, dtype=np.float32)
        )
        return np.clip(output, 0, 255).astype(np.uint8)

    def _save_selected_layer_similarity(
        ref_root,
        frame_index,
        original,
        similarity,
        valid_size,
    ):
        """Save the selected-layer map in the original frame coordinates."""
        height, width = original.shape[:2]
        prompt_size = 256
        valid_height, valid_width = (int(valid_size[0]), int(valid_size[1]))
        source_size = max(valid_height, valid_width)
        prompt_height = max(
            1, min(prompt_size, round(valid_height * prompt_size / source_size))
        )
        prompt_width = max(
            1, min(prompt_size, round(valid_width * prompt_size / source_size))
        )
        dense_square = _resize_scalar(
            similarity, (prompt_size, prompt_size), Image.Resampling.BILINEAR
        )
        aligned = _resize_scalar(
            dense_square[:prompt_height, :prompt_width], (width, height)
        )
        heatmap = _turbo_colormap(aligned)
        overlay = np.clip(
            (1.0 - args.heatmap_alpha) * original.astype(np.float32)
            + args.heatmap_alpha * heatmap.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        # Keep only the original-frame overlay for compact ablation outputs.
        Image.fromarray(overlay).save(
            os.path.join(ref_root, f"frame{frame_index}_overlay.jpg"), quality=95
        )

    def visualization(model, dataloader, save_root, name):
        save_root = os.path.join(save_root, name)
        os.makedirs(save_root, exist_ok=True)
        print(f"save_root: {save_root}")
        model.eval()
        exported = 0
        selected_layer_rows = []
        total_j = 0.0
        total_f = 0.0
        total_s = 0.0
        metric_count = 0
        for batch in tqdm(dataloader, desc=f"Visualization on {name} "):
            if args.max_visualizations >= 0 and exported >= args.max_visualizations:
                break
            input_dict = dict_to_cuda(batch)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                with torch.no_grad():
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
            if "similarity_maps" not in output_dict:
                raise RuntimeError("Model did not return similarity_maps")
            pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
            gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
            for metric_index in range(len(pred_masks)):
                num_seg = pred_masks[metric_index].shape[0]
                frame_count = pred_masks[metric_index].shape[1]
                weight = num_seg * frame_count
                if name == "test_null":
                    total_s += float(
                        utility.metric_s_for_null(pred_masks[metric_index])
                    ) * weight
                else:
                    total_j += float(
                        utility.mask_iou(
                            pred_masks[metric_index], gt_masks[metric_index]
                        )
                    ) * weight
                    total_f += float(
                        utility.Eval_Fmeasure(
                            pred_masks[metric_index], gt_masks[metric_index], None
                        )
                    ) * weight
                metric_count += weight
            similarity_maps = output_dict["similarity_maps"]
            if "selected_layers" not in output_dict:
                raise RuntimeError("Policy model did not return selected_layers")
            selected_layers = (
                output_dict["selected_layers"].detach().long().cpu().reshape(-1)
            )
            expected_layers = sum(int(count) for count in input_dict["refs_num"])
            if selected_layers.numel() != expected_layers:
                raise RuntimeError(
                    "selected_layers does not match refs_num: "
                    f"{selected_layers.numel()} vs {expected_layers}"
                )

            selected_offset = 0
            for b in range(len(pred_masks)):
                if args.max_visualizations >= 0 and exported >= args.max_visualizations:
                    break
                sample_similarity = similarity_maps[b].float().cpu()
                vid = input_dict["vids"][b]
                vid_root = os.path.join(save_root, _safe_name(vid))
                os.makedirs(vid_root, exist_ok=True)

                num_seg, T = sample_similarity.shape[:2]
                sample_selected_layers = selected_layers[
                    selected_offset:selected_offset + num_seg
                ]
                selected_offset += num_seg

                for seg_idx in range(num_seg):
                    ref = input_dict["refs"][b][seg_idx]
                    selected_layer = int(sample_selected_layers[seg_idx].item())
                    ref_root = os.path.join(vid_root, str(ref))
                    os.makedirs(ref_root, exist_ok=True)
                    selected_layer_rows.append(
                        {
                            "split": name,
                            "video": str(vid),
                            "reference_index": seg_idx,
                            "reference": str(ref),
                            "selected_layer": selected_layer,
                        }
                    )
                    print(
                        f"[{name}] video={vid} reference_index={seg_idx} "
                        f"reference={ref} selected_layer={selected_layer}",
                        flush=True,
                    )

                    for t in range(T):
                        frame_path = os.path.join(
                            args.data_dir, "media", vid, "frames", f"{t}.jpg"
                        )
                        original = np.asarray(Image.open(frame_path).convert("RGB"))
                        _save_selected_layer_similarity(
                            ref_root,
                            t,
                            original,
                            sample_similarity[seg_idx, t].numpy(),
                            input_dict["grounding_valid_sizes"][b][t],
                        )
                exported += 1
        csv_path = os.path.join(save_root, "selected_layers.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "split",
                    "video",
                    "reference_index",
                    "reference",
                    "selected_layer",
                ],
            )
            writer.writeheader()
            writer.writerows(selected_layer_rows)
        layer_counts = {}
        for row in selected_layer_rows:
            layer = int(row["selected_layer"])
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        print(f"\n[{name}] final selected-layer results")
        print(f"[{name}] references={len(selected_layer_rows)}")
        print(
            f"[{name}] layer_histogram="
            + str(dict(sorted(layer_counts.items())))
        )
        print(f"visualization finished: exported {exported} samples to {save_root}")
        print(f"selected-layer metadata saved to {csv_path}")
        if metric_count > 0:
            if name == "test_null":
                print(f"Final {name}: S={total_s / metric_count:.6f}")
            else:
                final_j = total_j / metric_count
                final_f = total_f / metric_count
                print(
                    f"Final {name}: J={final_j:.6f}, F={final_f:.6f}, "
                    f"J&F={(final_j + final_f) / 2.0:.6f}"
                )








    split_loaders = {
        "test_seen": val_dataloader_s,
        "test_unseen": val_dataloader_u,
        "test_null": val_dataloader_n,
    }
    requested_splits = [item.strip() for item in args.visualize_splits.split(",") if item.strip()]
    unknown_splits = sorted(set(requested_splits) - set(split_loaders))
    if unknown_splits:
        raise ValueError(f"Unknown visualization splits: {unknown_splits}")
    for split in requested_splits:
        visualization(model, split_loaders[split], save_root, split)
