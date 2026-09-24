import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ChatUniVi.model.language_model.llama import ChatUniViLlamaModel
from ChatUniVi.model.language_model.llama_ALS3P import (
    ChatUniViALS3PLlamaForCausalLM,
)

from models.segment_anything import build_sam_vit_h
from models.ALS3P_layer_selector import (
    FullInformationLayerSelector,
)
from models.ALS3P_prompts import (
    TripleGroundingMaskPrompt,
    GaussianMapLoss,
    decode_paired_sam_prompts,
)
from collections import defaultdict



def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale: float = 1000,
        eps: float = 1e-6,
):
    """  
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """  
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



def compute_alignment_loss(q: torch.Tensor, pos_feats: list, neg_feats: list, temperature=0.07):
    """  目标一致性语义对齐损失
    q: [B, D] embedding of the output SEG token
    pos_feats: List[B][List[Tensor[D]]]   semantic embeddings of positive sets
    """
    B, D = q.shape
    device = q.device
    total_loss = 0.0
    count = 0

    for i in range(B):
        pos = pos_feats[i]
        neg = neg_feats[i]  #负样本

        if len(pos) == 0:
            continue

        # === Normalize ===
        anchor = F.normalize(q[i].unsqueeze(0), dim=1)  # [1, D]
        pos_tensors = torch.stack(pos).to(device)  # [P, D]
        pos_tensors = F.normalize(pos_tensors, dim=1)    # [P, D]

        # === Alignment ===  只在正样本集合内部做归一化的对齐损失
        sim_pos = torch.matmul(anchor, pos_tensors.T) / temperature  # [1, P]
        log_probs = F.log_softmax(sim_pos, dim=1)
        loss = -log_probs.mean()
        total_loss += loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count




class Simtoken_MetaModel:
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(Simtoken_MetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
            self.visual_model.prompt_encoder.train()
            for param in self.visual_model.prompt_encoder.parameters():
                param.requires_grad = True

        # Projection layer  MLP
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]  
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Simtoken_Model(Simtoken_MetaModel, ChatUniViLlamaModel):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(Simtoken_Model, self).__init__(config, **kwargs)

        # The spatially aligned Grounding branch adapts independently without
        # changing SimToken's pretrained clustered-vision projection.
        self.grounding_mm_projector = copy.deepcopy(self.mm_projector)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

    def initialize_grounding_projector_from_mm(self):
        self.grounding_mm_projector.load_state_dict(self.mm_projector.state_dict())


class SemanticMemoryBank:
    def __init__(self, max_per_object=5):
        self.bank = defaultdict(lambda: defaultdict(list))  # bank[vid][fid] = [feat1, feat2, ...]
        self.max_per_object = max_per_object

    def add(self, vid: str, fid: int, feat: torch.Tensor):
        feat = feat.detach().cpu()
        self.bank[vid][fid].append(feat)
        if len(self.bank[vid][fid]) > self.max_per_object:
            self.bank[vid][fid] = self.bank[vid][fid][-self.max_per_object:]  # 保留最新的 K 个

    def add_batch(self, vids: list, fids: list, feats: torch.Tensor):
        for vid, fid, feat in zip(vids, fids, feats):
            self.add(vid, int(fid), feat)

    def get_positive_features(self, vids: list, fids: list):
        results = []
        for vid, fid in zip(vids, fids):
            pos = self.bank[vid][int(fid)].copy()  # List[Tensor]
            results.append(pos)
        return results

    def get_negative_features_same_vid(self, vids: list, fids: list):
        results = []
        for vid, fid in zip(vids, fids):
            neg = []
            for other_fid, feats in self.bank[vid].items():
                if other_fid != int(fid):
                    neg.extend(feats)
            results.append(neg)
        return results


class ALS3PForCausalLM(ChatUniViALS3PLlamaForCausalLM):
    def __init__(
            self,
            config,
            **kwargs,
    ):

        layer_strategy = kwargs.pop("uground_layer_strategy", "selector")
        candidate_layers = kwargs.pop("uground_candidate_layers", None)
        eval_layer_strategy = kwargs.pop("uground_eval_layer_strategy", "argmax")
        if layer_strategy != "selector" or eval_layer_strategy != "argmax":
            raise ValueError("ALS3P requires selector/argmax strategies")
        selector_temperature = kwargs.pop("uground_selector_temperature", 1.0)
        selector_target_temperature = kwargs.pop(
            "uground_selector_target_temperature", 1.0
        )
        selector_standardize_targets = kwargs.pop(
            "uground_selector_standardize_targets", False
        )
        self.selector_warmup_epochs = int(
            kwargs.pop("uground_selector_warmup_epochs", 0)
        )
        self.map_loss_weight = float(kwargs.pop("uground_map_loss_weight", 0.0))
        self.selector_loss_weight = float(
            kwargs.pop("uground_selector_loss_weight", 0.1)
        )
        map_bce_weight = kwargs.pop("uground_map_bce_weight", 1.0)
        map_dice_weight = kwargs.pop("uground_map_dice_weight", 1.0)
        gaussian_kernel = kwargs.pop("uground_gaussian_kernel", 7)
        gaussian_sigma = kwargs.pop("uground_gaussian_sigma", 2.0)
        initial_temperature = kwargs.pop("uground_temperature", 1.0)
        grounding_grid_size = int(kwargs.pop("uground_grounding_grid_size", 10))
        maximum_sequence_length = int(
            kwargs.pop("uground_max_sequence_length", 2048)
        )
        if grounding_grid_size <= 0:
            raise ValueError("uground_grounding_grid_size must be positive")
        if maximum_sequence_length <= 0:
            raise ValueError("uground_max_sequence_length must be positive")
        config.uground_grounding_grid_size = grounding_grid_size
        config.uground_max_sequence_length = maximum_sequence_length
        self.return_similarity_maps = bool(
            kwargs.pop("uground_return_similarity_maps", False)
        )

        if not hasattr(config, "train_mask_decoder"):
            #
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)

            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            # 从 kwargs 字典中取出 weight 的值，。如果 kwargs 里没有 eight，则返回 None
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            self.ce_loss_weight = getattr(config, "ce_loss_weight", 1.0)
            self.dice_loss_weight = getattr(config, "dice_loss_weight", 0.5)
            self.bce_loss_weight = getattr(config, "bce_loss_weight", 2.0)

        self.seg_token_idx = kwargs.pop("seg_token_idx")


        super().__init__(config)

        self.model = Simtoken_Model(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.audio_feature_layer = nn.Linear(in_features=128, out_features=4096)

        num_layers = int(config.num_hidden_layers) + 1
        self.layer_selector = FullInformationLayerSelector(
            num_layers=num_layers,
            hidden_dim=config.hidden_size,
            candidate_layers=candidate_layers,
            temperature=selector_temperature,
            target_temperature=selector_target_temperature,
            standardize_targets=selector_standardize_targets,
        )
        self.mask_as_prompt = TripleGroundingMaskPrompt(
            hidden_dim=config.hidden_size,
            grid_size=grounding_grid_size,
            initial_temperature=initial_temperature
        )
        self.similarity_map_loss = GaussianMapLoss(
            kernel_size=gaussian_kernel,
            sigma=gaussian_sigma,
            bce_weight=map_bce_weight,
            dice_weight=map_dice_weight,
        )
        self.layer_selector.reset_parameters()
        self.mask_as_prompt.reset_parameters(initial_temperature)

        config.uground_layer_strategy = layer_strategy
        config.uground_eval_layer_strategy = eval_layer_strategy
        config.uground_map_loss_weight = self.map_loss_weight
        config.uground_selector_loss_weight = self.selector_loss_weight
        config.uground_selector_warmup_epochs = self.selector_warmup_epochs
        config.uground_grounding_grid_size = grounding_grid_size
        config.uground_max_sequence_length = maximum_sequence_length

        self.memory = SemanticMemoryBank()

        self.compress = kwargs.pop("compress", True)

        self.start = kwargs.pop("start")




    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.model.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
            self,
            images: torch.FloatTensor,  #原始视频帧图，给SAM分割【后续并没有使用】
            images_clip: torch.FloatTensor,  # 经过CLIP预处理的图像，传给prepare_inputs_labels_for_multimodal()
            images_grounding: torch.FloatTensor,
            grounding_valid_sizes: List[List[tuple]],
            audio_features: torch.FloatTensor,  #预提取好的VGGish音频特征
            image_features: torch.FloatTensor,   #预提取好的SAM特征
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            masks_list: List[torch.FloatTensor],
            resize_list: List[tuple],
            orgsize_list: List[tuple],
            conversation_list: List[str],
            # num_frame_list: List[int], # 固定为10
            # num_conv_list: List[int],  # 固定为1
            ref_ids: List[torch.LongTensor],
            refs_num: List[int],
            vids,
            fids,
            epoch: int =0,
            inference: bool = False,
            num_frames: int = 10,
            contrast: float = 0.0,

            **kwargs,
    ):
        batch_size = len(images)
        image_embeddings = torch.cat(image_features, dim=0)
        # image_embeddings = self.get_visual_embs(torch.cat(images, dim=0)) # [BT, 256, 64, 64]

        # audio_embeddings = self.audio_feature_layer(torch.stack(audio_features, dim=0))  # [B, 10, 4096]
        audio_stack = torch.stack(audio_features, dim=0).to(self.audio_feature_layer.weight.dtype) #my
        audio_embeddings = self.audio_feature_layer(audio_stack)

        alignment_aux = None
        alignment_hook = getattr(
            self, "_compute_alignment_filtered_features", None
        )
        if alignment_hook is not None:
            _, alignment_aux = alignment_hook(
                images_clip=images_clip,
                audio_features=audio_stack,
                ref_ids=ref_ids,
                masks_list=masks_list,
                num_frames=num_frames,
                epoch=epoch,
                inference=inference,
            )

        # audio_embeddings = torch.cat(audio_features, dim=0) # [B*10, 128]
        # audio_embeddings = audio_features  # [B, 10, 128]

        target_frame = 1

        (
            input_ids,
            attention_masks,
            past_key_values,
            inputs_embeds,
            labels,
            _,
            _,
            grounding_ranges,
            grounding_grid_validity,
        ) = super().prepare_inputs_labels_for_multimodal(
            input_ids, attention_masks, past_key_values=None, labels=labels,
            images=images_clip, audio_features=audio_embeddings,
            grounding_images=images_grounding,
            grounding_valid_sizes=grounding_valid_sizes,
            target_frame=target_frame, ref_ids=ref_ids,
        )
        if grounding_ranges is None:
            raise RuntimeError("Aligned multimodal preparation returned no grounding ranges")
        if grounding_grid_validity is None:
            raise RuntimeError("Masked multimodal preparation returned no validity grid")


        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states

        seg_token_mask = output.labels[..., 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1), device=output.labels.device).bool(), ],
            dim=1, )  # [batch_size, seq_len]

        selector_warmup = (
            self.training
            and not inference
            and epoch < self.selector_warmup_epochs
        )
        selection = self.layer_selector(
            output_hidden_states,
            seg_token_mask,
            force_last=selector_warmup,
        )
        seg_embeddings = self.model.text_hidden_fcs[0](selection.hidden)

        expected_seg_count = sum(int(count) for count in refs_num)
        if seg_embeddings.shape[0] != expected_seg_count:
            raise ValueError(
                "The generated [SEG] count does not match refs_num: "
                f"{seg_embeddings.shape[0]} vs {expected_seg_count}"
            )

        # print("seg_embeddings in this batch:", seg_embeddings.shape)
        # print("vids:", vids)
        # print("fids:", fids)
        fis_flat = [fid[0] for fid in fids]
        # print("fids:", fis_flat )
        if not inference:
            # Keep the contrastive memory in the stable final-layer SAM prompt
            # space; selector-selected layers are used only by the grounding path.
            alignment_embeddings = self.model.text_hidden_fcs[0](
                output_hidden_states[-1][seg_token_mask]
            )

            pos_feats = self.memory.get_positive_features(vids, fis_flat )  # 正样本
            neg_feats = self.memory.get_negative_features_same_vid(vids, fis_flat )  # 负样本

            for i in range(len(neg_feats)):
                for j in range(len(alignment_embeddings)):
                    if j != i:
                        neg_feats[i].append(alignment_embeddings[j].detach().cpu())
            #语义对齐损失
            ct_loss = compute_alignment_loss(alignment_embeddings, pos_feats, neg_feats)

            # print("ct loss:", ct_loss)
            self.memory.add_batch(vids, fis_flat, alignment_embeddings)


        pred_embeddings = []
        selected_seg_hidden = []
        #--------------------------------------------------------------------------------------------
        pred_idx = 0
        for ref_num in refs_num:
            pred_embeddings.append(seg_embeddings[pred_idx:pred_idx + ref_num])
            selected_seg_hidden.append(
                selection.hidden[pred_idx:pred_idx + ref_num]
            )
            pred_idx += ref_num
        # list[B]:[num_seg, 256]

        grounding_hidden = self.mask_as_prompt.gather_same_layer_tokens(
            hidden_states=output_hidden_states,
            grounding_ranges=grounding_ranges,
            selected_layers=selection.indices,
            layer_probabilities=selection.probabilities,
            refs_num=refs_num,
            num_frames=num_frames,
            soft_selection=False,
        )

        candidate_seg_hidden = []
        candidate_grounding_hidden = None
        if not inference:
            candidate_grounding_hidden = (
                self.mask_as_prompt.gather_candidate_layer_tokens(
                    hidden_states=output_hidden_states,
                    grounding_ranges=grounding_ranges,
                    candidate_layers=self.layer_selector.candidate_layers,
                    refs_num=refs_num,
                    num_frames=num_frames,
                )
            )
            candidate_offset = 0
            for ref_num in refs_num:
                candidate_seg_hidden.append(
                    selection.candidate_hidden[
                        candidate_offset:candidate_offset + ref_num
                    ]
                )
                candidate_offset += ref_num

        pred_masks = []
        similarity_logits = []
        candidate_similarity_logits = []
        for i in range(batch_size):
            frame_embeddings = image_embeddings[
                i * num_frames: (i + 1) * num_frames
            ]
            sample_similarity = self.mask_as_prompt.similarity_logits(
                selected_seg_hidden[i],
                grounding_hidden[i],
                valid_grid=grounding_grid_validity[i],
            )
            similarity_logits.append(sample_similarity)
            if not inference:
                with torch.no_grad():
                    candidate_similarity_logits.append(
                        self.mask_as_prompt.candidate_similarity_logits(
                            candidate_seg_hidden[i].detach(),
                            candidate_grounding_hidden[i].detach(),
                            valid_grid=grounding_grid_validity[i],
                        )
                    )
            dense_masks = self.mask_as_prompt.dense_prompt_masks(
                sample_similarity,
                valid_sizes=grounding_valid_sizes[i],
            )
            pred_masks_sample = []
            for prompt_idx in range(len(pred_embeddings[i])):
                repeated_text = pred_embeddings[i][prompt_idx].reshape(1, 1, -1)
                repeated_text = repeated_text.expand(num_frames, -1, -1)
                prompt_dtype = next(
                    self.model.visual_model.prompt_encoder.parameters()
                ).dtype
                sparse_embeddings, dense_embeddings = (
                    self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=dense_masks[prompt_idx].to(prompt_dtype),
                        text_embeds=repeated_text.to(prompt_dtype),
                    )
                )
                decoder_dtype = next(
                    self.model.visual_model.mask_decoder.parameters()
                ).dtype
                low_res_masks, iou_predictions = decode_paired_sam_prompts(
                    decoder=self.model.visual_model.mask_decoder,
                    image_embeddings=frame_embeddings.to(decoder_dtype),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings.to(decoder_dtype),
                    dense_prompt_embeddings=dense_embeddings.to(decoder_dtype),
                    multimask_output=False,
                )

                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=orgsize_list[i]
                )
                pred_masks_sample.append(pred_mask.squeeze(1))

            pred_masks.append(torch.stack(pred_masks_sample, dim=0))
        gt_masks = masks_list # list[B]:[num_seg, T, H, W]

        if inference:
            inference_output = {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "selected_layers": selection.indices,
                "predicted_layers": self.layer_selector.last_predicted_indices,
            }
            inference_output.update(self.layer_selector.diagnostics(seg_embeddings.device))
            if self.return_similarity_maps:
                inference_output["similarity_maps"] = [
                    similarity_map for similarity_map in similarity_logits
                ]
            return inference_output

        model_output = output
        output = model_output.logits


        ce_loss = model_output.loss  #文本自回归损失  
        ce_loss = ce_loss * self.ce_loss_weight

        mask_bce_loss = ce_loss.new_zeros(())
        mask_dice_loss = ce_loss.new_zeros(())
        per_reference_map_losses = []
        per_reference_map_bce = []
        per_reference_map_dice = []
        per_reference_validity = []
        all_candidate_map_losses = []
        num_masks = 0

        # 计算预测掩码和gt之间的loss
        for batch_idx in range(batch_size):


            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            a, b, c, d = gt_mask.shape
            gt_mask = gt_mask.view(a*b, c, d)  # [num_ref*T, H, W]
            pred_mask = pred_mask.view(a*b, c, d)  # [num_ref*T, H, W]

            # print("gt_mask:", gt_mask.shape)


            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )  # 逐像素二分类交叉熵  
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )  # 关注的是预测区域和真实区域的重叠程度
            num_masks += gt_mask.shape[0]

            (
                sample_map_loss,
                sample_map_bce,
                sample_map_dice,
                _,
                _,
                sample_validity,
            ) = self.similarity_map_loss(
                    similarity_logits[batch_idx].detach(),
                    gt_masks[batch_idx],
                    valid_sizes=grounding_valid_sizes[batch_idx],
                )
            per_reference_map_losses.append(sample_map_loss)
            per_reference_map_bce.append(sample_map_bce)
            per_reference_map_dice.append(sample_map_dice)
            per_reference_validity.append(sample_validity)

            references, candidates, frames = candidate_similarity_logits[
                batch_idx
            ].shape[:3]
            candidate_masks = gt_masks[batch_idx].unsqueeze(1).expand(
                -1, candidates, -1, -1, -1
            ).reshape(
                references * candidates,
                frames,
                *gt_masks[batch_idx].shape[-2:],
            )
            (
                candidate_map_loss,
                _,
                _,
                _,
                _,
                candidate_validity,
            ) = self.similarity_map_loss(
                candidate_similarity_logits[batch_idx].reshape(
                    references * candidates,
                    frames,
                    *candidate_similarity_logits[batch_idx].shape[-2:],
                ),
                candidate_masks,
                valid_sizes=grounding_valid_sizes[batch_idx],
            )
            candidate_map_loss = candidate_map_loss.reshape(references, candidates)
            candidate_validity = candidate_validity.reshape(references, candidates)
            if not torch.equal(
                candidate_validity,
                candidate_validity[:, :1].expand_as(candidate_validity),
            ):
                raise RuntimeError("candidate layers disagree on GT reference validity")
            all_candidate_map_losses.append(candidate_map_loss)

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        per_reference_map_loss = torch.cat(per_reference_map_losses, dim=0)
        per_reference_bce = torch.cat(per_reference_map_bce, dim=0)
        per_reference_dice = torch.cat(per_reference_map_dice, dim=0)
        valid_references = torch.cat(per_reference_validity, dim=0)
        valid_weight = valid_references.to(per_reference_map_loss.dtype)
        valid_denominator = valid_weight.sum().clamp_min(1.0)
        similarity_loss = (per_reference_map_loss * valid_weight).sum() / valid_denominator
        similarity_bce_loss = (per_reference_bce * valid_weight).sum() / valid_denominator
        similarity_dice_loss = (per_reference_dice * valid_weight).sum() / valid_denominator
        candidate_map_loss = torch.cat(all_candidate_map_losses, dim=0)
        selector_loss = self.layer_selector.selector_loss(
            candidate_map_loss,
            valid_references=valid_references,
        )



        ct_weight = contrast

        alignment_aux_loss = mask_loss * 0.0
        if alignment_aux is not None:
            alignment_aux_loss = alignment_aux["loss"]


        if epoch >= self.start:
            loss = (
                ce_loss + mask_loss + ct_weight * ct_loss
                + alignment_aux_loss
                + self.map_loss_weight * similarity_loss
                + self.selector_loss_weight * selector_loss
            )
        else:
            loss = (
                ce_loss + mask_loss + alignment_aux_loss
                + self.map_loss_weight * similarity_loss
                + self.selector_loss_weight * selector_loss
            )

        result = {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "ct_loss": ct_loss,
            "similarity_loss": similarity_loss,
            "similarity_bce_loss": similarity_bce_loss,
            "similarity_dice_loss": similarity_dice_loss,
            "selector_loss": selector_loss,
            "selector_warmup_active": ce_loss.new_tensor(float(selector_warmup)),
            "selector_valid_reference_ratio": valid_weight.mean(),
            "selector_valid_references": valid_references.detach(),
            "selected_layers": selection.indices.detach(),
            "predicted_layers": self.layer_selector.last_predicted_indices,
            "oracle_layers": self.layer_selector.last_oracle_indices,
            "masp_temperature": self.mask_as_prompt.temperature.detach(),
            "trimalign_loss": alignment_aux_loss,
            "trimalign_alignment_loss": (
                alignment_aux["alignment_loss"]
                if alignment_aux is not None else alignment_aux_loss.detach()
            ),
            "trimalign_av_loss": (
                alignment_aux["av_loss"]
                if alignment_aux is not None else alignment_aux_loss.detach()
            ),
            "trimalign_audio_gate_mean": (
                alignment_aux["audio_gate_mean"]
                if alignment_aux is not None else alignment_aux_loss.detach()
            ),
            "trimalign_filtered_patch_ratio": (
                alignment_aux["filtered_patch_ratio"]
                if alignment_aux is not None else alignment_aux_loss.detach()
            ),
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
        }
        result.update(self.layer_selector.diagnostics(seg_embeddings.device))
        return result


    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")
