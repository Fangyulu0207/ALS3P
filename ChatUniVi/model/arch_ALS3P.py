"""ALS3P masked-grounding token extension for Chat-UniVi.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ChatUniVi.model.arch import ChatUniViMetaForCausalLM
from ChatUniVi.constants import (
    AUDIO_TOKEN_INDEX,
    GROUNDING_TOKEN_INDEX,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)


class ChatUniViALS3PMetaForCausalLM(ChatUniViMetaForCausalLM):
    @staticmethod
    def _build_patch_validity(
        valid_sizes: Sequence[Sequence[tuple[int, int]]],
        batch_size: int,
        num_frames: int,
        patch_side: int,
        image_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if len(valid_sizes) != batch_size:
            raise ValueError("one grounding-valid-size list is required per sample")
        rows = torch.arange(patch_side, device=device, dtype=torch.float32)
        columns = torch.arange(patch_side, device=device, dtype=torch.float32)
        cell_size = float(image_size) / patch_side
        row_start = rows * cell_size
        row_end = (rows + 1) * cell_size
        column_start = columns * cell_size
        column_end = (columns + 1) * cell_size

        masks = []
        for sample_sizes in valid_sizes:
            if len(sample_sizes) != num_frames:
                raise ValueError("one grounding valid size is required per frame")
            for valid_height, valid_width in sample_sizes:
                valid_height = float(valid_height)
                valid_width = float(valid_width)
                if not 0 < valid_height <= image_size or not 0 < valid_width <= image_size:
                    raise ValueError(
                        f"invalid grounding size {(valid_height, valid_width)} "
                        f"for square input {image_size}"
                    )
                row_coverage = (
                    torch.minimum(row_end, row_end.new_tensor(valid_height))
                    - row_start
                ).clamp(min=0.0, max=cell_size) / cell_size
                column_coverage = (
                    torch.minimum(column_end, column_end.new_tensor(valid_width))
                    - column_start
                ).clamp(min=0.0, max=cell_size) / cell_size
                masks.append(row_coverage[:, None] * column_coverage[None, :])
        return torch.stack(masks, dim=0).unsqueeze(1)

    def _build_grounding_tokens(
        self,
        raw_features: torch.Tensor,
        batch_size: int,
        num_frames: int,
        grid_size: int,
        valid_sizes: Sequence[Sequence[tuple[int, int]]],
        image_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(raw_features, torch.Tensor) or raw_features.ndim != 3:
            raise RuntimeError(
                "The vision hook must capture [B*T, patches, channels] features"
            )
        if raw_features.shape[0] != batch_size * num_frames:
            raise RuntimeError(
                "Captured CLIP batch does not match video batch: "
                f"{raw_features.shape[0]} vs {batch_size * num_frames}"
            )
        patch_side = math.isqrt(raw_features.shape[1])
        if patch_side * patch_side != raw_features.shape[1]:
            raise RuntimeError(
                f"CLIP patch count {raw_features.shape[1]} is not a square grid"
            )
        if not 1 <= grid_size <= patch_side:
            raise ValueError(
                f"grounding grid must be in [1, {patch_side}], got {grid_size}"
            )

        channels = raw_features.shape[-1]
        spatial = raw_features.reshape(
            batch_size, num_frames, patch_side, patch_side, channels
        )
        spatial = spatial.permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_frames, channels, patch_side, patch_side
        )
        patch_validity = ChatUniViALS3PMetaForCausalLM._build_patch_validity(
            valid_sizes=valid_sizes,
            batch_size=batch_size,
            num_frames=num_frames,
            patch_side=patch_side,
            image_size=image_size,
            device=spatial.device,
        )
        grid_validity = F.adaptive_avg_pool2d(
            patch_validity, (grid_size, grid_size)
        )
        pooled = F.adaptive_avg_pool2d(
            spatial.float() * patch_validity, (grid_size, grid_size)
        ) / grid_validity.clamp_min(1e-6)
        pooled = pooled * (grid_validity > 0).to(pooled.dtype)
        pooled = pooled.to(raw_features.dtype).flatten(2).transpose(1, 2)
        pooled = pooled.reshape(
            batch_size, num_frames * grid_size * grid_size, channels
        )
        projector = self.get_model().grounding_mm_projector
        projector_parameter = next(projector.parameters(), None)
        if projector_parameter is not None:
            pooled = pooled.to(
                device=projector_parameter.device, dtype=projector_parameter.dtype
            )
        projected = projector(pooled)
        grid_validity = grid_validity.reshape(
            batch_size, num_frames, grid_size, grid_size
        )
        return projected, grid_validity

    @staticmethod
    def _shift_ranges_after_grounding(
        ranges: Optional[Sequence[tuple[int, int]]],
        placeholder_positions: Sequence[int],
        inserted_delta: int,
    ) -> Optional[list[tuple[int, int]]]:
        if ranges is None:
            return None
        if len(ranges) != len(placeholder_positions):
            raise ValueError("range count does not match grounding placeholders")
        return [
            (
                start + inserted_delta if start > position else start,
                end + inserted_delta if end > position else end,
            )
            for (start, end), position in zip(ranges, placeholder_positions)
        ]

    @staticmethod
    def _grounding_placeholder_positions(input_ids: torch.Tensor) -> list[int]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        positions = []
        for sample_ids in input_ids:
            matches = torch.where(sample_ids == GROUNDING_TOKEN_INDEX)[0]
            if matches.numel() != 1:
                raise ValueError(
                    "each ALS3P prompt must contain exactly one <grounding> placeholder"
                )
            position = int(matches.item())
            multimodal = torch.where(
                (sample_ids == IMAGE_TOKEN_INDEX)
                | (sample_ids == AUDIO_TOKEN_INDEX)
            )[0]
            if multimodal.numel() and int(multimodal.min().item()) < position:
                raise ValueError(
                    "<grounding> must appear before <video>, <image>, and <audio>"
                )
            positions.append(position)
        return positions

    @staticmethod
    def _replace_grounding_placeholders(
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        grounding_tokens: torch.Tensor,
        grounding_attention_mask: torch.Tensor,
        placeholder_positions: Sequence[int],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        list[tuple[int, int]],
    ]:
        if inputs_embeds.ndim != 3 or grounding_tokens.ndim != 3:
            raise ValueError("inputs_embeds and grounding_tokens must be rank 3")
        if inputs_embeds.shape[0] != grounding_tokens.shape[0]:
            raise ValueError("grounding-token and language batch sizes differ")
        if len(placeholder_positions) != inputs_embeds.shape[0]:
            raise ValueError("one grounding placeholder position is required per sample")
        if grounding_attention_mask.shape != grounding_tokens.shape[:2]:
            raise ValueError(
                "grounding_attention_mask must match [batch, grounding tokens]"
            )
        grounding_tokens = grounding_tokens.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        grounding_length = grounding_tokens.shape[1]
        replaced_embeds = []
        replaced_labels = [] if labels is not None else None
        replaced_attention = [] if attention_mask is not None else None
        grounding_ranges = []
        for batch_index, position in enumerate(placeholder_positions):
            if not 0 <= position < inputs_embeds.shape[1]:
                raise ValueError("grounding placeholder lies outside the MLLM sequence")
            replaced_embeds.append(
                torch.cat(
                    [
                        inputs_embeds[batch_index, :position],
                        grounding_tokens[batch_index],
                        inputs_embeds[batch_index, position + 1:],
                    ],
                    dim=0,
                )
            )
            grounding_ranges.append((position, position + grounding_length))
            if labels is not None:
                ignored = torch.full(
                    (grounding_length,),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                replaced_labels.append(
                    torch.cat(
                        [
                            labels[batch_index, :position],
                            ignored,
                            labels[batch_index, position + 1:],
                        ],
                        dim=0,
                    )
                )
            if attention_mask is not None:
                visible = grounding_attention_mask[batch_index].to(
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                replaced_attention.append(
                    torch.cat(
                        [
                            attention_mask[batch_index, :position],
                            visible,
                            attention_mask[batch_index, position + 1:],
                        ],
                        dim=0,
                    )
                )
        inputs_embeds = torch.stack(replaced_embeds, dim=0)
        if replaced_labels is not None:
            labels = torch.stack(replaced_labels, dim=0)
        if replaced_attention is not None:
            attention_mask = torch.stack(replaced_attention, dim=0)
        return inputs_embeds, labels, attention_mask, grounding_ranges

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        grounding_images=None,
        grounding_valid_sizes=None,
        audio_features=None,
        target_frame=1,
        ref_ids=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            base = ChatUniViMetaForCausalLM.prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                audio_features=audio_features,
                target_frame=target_frame,
                ref_ids=ref_ids,
            )
            if len(base) == 5:
                # The original Chat-UniVi early-exit path does not return the
                # visual/audio ranges. Preserve the Triple caller's 9-item
                # contract by supplying all four unavailable range/map values.
                return (*base, None, None, None, None)
            if len(base) == 7:
                return (*base, None, None)
            raise RuntimeError(
                f"Unexpected Chat-UniVi fallback output count: {len(base)}"
            )

        if grounding_images is None:
            raise ValueError("grounding_images is required for triple-strategy preparation")
        if grounding_valid_sizes is None:
            raise ValueError("grounding_valid_sizes is required for masked grounding")

        placeholder_positions = (
            ChatUniViALS3PMetaForCausalLM._grounding_placeholder_positions(input_ids)
        )
        placeholder_mask = input_ids == GROUNDING_TOKEN_INDEX
        safe_token_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        base_input_ids = input_ids.masked_fill(placeholder_mask, safe_token_id)
        base_labels = labels
        if labels is not None:
            base_labels = labels.masked_fill(placeholder_mask, IGNORE_INDEX)

        base = ChatUniViMetaForCausalLM.prepare_inputs_labels_for_multimodal(
            self,
            base_input_ids,
            attention_mask,
            past_key_values,
            base_labels,
            images,
            audio_features=audio_features,
            target_frame=target_frame,
            ref_ids=ref_ids,
        )

        if len(base) == 5:
            (
                new_input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = base
            visual_ranges = None
            audio_ranges = None
        elif len(base) == 7:
            (
                new_input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                visual_ranges,
                audio_ranges,
            ) = base
        else:
            raise RuntimeError(
                f"Expected five or seven Chat-UniVi outputs, received {len(base)}"
            )
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        num_frames = images[0].shape[0] if isinstance(images, list) else images.shape[1]
        if isinstance(grounding_images, list):
            flat_grounding = torch.cat(grounding_images, dim=0)
        elif grounding_images.ndim == 5:
            flat_grounding = grounding_images.flatten(0, 1)
        else:
            raise ValueError(
                "grounding_images must be list[B][T,C,H,W] or [B,T,C,H,W]"
            )
        expected_frames = batch_size * num_frames
        if flat_grounding.shape[0] != expected_frames:
            raise ValueError(
                f"Grounding frame count {flat_grounding.shape[0]} != {expected_frames}"
            )
        grounding_features = vision_tower(flat_grounding, select_feature="patch")
        grid_size = int(getattr(self.config, "uground_grounding_grid_size", 8))
        grounding_tokens, grounding_grid_validity = (
            ChatUniViALS3PMetaForCausalLM._build_grounding_tokens(
                self,
                grounding_features,
                batch_size,
                num_frames,
                grid_size,
                grounding_valid_sizes,
                int(flat_grounding.shape[-1]),
            )
        )
        grounding_attention_mask = grounding_grid_validity.flatten(1) > 0
        inputs_embeds, labels, attention_mask, grounding_ranges = (
            ChatUniViALS3PMetaForCausalLM._replace_grounding_placeholders(
                inputs_embeds,
                labels,
                attention_mask,
                grounding_tokens,
                grounding_attention_mask,
                placeholder_positions,
            )
        )

        configured_maximum = int(
            getattr(self.config, "uground_max_sequence_length", 2048)
        )
        model_maximum = int(
            getattr(self.config, "max_position_embeddings", configured_maximum)
        )
        maximum_length = min(configured_maximum, model_maximum)
        if inputs_embeds.shape[1] > maximum_length:
            raise RuntimeError(
                "Aligned grounding tokens exceed the usable sequence length: "
                f"{inputs_embeds.shape[1]} > {maximum_length} "
                f"(configured={configured_maximum}, model={model_maximum}). "
                "Reduce --grounding_grid_size."
            )

        inserted_delta = grounding_tokens.shape[1] - 1
        visual_ranges = ChatUniViALS3PMetaForCausalLM._shift_ranges_after_grounding(
            visual_ranges, placeholder_positions, inserted_delta
        )
        audio_ranges = ChatUniViALS3PMetaForCausalLM._shift_ranges_after_grounding(
            audio_ranges, placeholder_positions, inserted_delta
        )
        return (
            new_input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            visual_ranges,
            audio_ranges,
            grounding_ranges,
            grounding_grid_validity,
        )
