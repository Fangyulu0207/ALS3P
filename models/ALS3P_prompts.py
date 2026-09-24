"""Validity-aware prompts and all-layer map evaluation for ALS3P."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_paired_sam_prompts(
    decoder: nn.Module,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one prompt per image without SAM's Cartesian batch expansion."""
    prompt_batch = sparse_prompt_embeddings.shape[0]
    if image_embeddings.shape[0] != prompt_batch:
        raise ValueError("paired SAM decoding requires one image per sparse prompt")
    if dense_prompt_embeddings.shape[0] != prompt_batch:
        raise ValueError("paired SAM decoding requires one dense prompt per image")
    if image_pe.shape[0] == 1:
        position_embeddings = image_pe.expand(prompt_batch, -1, -1, -1)
    elif image_pe.shape[0] == prompt_batch:
        position_embeddings = image_pe
    else:
        raise ValueError("SAM positional encoding batch must be one or prompt batch")

    output_tokens = torch.cat(
        [decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0
    )
    output_tokens = output_tokens.unsqueeze(0).expand(prompt_batch, -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    source = image_embeddings + dense_prompt_embeddings
    batch, channels, height, width = source.shape
    hidden, source = decoder.transformer(source, position_embeddings, tokens)
    iou_token_output = hidden[:, 0, :]
    mask_tokens_output = hidden[:, 1 : 1 + decoder.num_mask_tokens, :]

    source = source.transpose(1, 2).view(batch, channels, height, width)
    upscaled_embedding = decoder.output_upscaling(source)
    hypernetwork_inputs = torch.stack(
        [
            decoder.output_hypernetworks_mlps[index](
                mask_tokens_output[:, index, :]
            )
            for index in range(decoder.num_mask_tokens)
        ],
        dim=1,
    )
    batch, channels, height, width = upscaled_embedding.shape
    masks = (
        hypernetwork_inputs @ upscaled_embedding.view(batch, channels, height * width)
    ).view(batch, decoder.num_mask_tokens, height, width)
    iou_predictions = decoder.iou_prediction_head(iou_token_output)

    mask_slice = slice(1, None) if multimask_output else slice(0, 1)
    return masks[:, mask_slice], iou_predictions[:, mask_slice]


class GaussianMapLoss(nn.Module):
    def __init__(
        self,
        kernel_size: int = 7,
        sigma: float = 2.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        coordinates = torch.arange(kernel_size, dtype=torch.float32)
        coordinates -= (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer("kernel", kernel_2d[None, None], persistent=True)
        self.padding = kernel_size // 2
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def softened_target(
        self,
        masks: torch.Tensor,
        target_size: Tuple[int, int],
        valid_sizes: Sequence[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if masks.ndim != 4:
            raise ValueError("masks must have shape [R, T, H, W]")
        references, frames = masks.shape[:2]
        if valid_sizes is not None:
            if len(valid_sizes) != frames:
                raise ValueError("one grounding valid size is required per frame")
            padded_frames = []
            padded_source_size = max(int(value) for value in valid_sizes[0])
            for frame_index, (valid_height, valid_width) in enumerate(valid_sizes):
                valid_height, valid_width = int(valid_height), int(valid_width)
                source_size = max(valid_height, valid_width)
                if source_size != padded_source_size:
                    raise ValueError("all frames must use the same padded CLIP size")
                frame_masks = F.interpolate(
                    masks[:, frame_index].float().unsqueeze(1),
                    size=(valid_height, valid_width),
                    mode="bilinear",
                    align_corners=False,
                )
                frame_masks = F.pad(
                    frame_masks,
                    (0, source_size - valid_width, 0, source_size - valid_height),
                )
                padded_frames.append(frame_masks)
            flattened = torch.stack(padded_frames, dim=1).reshape(
                references * frames, 1, padded_source_size, padded_source_size
            )
        else:
            flattened = masks.float().reshape(
                references * frames, 1, *masks.shape[-2:]
            )
        resized = F.interpolate(
            flattened, size=target_size, mode="bilinear", align_corners=False
        )
        kernel = self.kernel.to(device=resized.device, dtype=resized.dtype)
        softened = F.conv2d(resized, kernel, padding=self.padding)
        return softened.clamp_(0.0, 1.0).reshape(
            references, frames, *target_size
        )

    @staticmethod
    def _frame_dice_loss(
        probabilities: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probabilities = probabilities.flatten(2)
        targets = targets.flatten(2)
        numerator = 2.0 * (probabilities * targets).sum(dim=-1)
        denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
        return 1.0 - (numerator + 1e-6) / (denominator + 1e-6)

    def forward(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        valid_sizes: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [R, T, H, W]")
        targets = self.softened_target(
            masks,
            logits.shape[-2:],
            valid_sizes=valid_sizes,
        )
        probabilities = logits.float().clamp(1e-4, 1.0 - 1e-4)
        frame_bce = F.binary_cross_entropy(
            probabilities, targets, reduction="none"
        ).flatten(2).mean(dim=-1)
        frame_dice = self._frame_dice_loss(probabilities, targets)
        frame_total = self.bce_weight * frame_bce + self.dice_weight * frame_dice

        valid_frames = masks.float().flatten(2).sum(dim=-1) > 0
        valid_count = valid_frames.sum(dim=-1)
        denominator = valid_count.clamp_min(1).to(frame_total.dtype)
        valid_weight = valid_frames.to(frame_total.dtype)
        per_reference_total = (frame_total * valid_weight).sum(dim=-1) / denominator
        per_reference_bce = (frame_bce * valid_weight).sum(dim=-1) / denominator
        per_reference_dice = (frame_dice * valid_weight).sum(dim=-1) / denominator
        valid_references = valid_count > 0
        return (
            per_reference_total,
            per_reference_bce,
            per_reference_dice,
            targets,
            valid_frames,
            valid_references,
        )


class TripleGroundingMaskPrompt(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        grid_size: int = 10,
        prompt_size: int = 256,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or grid_size <= 0 or prompt_size <= 0:
            raise ValueError("hidden_dim, grid_size, and prompt_size must be positive")
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.prompt_size = prompt_size
        # Kept only for CLI/checkpoint compatibility. Released UGround uses
        # raw dot products followed by per-map min-max normalization.
        self.register_buffer(
            "configured_temperature",
            torch.tensor([initial_temperature], dtype=torch.float32),
            persistent=False,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.configured_temperature

    def reset_parameters(self, initial_temperature: float = 1.0) -> None:
        with torch.no_grad():
            self.configured_temperature.fill_(initial_temperature)

    def gather_same_layer_tokens(
        self,
        hidden_states: Sequence[torch.Tensor],
        grounding_ranges: Sequence[tuple[int, int]],
        selected_layers: torch.Tensor,
        layer_probabilities: torch.Tensor,
        refs_num: Sequence[int],
        num_frames: int,
        soft_selection: bool = False,
    ) -> list[torch.Tensor]:
        expected_tokens = num_frames * self.grid_size * self.grid_size
        if len(grounding_ranges) != len(refs_num):
            raise ValueError("one grounding range is required per batch sample")
        if selected_layers.numel() != sum(int(count) for count in refs_num):
            raise ValueError("selected layer count does not match refs_num")

        outputs = []
        reference_offset = 0
        for batch_index, ((start, end), reference_count) in enumerate(
            zip(grounding_ranges, refs_num)
        ):
            reference_count = int(reference_count)
            if end - start != expected_tokens:
                raise ValueError(
                    f"grounding range contains {end - start} tokens, expected {expected_tokens}"
                )
            sample_tokens = []
            for local_reference in range(reference_count):
                flat_reference = reference_offset + local_reference
                if soft_selection:
                    weights = layer_probabilities[flat_reference]
                    layer_tokens = torch.stack(
                        [state[batch_index, start:end] for state in hidden_states], dim=0
                    )
                    tokens = torch.einsum(
                        "l,lpd->pd", weights.to(layer_tokens.dtype), layer_tokens
                    )
                else:
                    layer = int(selected_layers[flat_reference].item())
                    tokens = hidden_states[layer][batch_index, start:end]
                sample_tokens.append(tokens)
            outputs.append(
                torch.stack(sample_tokens, dim=0).reshape(
                    reference_count,
                    num_frames,
                    self.grid_size * self.grid_size,
                    self.hidden_dim,
                )
            )
            reference_offset += reference_count
        return outputs

    def gather_candidate_layer_tokens(
        self,
        hidden_states: Sequence[torch.Tensor],
        grounding_ranges: Sequence[tuple[int, int]],
        candidate_layers: torch.Tensor,
        refs_num: Sequence[int],
        num_frames: int,
    ) -> list[torch.Tensor]:
        """Gather grounding states as [references, candidates, frames, patches, dim]."""
        expected_tokens = num_frames * self.grid_size * self.grid_size
        if len(grounding_ranges) != len(refs_num):
            raise ValueError("one grounding range is required per batch sample")
        candidate_indices = [int(index) for index in candidate_layers.tolist()]
        outputs = []
        for batch_index, ((start, end), reference_count) in enumerate(
            zip(grounding_ranges, refs_num)
        ):
            reference_count = int(reference_count)
            if end - start != expected_tokens:
                raise ValueError(
                    f"grounding range contains {end - start} tokens, expected {expected_tokens}"
                )
            layer_tokens = torch.stack(
                [hidden_states[layer][batch_index, start:end] for layer in candidate_indices],
                dim=0,
            ).reshape(
                len(candidate_indices),
                num_frames,
                self.grid_size * self.grid_size,
                self.hidden_dim,
            )
            outputs.append(
                layer_tokens.unsqueeze(0).expand(reference_count, -1, -1, -1, -1)
            )
        return outputs

    def candidate_similarity_logits(
        self,
        candidate_seg_hidden: torch.Tensor,
        candidate_grounding_hidden: torch.Tensor,
        valid_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute min-max normalized maps for all candidate layers in parallel."""
        if candidate_seg_hidden.ndim != 3:
            raise ValueError("candidate_seg_hidden must have shape [R, K, D]")
        if candidate_grounding_hidden.ndim != 5:
            raise ValueError(
                "candidate_grounding_hidden must have shape [R, K, T, P, D]"
            )
        if candidate_seg_hidden.shape[:2] != candidate_grounding_hidden.shape[:2]:
            raise ValueError("candidate SEG and grounding dimensions do not match")
        raw_similarity = torch.einsum(
            "rkd,rktpd->rktp",
            candidate_seg_hidden.float(),
            candidate_grounding_hidden.float(),
        )
        if valid_grid is None:
            valid = torch.ones_like(raw_similarity, dtype=torch.bool)
        else:
            frames = candidate_grounding_hidden.shape[2]
            if valid_grid.shape != (frames, self.grid_size, self.grid_size):
                raise ValueError(
                    "valid_grid must have shape [T, grid_size, grid_size]"
                )
            valid = valid_grid.reshape(1, 1, frames, -1) > 0
            valid = valid.expand(
                candidate_seg_hidden.shape[0], candidate_seg_hidden.shape[1], -1, -1
            )
        minimum = raw_similarity.masked_fill(~valid, float("inf")).amin(
            dim=-1, keepdim=True
        )
        maximum = raw_similarity.masked_fill(~valid, float("-inf")).amax(
            dim=-1, keepdim=True
        )
        similarity = (raw_similarity - minimum) / (maximum - minimum).clamp_min(1e-6)
        similarity = similarity.masked_fill(~valid, 0.0)
        return similarity.reshape(
            candidate_seg_hidden.shape[0],
            candidate_seg_hidden.shape[1],
            candidate_grounding_hidden.shape[2],
            self.grid_size,
            self.grid_size,
        )

    def similarity_logits(
        self,
        seg_hidden: torch.Tensor,
        grounding_hidden: torch.Tensor,
        valid_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seg_hidden.ndim != 2:
            raise ValueError("seg_hidden must have shape [R, D]")
        if grounding_hidden.ndim != 4:
            raise ValueError("grounding_hidden must have shape [R, T, P, D]")
        if seg_hidden.shape[0] != grounding_hidden.shape[0]:
            raise ValueError("reference counts do not match")
        if seg_hidden.shape[-1] != self.hidden_dim or grounding_hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden dimension does not match module configuration")
        raw_similarity = torch.einsum(
            "rd,rtpd->rtp", seg_hidden.float(), grounding_hidden.float()
        )
        if valid_grid is None:
            valid = torch.ones_like(raw_similarity, dtype=torch.bool)
        else:
            if valid_grid.shape != (
                grounding_hidden.shape[1], self.grid_size, self.grid_size
            ):
                raise ValueError(
                    "valid_grid must have shape [T, grid_size, grid_size]"
                )
            valid = valid_grid.reshape(1, grounding_hidden.shape[1], -1) > 0
            valid = valid.expand(seg_hidden.shape[0], -1, -1)
        minimum = raw_similarity.masked_fill(~valid, float("inf")).amin(
            dim=-1, keepdim=True
        )
        maximum = raw_similarity.masked_fill(~valid, float("-inf")).amax(
            dim=-1, keepdim=True
        )
        similarity = (raw_similarity - minimum) / (maximum - minimum).clamp_min(1e-6)
        similarity = similarity.masked_fill(~valid, 0.0)
        return similarity.reshape(
            seg_hidden.shape[0],
            grounding_hidden.shape[1],
            self.grid_size,
            self.grid_size,
        )

    def dense_prompt_masks(
        self,
        logits: torch.Tensor,
        valid_sizes: Sequence[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [R, T, H, W]")
        references, frames = logits.shape[:2]
        probabilities = logits.clamp(0.0, 1.0).reshape(
            references * frames, 1, *logits.shape[-2:]
        )
        dense = F.interpolate(
            probabilities,
            size=(self.prompt_size, self.prompt_size),
            mode="bilinear",
            align_corners=False,
        )
        if valid_sizes is not None:
            if len(valid_sizes) != frames:
                raise ValueError("one grounding valid size is required per frame")
            frame_validity = dense.new_zeros(
                frames, 1, self.prompt_size, self.prompt_size
            )
            for frame_index, (valid_height, valid_width) in enumerate(valid_sizes):
                source_size = max(int(valid_height), int(valid_width))
                if source_size <= 0:
                    raise ValueError("grounding valid sizes must be positive")
                prompt_height = min(
                    self.prompt_size,
                    int(round(int(valid_height) * self.prompt_size / source_size)),
                )
                prompt_width = min(
                    self.prompt_size,
                    int(round(int(valid_width) * self.prompt_size / source_size)),
                )
                frame_validity[frame_index, :, :prompt_height, :prompt_width] = 1
            validity = frame_validity.unsqueeze(0).expand(
                references, -1, -1, -1, -1
            ).reshape(references * frames, 1, self.prompt_size, self.prompt_size)
            dense = dense * validity
        return dense.reshape(
            references, frames, 1, self.prompt_size, self.prompt_size
        )
