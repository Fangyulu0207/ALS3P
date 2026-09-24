"""Full-information per-reference layer selector for ALS3P."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerSelection:
    hidden: torch.Tensor
    indices: torch.Tensor
    probabilities: torch.Tensor
    candidate_hidden: torch.Tensor


class FullInformationLayerSelector(nn.Module):
    """Hard-select one layer and learn from every candidate's map loss."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        candidate_layers: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        target_temperature: float = 1.0,
        standardize_targets: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or hidden_dim <= 0:
            raise ValueError("num_layers and hidden_dim must be positive")
        if temperature <= 0 or target_temperature <= 0:
            raise ValueError("selector temperatures must be positive")
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        candidates = self._validate_candidates(candidate_layers)
        self.register_buffer(
            "candidate_layers", torch.tensor(candidates, dtype=torch.long), persistent=True
        )
        self.layer_gate = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.register_buffer(
            "current_temperature",
            torch.tensor([temperature], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_temperature",
            torch.tensor([target_temperature], dtype=torch.float32),
            persistent=True,
        )
        self.standardize_targets = bool(standardize_targets)
        self._active_probabilities: Optional[torch.Tensor] = None
        self.last_probabilities: Optional[torch.Tensor] = None
        self.last_indices: Optional[torch.Tensor] = None
        self.last_predicted_indices: Optional[torch.Tensor] = None
        self.last_target_probabilities: Optional[torch.Tensor] = None
        self.last_oracle_indices: Optional[torch.Tensor] = None
        self.last_top1_accuracy: Optional[torch.Tensor] = None
        self.last_regret: Optional[torch.Tensor] = None
        self.last_oracle_loss: Optional[torch.Tensor] = None
        self.last_selected_loss: Optional[torch.Tensor] = None

    def _resolve_index(self, index: int) -> int:
        resolved = index if index >= 0 else self.num_layers + index
        if not 0 <= resolved < self.num_layers:
            raise ValueError(
                f"layer index {index} resolves outside [0, {self.num_layers})"
            )
        return resolved

    def _validate_candidates(
        self, candidate_layers: Optional[Sequence[int]]
    ) -> Tuple[int, ...]:
        if candidate_layers is None:
            return tuple(range(1, self.num_layers))
        resolved = tuple(dict.fromkeys(self._resolve_index(i) for i in candidate_layers))
        if not resolved:
            raise ValueError("candidate_layers cannot be empty")
        return resolved

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.layer_gate)
        self.clear_diagnostics()

    def clear_diagnostics(self) -> None:
        self._active_probabilities = None
        self.last_probabilities = None
        self.last_indices = None
        self.last_predicted_indices = None
        self.last_target_probabilities = None
        self.last_oracle_indices = None
        self.last_top1_accuracy = None
        self.last_regret = None
        self.last_oracle_loss = None
        self.last_selected_loss = None

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("selector temperature must be positive")
        self.current_temperature.fill_(float(temperature))

    def nonfinite_parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if not torch.isfinite(parameter).all()
        ]

    @staticmethod
    def extract_seg_hidden_states(
        hidden_states: Sequence[torch.Tensor], seg_token_mask: torch.Tensor
    ) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("hidden_states cannot be empty")
        stacked = torch.stack(tuple(hidden_states), dim=1)
        if stacked.ndim != 4:
            raise ValueError("hidden states must have shape [B, S, D] per layer")
        if seg_token_mask.shape != stacked.shape[:1] + stacked.shape[2:3]:
            raise ValueError(
                "seg_token_mask must match [B, S]; got "
                f"{tuple(seg_token_mask.shape)} for {tuple(stacked.shape)}"
            )
        return stacked.permute(0, 2, 1, 3)[seg_token_mask.bool()]

    def _candidate_probabilities(self, candidate_hidden: torch.Tensor) -> torch.Tensor:
        candidates = self.candidate_layers.to(candidate_hidden.device)
        selected_gate = self.layer_gate.index_select(0, candidates)
        normalized_hidden = F.layer_norm(
            candidate_hidden.detach().float(), (self.hidden_dim,)
        ).to(selected_gate.dtype)
        logits = torch.einsum("nkd,kd->nk", normalized_hidden, selected_gate)
        scale = math.sqrt(self.hidden_dim) * float(self.current_temperature.item())
        return torch.softmax((logits / scale).float(), dim=-1)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        seg_token_mask: torch.Tensor,
        force_last: bool = False,
    ) -> LayerSelection:
        self.clear_diagnostics()
        seg_hidden = self.extract_seg_hidden_states(hidden_states, seg_token_mask)
        if seg_hidden.shape[1] != self.num_layers:
            raise ValueError(
                f"configured for {self.num_layers} layers, received {seg_hidden.shape[1]}"
            )
        candidates = self.candidate_layers.to(seg_hidden.device)
        candidate_hidden = seg_hidden.index_select(1, candidates)
        if seg_hidden.shape[0] == 0:
            empty_indices = torch.empty(0, dtype=torch.long, device=seg_hidden.device)
            empty_probs = seg_hidden.new_empty((0, candidates.numel()), dtype=torch.float32)
            return LayerSelection(seg_hidden[:, 0], empty_indices, empty_probs, candidate_hidden)

        candidate_probabilities = self._candidate_probabilities(candidate_hidden)
        predicted_local = candidate_probabilities.argmax(dim=-1)
        predicted_indices = candidates[predicted_local]
        indices = (
            torch.full_like(predicted_indices, self.num_layers - 1)
            if force_last else predicted_indices
        )
        hidden = seg_hidden[
            torch.arange(seg_hidden.shape[0], device=seg_hidden.device), indices
        ]
        full_probabilities = torch.zeros(
            seg_hidden.shape[0], self.num_layers,
            dtype=candidate_probabilities.dtype, device=seg_hidden.device,
        )
        full_probabilities.scatter_(
            1, candidates.unsqueeze(0).expand_as(candidate_probabilities),
            candidate_probabilities,
        )
        self._active_probabilities = candidate_probabilities
        self.last_probabilities = full_probabilities.detach()
        self.last_indices = indices.detach()
        self.last_predicted_indices = predicted_indices.detach()
        return LayerSelection(hidden, indices, full_probabilities, candidate_hidden)

    def selector_loss(
        self,
        candidate_losses: torch.Tensor,
        valid_references: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy from detached all-layer map quality to selector probabilities."""
        if candidate_losses.ndim != 2:
            raise ValueError("candidate_losses must have shape [references, candidates]")
        if candidate_losses.shape[1] != self.candidate_layers.numel():
            raise ValueError("candidate loss count does not match candidate layers")
        predicted = self._active_probabilities
        if predicted is None or predicted.shape != candidate_losses.shape:
            raise RuntimeError("selector_loss requires a matching selector forward")
        if valid_references is None:
            valid = torch.ones(
                candidate_losses.shape[0], dtype=torch.bool, device=candidate_losses.device
            )
        else:
            valid = valid_references.to(candidate_losses.device, dtype=torch.bool).reshape(-1)
        if valid.shape[0] != candidate_losses.shape[0]:
            raise ValueError("valid reference count does not match candidate losses")
        if not bool(valid.any()):
            loss = predicted.sum() * 0.0
            zero = candidate_losses.new_zeros(())
            self.last_target_probabilities = None
            self.last_oracle_indices = self.candidate_layers.new_empty(0)
            self.last_top1_accuracy = zero
            self.last_regret = zero
            self.last_oracle_loss = zero
            self.last_selected_loss = zero
            self._active_probabilities = None
            return loss

        detached_losses = candidate_losses.detach().float()
        target_scores = detached_losses
        if self.standardize_targets and detached_losses.shape[1] > 1:
            mean = detached_losses.mean(dim=-1, keepdim=True)
            std = detached_losses.std(
                dim=-1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            target_scores = (detached_losses - mean) / std
        target = torch.softmax(
            -target_scores / float(self.target_temperature.item()), dim=-1
        )
        per_reference = -(
            target * predicted.float().clamp_min(1e-8).log()
        ).sum(dim=-1)
        loss = per_reference[valid].mean()

        predicted_local = predicted.detach().argmax(dim=-1)
        oracle_local = detached_losses.argmin(dim=-1)
        selected_loss = detached_losses.gather(1, predicted_local[:, None]).squeeze(1)
        oracle_loss = detached_losses.gather(1, oracle_local[:, None]).squeeze(1)
        candidates = self.candidate_layers.to(candidate_losses.device)
        self.last_target_probabilities = target[valid].detach()
        self.last_oracle_indices = candidates[oracle_local[valid]].detach()
        self.last_top1_accuracy = (
            (predicted_local[valid] == oracle_local[valid]).float().mean().detach()
        )
        self.last_regret = (selected_loss[valid] - oracle_loss[valid]).mean().detach()
        self.last_oracle_loss = oracle_loss[valid].mean().detach()
        self.last_selected_loss = selected_loss[valid].mean().detach()
        self._active_probabilities = None
        return loss

    def diagnostics(self, device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if self.last_probabilities is None or self.last_probabilities.numel() == 0:
            return {
                "selected_layer_mean": zero,
                "predicted_layer_mean": zero,
                "oracle_layer_mean": zero,
                "selector_entropy": zero,
                "selector_max_probability": zero,
                "selector_target_entropy": zero,
                "selector_target_max_probability": zero,
                "selector_top1_accuracy": zero,
                "selector_regret": zero,
                "selector_oracle_loss": zero,
                "selector_selected_loss": zero,
                "selector_temperature": self.current_temperature.squeeze(0).to(device),
            }
        candidates = self.candidate_layers.to(self.last_probabilities.device)
        predicted = self.last_probabilities.index_select(1, candidates).float()
        entropy = -(predicted * predicted.clamp_min(1e-8).log()).sum(dim=-1).mean()
        target_entropy = zero
        target_max = zero
        if (
            self.last_target_probabilities is not None
            and self.last_target_probabilities.numel() > 0
        ):
            target = self.last_target_probabilities.float()
            target_entropy = -(target * target.clamp_min(1e-8).log()).sum(dim=-1).mean()
            target_max = target.max(dim=-1).values.mean()
        return {
            "selected_layer_mean": self.last_indices.float().mean().to(device),
            "predicted_layer_mean": self.last_predicted_indices.float().mean().to(device),
            "oracle_layer_mean": (
                self.last_oracle_indices.float().mean().to(device)
                if self.last_oracle_indices is not None
                and self.last_oracle_indices.numel() > 0
                else zero
            ),
            "selector_entropy": entropy.to(device),
            "selector_max_probability": predicted.max(dim=-1).values.mean().to(device),
            "selector_target_entropy": target_entropy.to(device),
            "selector_target_max_probability": target_max.to(device),
            "selector_top1_accuracy": (
                self.last_top1_accuracy.to(device)
                if self.last_top1_accuracy is not None else zero
            ),
            "selector_regret": self.last_regret.to(device) if self.last_regret is not None else zero,
            "selector_oracle_loss": (
                self.last_oracle_loss.to(device) if self.last_oracle_loss is not None else zero
            ),
            "selector_selected_loss": (
                self.last_selected_loss.to(device)
                if self.last_selected_loss is not None else zero
            ),
            "selector_temperature": self.current_temperature.squeeze(0).to(device),
        }
