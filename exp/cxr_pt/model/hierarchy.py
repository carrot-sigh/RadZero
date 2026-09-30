import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .vlcabs import VLCABS, VLCABSOutput


def _expand_queries(queries, batch_size, hidden_dim, name):
    if queries.dim() == 2:
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
    elif queries.dim() == 3 and queries.size(0) == batch_size:
        pass
    else:
        raise ValueError(f"{name} must have shape [Q, D] or [B, Q, D].")
    if queries.size(-1) != hidden_dim:
        raise ValueError(f"{name} must use hidden size {hidden_dim}.")
    return queries


def _expand_bool_mask(mask, shape, name, device):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    try:
        return torch.broadcast_to(mask.to(device=device, dtype=torch.bool), shape)
    except RuntimeError as error:
        raise ValueError(f"{name} cannot broadcast to {shape}.") from error


def _validate_temperature(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _bounded_temperature(log_temperature, minimum=1e-3, maximum=1.0):
    return log_temperature.clamp(math.log(minimum), math.log(maximum)).exp()


@dataclass
class AnatomyGroundingOutput:
    roi_mask_logits: torch.Tensor
    roi_masks: torch.Tensor
    similarity_scores: torch.Tensor
    attention_weights: torch.Tensor
    evidence_tokens: torch.Tensor
    matching_logits: torch.Tensor


class AnatomyGroundingHead(nn.Module):
    """Predict patch-level anatomy masks and retain VL-CABS evidence."""

    def __init__(
        self,
        hidden_dim,
        num_anatomies=None,
        attention_temperature=0.07,
        empty_mask_policy="all",
    ):
        super().__init__()
        _validate_temperature(attention_temperature, "attention_temperature")
        self.hidden_dim = hidden_dim
        parameter_count = num_anatomies or 1
        self.log_roi_scale = nn.Parameter(torch.zeros(parameter_count))
        self.roi_bias = nn.Parameter(torch.zeros(parameter_count))
        self.roi_refiner = None
        if num_anatomies is not None:
            self.roi_refiner = nn.Sequential(
                nn.Conv2d(num_anatomies, num_anatomies, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(num_anatomies, num_anatomies, kernel_size=1),
            )
            nn.init.zeros_(self.roi_refiner[-1].weight)
            nn.init.zeros_(self.roi_refiner[-1].bias)
        self.register_buffer(
            "log_attention_temperature",
            torch.tensor(math.log(attention_temperature)),
        )
        self.vlcabs = VLCABS(sim_op="cos", empty_mask_policy=empty_mask_policy)

    def forward(
        self,
        anatomy_queries,
        visual_tokens,
        anatomy_query_mask=None,
        visual_token_mask=None,
    ):
        if visual_tokens.dim() != 3 or visual_tokens.size(-1) != self.hidden_dim:
            raise ValueError("visual_tokens must have shape [B, L, hidden_dim].")
        batch_size, token_count, _ = visual_tokens.shape
        queries = _expand_queries(
            anatomy_queries, batch_size, self.hidden_dim, "anatomy_queries"
        )
        anatomy_count = queries.size(1)
        valid_queries = _expand_bool_mask(
            anatomy_query_mask,
            (batch_size, anatomy_count),
            "anatomy_query_mask",
            visual_tokens.device,
        )
        valid_tokens = _expand_bool_mask(
            visual_token_mask,
            (batch_size, token_count),
            "visual_token_mask",
            visual_tokens.device,
        )

        vlcabs_output = self.vlcabs(
            queries,
            visual_tokens,
            spatial_mask=valid_tokens,
            query_mask=valid_queries,
            temperature=_bounded_temperature(self.log_attention_temperature),
        )
        if self.log_roi_scale.numel() not in {1, anatomy_count}:
            raise ValueError("ROI parameters must be scalar or match anatomy count.")
        roi_scale = self.log_roi_scale.clamp(-10.0, 10.0).exp().view(1, -1, 1)
        roi_bias = self.roi_bias.view(1, -1, 1)
        roi_mask_logits = self._roi_logits_from_similarity(
            vlcabs_output.similarity_scores, roi_scale, roi_bias
        )
        roi_mask_logits = roi_mask_logits.masked_fill(
            ~valid_tokens.unsqueeze(1), torch.finfo(roi_mask_logits.dtype).min
        )
        roi_masks = torch.sigmoid(roi_mask_logits)

        query_multiplier = valid_queries.unsqueeze(-1)
        roi_mask_logits = roi_mask_logits * query_multiplier
        roi_masks = roi_masks * query_multiplier

        return AnatomyGroundingOutput(
            roi_mask_logits=roi_mask_logits,
            roi_masks=roi_masks,
            similarity_scores=vlcabs_output.similarity_scores,
            attention_weights=vlcabs_output.attention_weights,
            evidence_tokens=vlcabs_output.evidence_tokens,
            matching_logits=vlcabs_output.matching_logits,
        )

    def _roi_logits_from_similarity(self, similarity_scores, roi_scale, roi_bias):
        logits = roi_scale * similarity_scores + roi_bias
        if self.roi_refiner is None:
            return logits
        side = math.isqrt(similarity_scores.size(-1))
        if side * side != similarity_scores.size(-1):
            return logits
        spatial_scores = similarity_scores.reshape(
            similarity_scores.size(0), similarity_scores.size(1), side, side
        )
        refinement = self.roi_refiner(spatial_scores).flatten(2)
        return logits + refinement


@dataclass
class FindingRecognitionOutput:
    similarity_scores: torch.Tensor
    masked_similarity_scores: torch.Tensor
    attention_weights: torch.Tensor
    evidence_tokens: torch.Tensor
    matching_scores: torch.Tensor
    finding_logits: torch.Tensor
    pair_mask: torch.Tensor


class FeatureROIExtractor(nn.Module):
    """Extract fixed-resolution anatomy features from a full patch grid."""

    def __init__(self, output_size=14, context_margin=0.0, aligned=True):
        super().__init__()
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("Feature ROI output_size must be a positive integer.")
        if context_margin < 0:
            raise ValueError("Feature ROI context_margin must be non-negative.")
        self.output_size = output_size
        self.context_margin = float(context_margin)
        self.aligned = aligned

    def forward(self, visual_tokens, anatomy_boxes, anatomy_valid_mask=None):
        if visual_tokens.dim() != 3:
            raise ValueError("visual_tokens must have shape [B, L, D].")
        batch_size, token_count, hidden_dim = visual_tokens.shape
        grid_size = math.isqrt(token_count)
        if grid_size * grid_size != token_count:
            raise ValueError("Feature ROI requires a square visual-token grid.")
        if anatomy_boxes.dim() != 3 or anatomy_boxes.shape[0] != batch_size:
            raise ValueError("anatomy_boxes must have shape [B, A, 4].")
        if anatomy_boxes.size(-1) != 4:
            raise ValueError("anatomy_boxes must have shape [B, A, 4].")

        boxes = anatomy_boxes.to(device=visual_tokens.device, dtype=torch.float32)
        boxes = self._expand_and_clip(boxes, grid_size, grid_size)
        finite = torch.isfinite(boxes).all(dim=-1)
        valid = (
            finite & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
        )
        if anatomy_valid_mask is not None:
            valid = valid & _expand_bool_mask(
                anatomy_valid_mask,
                valid.shape,
                "anatomy_valid_mask",
                visual_tokens.device,
            )

        safe_boxes = boxes.clone()
        safe_boxes[~valid] = safe_boxes.new_tensor([0.0, 0.0, 1.0, 1.0])
        batch_indices = (
            torch.arange(batch_size, device=visual_tokens.device)
            .view(-1, 1)
            .expand_as(valid)
            .reshape(-1, 1)
            .to(dtype=safe_boxes.dtype)
        )
        rois = torch.cat([batch_indices, safe_boxes.reshape(-1, 4)], dim=-1)
        feature_map = visual_tokens.transpose(1, 2).reshape(
            batch_size, hidden_dim, grid_size, grid_size
        )
        local = roi_align(
            feature_map.float(),
            rois,
            output_size=(self.output_size, self.output_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=self.aligned,
        ).to(dtype=visual_tokens.dtype)
        local = (
            local.flatten(2)
            .transpose(1, 2)
            .reshape(
                batch_size,
                anatomy_boxes.size(1),
                self.output_size * self.output_size,
                hidden_dim,
            )
        )
        local = local * valid[..., None, None]
        return local, valid, boxes

    def _expand_and_clip(self, boxes, width, height):
        if self.context_margin:
            centers = (boxes[..., :2] + boxes[..., 2:]) / 2
            half_sizes = (boxes[..., 2:] - boxes[..., :2]) / 2
            half_sizes = half_sizes * (1.0 + self.context_margin)
            boxes = torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        return torch.stack(
            [
                x1.clamp(0, width),
                y1.clamp(0, height),
                x2.clamp(0, width),
                y2.clamp(0, height),
            ],
            dim=-1,
        )


class FindingLocalRefiner(nn.Module):
    """Small bottleneck spatial block for local anatomy feature grids."""

    def __init__(self, hidden_dim, bottleneck_dim=128):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.down_projection = nn.Linear(hidden_dim, bottleneck_dim)
        self.depthwise = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim,
        )
        self.channel_mixing = nn.Conv2d(bottleneck_dim, bottleneck_dim, 1)
        self.output_projection = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens, grid_size):
        if tokens.dim() != 3 or tokens.size(-1) != self.hidden_dim:
            raise ValueError("Local-refiner tokens must have shape [N, L, D].")
        if tokens.size(1) != grid_size * grid_size:
            raise ValueError("Local-refiner token count must match grid_size squared.")
        residual = tokens
        features = self.down_projection(self.input_norm(tokens))
        features = features.transpose(1, 2).reshape(
            tokens.size(0), -1, grid_size, grid_size
        )
        features = self.channel_mixing(F.gelu(self.depthwise(features)))
        features = features.flatten(2).transpose(1, 2)
        return residual + self.output_projection(F.gelu(features))


class AnatomyMaskedFindingHead(nn.Module):
    """Run VL-CABS for each valid anatomy-finding pair."""

    def __init__(
        self,
        hidden_dim,
        num_findings,
        attention_temperature=0.07,
        logit_temperature=0.07,
        finding_chunk_size=None,
        empty_mask_policy="all",
    ):
        super().__init__()
        _validate_temperature(attention_temperature, "attention_temperature")
        _validate_temperature(logit_temperature, "logit_temperature")
        if num_findings <= 0:
            raise ValueError("num_findings must be positive.")
        if finding_chunk_size is not None and finding_chunk_size <= 0:
            raise ValueError("finding_chunk_size must be positive when provided.")
        self.hidden_dim = hidden_dim
        self.num_findings = num_findings
        self.finding_chunk_size = finding_chunk_size
        self.log_attention_temperature = nn.Parameter(
            torch.tensor(math.log(attention_temperature))
        )
        self.log_logit_temperature = nn.Parameter(
            torch.tensor(math.log(logit_temperature))
        )
        self.finding_bias = nn.Parameter(torch.zeros(num_findings))
        self.vlcabs = VLCABS(sim_op="cos", empty_mask_policy=empty_mask_policy)

    def forward(
        self,
        finding_queries,
        visual_tokens,
        anatomy_masks,
        anatomy_query_mask=None,
        compatibility_mask=None,
        finding_chunk_size=None,
    ):
        if visual_tokens.dim() != 3 or visual_tokens.size(-1) != self.hidden_dim:
            raise ValueError("visual_tokens must have shape [B, L, hidden_dim].")
        batch_size, token_count, _ = visual_tokens.shape
        queries = _expand_queries(
            finding_queries, batch_size, self.hidden_dim, "finding_queries"
        )
        if queries.size(1) != self.num_findings:
            raise ValueError(
                f"Expected {self.num_findings} finding queries, got {queries.size(1)}."
            )
        if anatomy_masks.dim() != 3:
            raise ValueError("anatomy_masks must have shape [B, A, L].")
        if anatomy_masks.size(0) != batch_size or anatomy_masks.size(2) != token_count:
            raise ValueError(
                "anatomy_masks must match visual token B and L dimensions."
            )
        anatomy_count = anatomy_masks.size(1)

        valid_anatomies = _expand_bool_mask(
            anatomy_query_mask,
            (batch_size, anatomy_count),
            "anatomy_query_mask",
            visual_tokens.device,
        )
        valid_pairs = self._prepare_compatibility_mask(
            compatibility_mask,
            batch_size,
            anatomy_count,
            visual_tokens.device,
        )
        valid_pairs = valid_pairs & valid_anatomies.unsqueeze(-1)

        chunk_size = finding_chunk_size or self.finding_chunk_size or self.num_findings
        if chunk_size <= 0:
            raise ValueError("finding_chunk_size must be positive.")
        chunk_outputs = []
        for start in range(0, self.num_findings, chunk_size):
            stop = min(start + chunk_size, self.num_findings)
            chunk_outputs.append(
                self._forward_chunk(
                    queries[:, start:stop],
                    visual_tokens,
                    anatomy_masks,
                    valid_pairs[:, :, start:stop],
                )
            )

        similarity_scores = torch.cat(
            [output.similarity_scores for output in chunk_outputs], dim=2
        )
        masked_scores = torch.cat(
            [output.masked_similarity_scores for output in chunk_outputs], dim=2
        )
        attention_weights = torch.cat(
            [output.attention_weights for output in chunk_outputs], dim=2
        )
        evidence_tokens = torch.cat(
            [output.evidence_tokens for output in chunk_outputs], dim=2
        )
        matching_scores = torch.cat(
            [output.matching_logits for output in chunk_outputs], dim=2
        )
        finding_logits = matching_scores / _bounded_temperature(
            self.log_logit_temperature
        ) + self.finding_bias.view(1, 1, -1)
        finding_logits = finding_logits * valid_pairs

        return FindingRecognitionOutput(
            similarity_scores=similarity_scores,
            masked_similarity_scores=masked_scores,
            attention_weights=attention_weights,
            evidence_tokens=evidence_tokens,
            matching_scores=matching_scores,
            finding_logits=finding_logits,
            pair_mask=valid_pairs,
        )

    def forward_local(
        self,
        finding_queries,
        local_tokens,
        anatomy_valid_mask=None,
        compatibility_mask=None,
        finding_chunk_size=None,
    ):
        """Recognize findings from anatomy-specific local token grids."""
        if local_tokens.dim() != 4 or local_tokens.size(-1) != self.hidden_dim:
            raise ValueError("local_tokens must have shape [B, A, K, D].")
        batch_size, anatomy_count, token_count, _ = local_tokens.shape
        queries = _expand_queries(
            finding_queries, batch_size, self.hidden_dim, "finding_queries"
        )
        if queries.size(1) != self.num_findings:
            raise ValueError(
                f"Expected {self.num_findings} finding queries, got {queries.size(1)}."
            )
        valid_anatomies = _expand_bool_mask(
            anatomy_valid_mask,
            (batch_size, anatomy_count),
            "anatomy_valid_mask",
            local_tokens.device,
        )
        valid_pairs = self._prepare_compatibility_mask(
            compatibility_mask, batch_size, anatomy_count, local_tokens.device
        ) & valid_anatomies.unsqueeze(-1)
        chunk_size = finding_chunk_size or self.finding_chunk_size or self.num_findings
        outputs = []
        normalized_tokens = F.normalize(local_tokens, dim=-1)
        for start in range(0, self.num_findings, chunk_size):
            stop = min(start + chunk_size, self.num_findings)
            normalized_queries = F.normalize(queries[:, start:stop], dim=-1)
            similarity = torch.einsum(
                "bfd,bakd->bafk", normalized_queries, normalized_tokens
            )
            pair_mask = valid_pairs[:, :, start:stop]
            attention = torch.softmax(
                similarity / _bounded_temperature(self.log_attention_temperature),
                dim=-1,
            )
            attention = attention * pair_mask.unsqueeze(-1)
            evidence = torch.einsum("bafk,bakd->bafd", attention, local_tokens)
            matching = F.cosine_similarity(
                evidence,
                queries[:, None, start:stop, :],
                dim=-1,
            )
            outputs.append((similarity, attention, evidence, matching))

        similarity_scores = torch.cat([item[0] for item in outputs], dim=2)
        attention_weights = torch.cat([item[1] for item in outputs], dim=2)
        evidence_tokens = torch.cat([item[2] for item in outputs], dim=2)
        matching_scores = torch.cat([item[3] for item in outputs], dim=2)
        finding_logits = matching_scores / _bounded_temperature(
            self.log_logit_temperature
        ) + self.finding_bias.view(1, 1, -1)
        finding_logits = finding_logits * valid_pairs
        return FindingRecognitionOutput(
            similarity_scores=similarity_scores,
            masked_similarity_scores=similarity_scores,
            attention_weights=attention_weights,
            evidence_tokens=evidence_tokens,
            matching_scores=matching_scores,
            finding_logits=finding_logits,
            pair_mask=valid_pairs,
        )

    def _forward_chunk(self, queries, visual_tokens, anatomy_masks, pair_mask):
        batch_size, anatomy_count, token_count = anatomy_masks.shape
        finding_count = queries.size(1)
        pair_queries = (
            queries.unsqueeze(1)
            .expand(-1, anatomy_count, -1, -1)
            .reshape(batch_size, anatomy_count * finding_count, self.hidden_dim)
        )
        pair_masks = (
            anatomy_masks.unsqueeze(2)
            .expand(-1, -1, finding_count, -1)
            .reshape(batch_size, anatomy_count * finding_count, token_count)
        )
        flat_pair_mask = pair_mask.reshape(batch_size, anatomy_count * finding_count)
        outputs = self.vlcabs(
            pair_queries,
            visual_tokens,
            spatial_mask=pair_masks,
            query_mask=flat_pair_mask,
            temperature=_bounded_temperature(self.log_attention_temperature),
        )

        return VLCABSOutput(
            similarity_scores=outputs.similarity_scores.reshape(
                batch_size, anatomy_count, finding_count, token_count
            ),
            masked_similarity_scores=outputs.masked_similarity_scores.reshape(
                batch_size, anatomy_count, finding_count, token_count
            ),
            attention_weights=outputs.attention_weights.reshape(
                batch_size, anatomy_count, finding_count, token_count
            ),
            evidence_tokens=outputs.evidence_tokens.reshape(
                batch_size, anatomy_count, finding_count, self.hidden_dim
            ),
            matching_logits=outputs.matching_logits.reshape(
                batch_size, anatomy_count, finding_count
            ),
        )

    def _prepare_compatibility_mask(
        self, compatibility_mask, batch_size, anatomy_count, device
    ):
        target_shape = (batch_size, anatomy_count, self.num_findings)
        if compatibility_mask is None:
            return torch.ones(target_shape, dtype=torch.bool, device=device)
        if compatibility_mask.dim() == 2:
            compatibility_mask = compatibility_mask.unsqueeze(0)
        elif compatibility_mask.dim() != 3:
            raise ValueError("compatibility_mask must have shape [A, F] or [B, A, F].")
        return _expand_bool_mask(
            compatibility_mask,
            target_shape,
            "compatibility_mask",
            device,
        )


@dataclass
class FindingAggregationOutput:
    evidence_tokens: torch.Tensor
    attention_weights: torch.Tensor
    null_attention: torch.Tensor
    pair_mask: torch.Tensor


class FindingEvidenceAggregator(nn.Module):
    """Use global label queries to aggregate grounded finding evidence."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.score_bias_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        global_queries,
        finding_evidence,
        finding_scores=None,
        pair_mask=None,
        global_query_mask=None,
    ):
        if finding_evidence.dim() != 4:
            raise ValueError("finding_evidence must have shape [B, A, F, D].")
        batch_size, anatomy_count, finding_count, hidden_dim = finding_evidence.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"finding_evidence must use hidden size {self.hidden_dim}."
            )
        queries = _expand_queries(
            global_queries, batch_size, self.hidden_dim, "global_queries"
        )
        global_count = queries.size(1)
        pair_shape = (batch_size, anatomy_count, finding_count)
        valid_pairs = _expand_bool_mask(
            pair_mask, pair_shape, "pair_mask", finding_evidence.device
        )
        if finding_scores is None:
            finding_scores = finding_evidence.new_zeros(pair_shape)
        elif finding_scores.shape != pair_shape:
            raise ValueError("finding_scores must have shape [B, A, F].")

        flat_evidence = finding_evidence.reshape(batch_size, -1, hidden_dim)
        flat_scores = finding_scores.reshape(batch_size, -1)
        flat_pair_mask = valid_pairs.reshape(batch_size, -1)
        null_evidence = self.null_evidence.expand(batch_size, -1, -1)
        keys_and_values = torch.cat([flat_evidence, null_evidence], dim=1)

        minimum = torch.finfo(flat_scores.dtype).min
        score_bias = self.score_bias_scale * flat_scores
        score_bias = score_bias.masked_fill(~flat_pair_mask, minimum)
        score_bias = torch.cat([score_bias, score_bias.new_zeros(batch_size, 1)], dim=1)
        attention_mask = (
            score_bias[:, None, None, :]
            .expand(-1, self.num_heads, global_count, -1)
            .reshape(batch_size * self.num_heads, global_count, -1)
        )
        evidence_tokens, attention_weights = self.cross_attention(
            queries,
            keys_and_values,
            keys_and_values,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        valid_queries = _expand_bool_mask(
            global_query_mask,
            (batch_size, global_count),
            "global_query_mask",
            finding_evidence.device,
        )
        query_multiplier = valid_queries.unsqueeze(-1)
        evidence_tokens = evidence_tokens * query_multiplier
        attention_weights = attention_weights * query_multiplier
        pair_attention = attention_weights[..., :-1].reshape(
            batch_size, global_count, anatomy_count, finding_count
        )

        return FindingAggregationOutput(
            evidence_tokens=evidence_tokens,
            attention_weights=pair_attention,
            null_attention=attention_weights[..., -1],
            pair_mask=valid_pairs,
        )


class GlobalEvidenceBranch(nn.Module):
    """Extract label-specific evidence directly from global visual tokens."""

    def __init__(self, hidden_dim, attention_temperature=0.07, empty_mask_policy="all"):
        super().__init__()
        _validate_temperature(attention_temperature, "attention_temperature")
        self.hidden_dim = hidden_dim
        self.register_buffer(
            "log_attention_temperature",
            torch.tensor(math.log(attention_temperature)),
        )
        self.vlcabs = VLCABS(sim_op="cos", empty_mask_policy=empty_mask_policy)

    def forward(
        self,
        global_queries,
        visual_tokens,
        visual_token_mask=None,
        global_query_mask=None,
        temperature=None,
    ):
        if visual_tokens.dim() != 3 or visual_tokens.size(-1) != self.hidden_dim:
            raise ValueError("visual_tokens must have shape [B, L, hidden_dim].")
        if temperature is None:
            temperature = _bounded_temperature(self.log_attention_temperature)
        return self.vlcabs(
            global_queries,
            visual_tokens,
            spatial_mask=visual_token_mask,
            query_mask=global_query_mask,
            temperature=temperature,
        )


@dataclass
class GatedFusionOutput:
    finding_evidence: torch.Tensor
    global_evidence: torch.Tensor
    fused_evidence: torch.Tensor
    gates: torch.Tensor
    logits: torch.Tensor


class GatedEvidenceFusion(nn.Module):
    """Fuse finding and global evidence with sample-specific label gates."""

    def __init__(self, hidden_dim, num_global_labels, logit_temperature=0.07):
        super().__init__()
        _validate_temperature(logit_temperature, "logit_temperature")
        if num_global_labels <= 0:
            raise ValueError("num_global_labels must be positive.")
        self.hidden_dim = hidden_dim
        self.num_global_labels = num_global_labels
        self.finding_projection = nn.Linear(hidden_dim, hidden_dim)
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.log_logit_temperature = nn.Parameter(
            torch.tensor(math.log(logit_temperature))
        )
        self.label_bias = nn.Parameter(torch.zeros(num_global_labels))

    def forward(
        self,
        finding_evidence,
        global_evidence,
        global_queries,
        global_query_mask=None,
    ):
        if finding_evidence.shape != global_evidence.shape:
            raise ValueError("Finding and global evidence must have identical shapes.")
        if finding_evidence.dim() != 3 or finding_evidence.size(-1) != self.hidden_dim:
            raise ValueError("Evidence must have shape [B, J, hidden_dim].")
        batch_size, global_count, _ = finding_evidence.shape
        if global_count != self.num_global_labels:
            raise ValueError(
                f"Expected {self.num_global_labels} global labels, got {global_count}."
            )
        queries = _expand_queries(
            global_queries, batch_size, self.hidden_dim, "global_queries"
        )
        if queries.size(1) != global_count:
            raise ValueError("global_queries and evidence must share J.")

        projected_finding = self.finding_projection(finding_evidence)
        projected_global = self.global_projection(global_evidence)
        gates = torch.sigmoid(
            self.gate(torch.cat([projected_finding, projected_global, queries], dim=-1))
        )
        fused_evidence = gates * projected_finding + (1.0 - gates) * projected_global
        logits = F.cosine_similarity(fused_evidence, queries, dim=-1)
        logits = logits / _bounded_temperature(
            self.log_logit_temperature
        ) + self.label_bias.view(1, -1)

        valid_queries = _expand_bool_mask(
            global_query_mask,
            (batch_size, global_count),
            "global_query_mask",
            finding_evidence.device,
        )
        query_multiplier = valid_queries.unsqueeze(-1)
        projected_finding = projected_finding * query_multiplier
        projected_global = projected_global * query_multiplier
        fused_evidence = fused_evidence * query_multiplier
        gates = gates * query_multiplier
        logits = logits * valid_queries

        return GatedFusionOutput(
            finding_evidence=projected_finding,
            global_evidence=projected_global,
            fused_evidence=fused_evidence,
            gates=gates,
            logits=logits,
        )
