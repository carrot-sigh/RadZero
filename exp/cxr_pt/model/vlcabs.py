import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VLCABSOutput:
    similarity_scores: torch.Tensor
    masked_similarity_scores: torch.Tensor
    attention_weights: torch.Tensor
    evidence_tokens: torch.Tensor
    matching_logits: torch.Tensor


class VLCABS(nn.Module):
    """Similarity-based cross-attention with optional spatial constraints."""

    def __init__(self, sim_op="dot", empty_mask_policy="all"):
        super().__init__()
        if sim_op not in {"cos", "dot"}:
            raise ValueError(f"Unsupported similarity operation: {sim_op}")
        if empty_mask_policy not in {"all", "error"}:
            raise ValueError(f"Unsupported empty-mask policy: {empty_mask_policy}")
        self.sim_op = sim_op
        self.empty_mask_policy = empty_mask_policy

    def forward(
        self,
        queries: torch.Tensor,
        local_tokens: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
    ) -> VLCABSOutput:
        """
        Args:
            queries: Shared ``[Q, D]`` or batched ``[B, Q, D]`` queries.
            local_tokens: Visual tokens with shape ``[B, L, D]``.
            spatial_mask: Optional ``[B, L]`` or ``[B, Q, L]`` mask. Boolean
                masks select valid tokens. Floating masks are non-negative
                attention priors and are applied before softmax in log space.
            query_mask: Optional ``[Q]`` or ``[B, Q]`` validity mask.
            temperature: Required scalar temperature for cosine similarity.
        """
        query_features = self._prepare_queries(queries, local_tokens)

        if self.sim_op == "cos":
            if temperature is None:
                raise ValueError("Cosine similarity requires a temperature.")
            denominator = temperature
            query_features = F.normalize(query_features, p=2, dim=-1)
            attention_tokens = F.normalize(local_tokens, p=2, dim=-1)
        else:
            denominator = math.sqrt(local_tokens.size(-1))
            attention_tokens = local_tokens

        similarity_scores = (
            torch.bmm(query_features, attention_tokens.transpose(1, 2)) / denominator
        )
        masked_scores = self._apply_spatial_mask(similarity_scores, spatial_mask)
        attention_weights = F.softmax(masked_scores, dim=-1)
        evidence_tokens = torch.matmul(attention_weights, attention_tokens)

        normalized_queries = F.normalize(query_features, p=2, dim=-1)
        normalized_evidence = F.normalize(evidence_tokens, p=2, dim=-1)
        matching_logits = (
            torch.matmul(
                normalized_queries.unsqueeze(2), normalized_evidence.unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )

        if query_mask is not None:
            valid_queries = self._prepare_query_mask(
                query_mask,
                batch_size=local_tokens.size(0),
                query_count=query_features.size(1),
                device=local_tokens.device,
            )
            score_mask = valid_queries.unsqueeze(-1)
            similarity_scores = similarity_scores * score_mask
            masked_scores = masked_scores * score_mask
            attention_weights = attention_weights * score_mask
            evidence_tokens = evidence_tokens * score_mask
            matching_logits = matching_logits * valid_queries

        return VLCABSOutput(
            similarity_scores=similarity_scores,
            masked_similarity_scores=masked_scores,
            attention_weights=attention_weights,
            evidence_tokens=evidence_tokens,
            matching_logits=matching_logits,
        )

    @staticmethod
    def _prepare_queries(queries, local_tokens):
        if local_tokens.dim() != 3:
            raise ValueError("local_tokens must have shape [B, L, D].")
        if queries.dim() == 2:
            queries = queries.unsqueeze(0).expand(local_tokens.size(0), -1, -1)
        elif queries.dim() == 3:
            if queries.size(0) != local_tokens.size(0):
                raise ValueError("Batched queries and local_tokens must share B.")
        else:
            raise ValueError("queries must have shape [Q, D] or [B, Q, D].")
        if queries.size(-1) != local_tokens.size(-1):
            raise ValueError("Queries and local_tokens must share hidden size D.")
        return queries

    def _apply_spatial_mask(self, scores, spatial_mask):
        if spatial_mask is None:
            return scores

        mask = self._prepare_spatial_mask(spatial_mask, scores)
        if mask.dtype == torch.bool:
            empty_rows = ~mask.any(dim=-1, keepdim=True)
            mask = self._resolve_empty_rows(mask, empty_rows)
            return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        mask = mask.to(dtype=scores.dtype)
        if torch.any(mask < 0):
            raise ValueError("Floating spatial masks must be non-negative.")

        empty_rows = ~(mask > 0).any(dim=-1, keepdim=True)
        mask = self._resolve_empty_rows(mask, empty_rows)
        positive = mask > 0
        log_mask = torch.log(mask.clamp_min(torch.finfo(scores.dtype).tiny))
        masked_scores = scores + log_mask
        return masked_scores.masked_fill(~positive, torch.finfo(scores.dtype).min)

    @staticmethod
    def _prepare_spatial_mask(spatial_mask, scores):
        if spatial_mask.dim() == 2:
            spatial_mask = spatial_mask.unsqueeze(1)
        elif spatial_mask.dim() != 3:
            raise ValueError("spatial_mask must have shape [B, L] or [B, Q, L].")
        if spatial_mask.size(-1) != scores.size(-1):
            raise ValueError("spatial_mask and local_tokens must share L.")
        try:
            return torch.broadcast_to(spatial_mask.to(scores.device), scores.shape)
        except RuntimeError as error:
            raise ValueError(
                "spatial_mask cannot broadcast to shape [B, Q, L]."
            ) from error

    def _resolve_empty_rows(self, mask, empty_rows):
        if not torch.any(empty_rows):
            return mask
        if self.empty_mask_policy == "error":
            raise ValueError("spatial_mask contains an empty query ROI.")
        return torch.where(empty_rows, torch.ones_like(mask), mask)

    @staticmethod
    def _prepare_query_mask(query_mask, batch_size, query_count, device):
        if query_mask.dim() == 1:
            query_mask = query_mask.unsqueeze(0)
        elif query_mask.dim() != 2:
            raise ValueError("query_mask must have shape [Q] or [B, Q].")
        try:
            query_mask = torch.broadcast_to(
                query_mask.to(device=device, dtype=torch.bool),
                (batch_size, query_count),
            )
        except RuntimeError as error:
            raise ValueError("query_mask cannot broadcast to shape [B, Q].") from error
        return query_mask
