from dataclasses import dataclass

import torch

from .hierarchy import (
    AnatomyGroundingHead,
    AnatomyGroundingOutput,
    AnatomyMaskedFindingHead,
    FindingAggregationOutput,
    FindingEvidenceAggregator,
    FindingRecognitionOutput,
    GatedEvidenceFusion,
    GatedFusionOutput,
    GlobalEvidenceBranch,
)
from .modeling import CxrAlignModel
from .vlcabs import VLCABSOutput


@dataclass
class HierarchicalModelOutput:
    vision_outputs: dict
    anatomy: AnatomyGroundingOutput
    finding: FindingRecognitionOutput
    finding_aggregation: FindingAggregationOutput
    global_evidence: VLCABSOutput
    fusion: GatedFusionOutput
    anatomy_masks_used: torch.Tensor

    @property
    def logits(self):
        return self.fusion.logits


class HierarchicalCxrAlignModel(CxrAlignModel):
    """RadZero with an Anatomy -> Finding -> Global reasoning path."""

    pretrained_missing_key_prefixes = (
        "anatomy_head.",
        "finding_head.",
        "finding_aggregator.",
        "global_branch.",
        "evidence_fusion.",
    )

    def __init__(self, config):
        super().__init__(config)
        hierarchy_config = config.kwargs.get("hierarchy_config")
        if hierarchy_config is None:
            raise ValueError("hierarchy_config is required for the hierarchy model.")

        self.num_anatomies = self._positive_int(hierarchy_config, "num_anatomies")
        self.num_findings = self._positive_int(hierarchy_config, "num_findings")
        self.num_global_labels = self._positive_int(
            hierarchy_config, "num_global_labels"
        )
        attention_temperature = hierarchy_config.get("attention_temperature", 0.07)
        empty_mask_policy = hierarchy_config.get("empty_mask_policy", "all")

        self.anatomy_head = AnatomyGroundingHead(
            self.hidden_size,
            projection_dim=hierarchy_config.get("anatomy_projection_dim"),
            attention_temperature=attention_temperature,
            empty_mask_policy=empty_mask_policy,
        )
        self.finding_head = AnatomyMaskedFindingHead(
            self.hidden_size,
            self.num_findings,
            attention_temperature=attention_temperature,
            logit_temperature=hierarchy_config.get("finding_logit_temperature", 0.07),
            finding_chunk_size=hierarchy_config.get("finding_chunk_size"),
            empty_mask_policy=empty_mask_policy,
        )
        self.finding_aggregator = FindingEvidenceAggregator(
            self.hidden_size,
            num_heads=hierarchy_config.get("aggregation_num_heads", 8),
            dropout=hierarchy_config.get("aggregation_dropout", 0.0),
        )
        self.global_branch = GlobalEvidenceBranch(
            self.hidden_size,
            attention_temperature=attention_temperature,
            empty_mask_policy=empty_mask_policy,
        )
        self.evidence_fusion = GatedEvidenceFusion(
            self.hidden_size,
            self.num_global_labels,
            logit_temperature=hierarchy_config.get("global_logit_temperature", 0.07),
        )

    def forward_hierarchy(
        self,
        pixel_values,
        encoded_anatomy_prompts,
        encoded_finding_prompts,
        encoded_global_prompts,
        anatomy_query_mask=None,
        global_query_mask=None,
        visual_token_mask=None,
        compatibility_mask=None,
        anatomy_mask_override=None,
        detach_anatomy_masks=False,
        finding_chunk_size=None,
    ):
        vision_outputs = self.forward_vision_model(pixel_values)
        anatomy_queries = self._encode_prompt_bank(
            encoded_anatomy_prompts, self.num_anatomies, "anatomy"
        )
        finding_queries = self._encode_prompt_bank(
            encoded_finding_prompts, self.num_findings, "finding"
        )
        global_queries = self._encode_prompt_bank(
            encoded_global_prompts, self.num_global_labels, "global"
        )
        return self.forward_hierarchy_from_features(
            visual_tokens=vision_outputs["image_patch_tokens"],
            anatomy_queries=anatomy_queries,
            finding_queries=finding_queries,
            global_queries=global_queries,
            vision_outputs=vision_outputs,
            anatomy_query_mask=anatomy_query_mask,
            global_query_mask=global_query_mask,
            visual_token_mask=visual_token_mask,
            compatibility_mask=compatibility_mask,
            anatomy_mask_override=anatomy_mask_override,
            detach_anatomy_masks=detach_anatomy_masks,
            finding_chunk_size=finding_chunk_size,
        )

    def forward_hierarchy_from_features(
        self,
        visual_tokens,
        anatomy_queries,
        finding_queries,
        global_queries,
        vision_outputs=None,
        anatomy_query_mask=None,
        global_query_mask=None,
        visual_token_mask=None,
        compatibility_mask=None,
        anatomy_mask_override=None,
        detach_anatomy_masks=False,
        finding_chunk_size=None,
    ):
        self._validate_feature_counts(anatomy_queries, finding_queries, global_queries)
        anatomy = self.anatomy_head(
            anatomy_queries,
            visual_tokens,
            anatomy_query_mask=anatomy_query_mask,
            visual_token_mask=visual_token_mask,
        )
        anatomy_masks_used = self._select_anatomy_masks(
            anatomy.roi_masks,
            anatomy_mask_override,
            detach_anatomy_masks,
        )
        finding = self.finding_head(
            finding_queries,
            visual_tokens,
            anatomy_masks_used,
            anatomy_query_mask=anatomy_query_mask,
            compatibility_mask=compatibility_mask,
            finding_chunk_size=finding_chunk_size,
        )
        finding_aggregation = self.finding_aggregator(
            global_queries,
            finding.evidence_tokens,
            finding_scores=finding.finding_logits,
            pair_mask=finding.pair_mask,
            global_query_mask=global_query_mask,
        )
        global_evidence = self.global_branch(
            global_queries,
            visual_tokens,
            visual_token_mask=visual_token_mask,
            global_query_mask=global_query_mask,
        )
        fusion = self.evidence_fusion(
            finding_aggregation.evidence_tokens,
            global_evidence.evidence_tokens,
            global_queries,
            global_query_mask=global_query_mask,
        )
        if vision_outputs is None:
            vision_outputs = {"image_patch_tokens": visual_tokens}

        return HierarchicalModelOutput(
            vision_outputs=vision_outputs,
            anatomy=anatomy,
            finding=finding,
            finding_aggregation=finding_aggregation,
            global_evidence=global_evidence,
            fusion=fusion,
            anatomy_masks_used=anatomy_masks_used,
        )

    def _encode_prompt_bank(self, encoded_prompts, expected_count, name):
        if encoded_prompts is None:
            raise ValueError(f"encoded_{name}_prompts is required.")
        outputs = self.forward_text_model(encoded_prompts)
        features = outputs["text_features_wo_l2_norm"]
        if features.size(-1) == 2 * self.hidden_size:
            features = features[:, self.hidden_size :]
        if features.dim() != 2 or features.size(-1) != self.hidden_size:
            raise ValueError(
                f"{name} prompt features must have shape [Q, {self.hidden_size}]."
            )
        if features.size(0) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {name} prompts, got {features.size(0)}."
            )
        return features

    def _validate_feature_counts(
        self, anatomy_queries, finding_queries, global_queries
    ):
        expected = (
            (anatomy_queries, self.num_anatomies, "anatomy"),
            (finding_queries, self.num_findings, "finding"),
            (global_queries, self.num_global_labels, "global"),
        )
        for queries, count, name in expected:
            if queries.dim() not in {2, 3} or queries.size(-2) != count:
                raise ValueError(f"Expected {count} {name} queries.")

    @staticmethod
    def _select_anatomy_masks(predicted_masks, override, detach):
        masks = predicted_masks
        if override is not None:
            if override.shape != predicted_masks.shape:
                raise ValueError(
                    "anatomy_mask_override must match predicted anatomy mask shape."
                )
            masks = override.to(
                device=predicted_masks.device, dtype=predicted_masks.dtype
            )
        if detach:
            masks = masks.detach()
        return masks

    @staticmethod
    def _positive_int(config, name):
        value = config.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"hierarchy_config.{name} must be a positive integer.")
        return value
