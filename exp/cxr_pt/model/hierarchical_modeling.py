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
from .hierarchy_losses import HierarchyLoss
from .hierarchy_training import HierarchyTrainingController
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
    losses: dict = None

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
        "training_controller.",
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
        loss_config = hierarchy_config.get("loss", {})
        training_config = hierarchy_config.get("training", {})

        self.anatomy_head = AnatomyGroundingHead(
            self.hidden_size,
            num_anatomies=self.num_anatomies,
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
        self.hierarchy_loss = HierarchyLoss(**loss_config)
        self.training_controller = HierarchyTrainingController(**training_config)

    def forward(self, pixel_values, *args, **kwargs):
        """Route hierarchy batches through the standard parallel-model entry point."""
        if "encoded_anatomy_prompts" in kwargs:
            return self.forward_hierarchy(pixel_values, *args, **kwargs)
        return super().forward(pixel_values, *args, **kwargs)

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
        gt_anatomy_masks=None,
        teacher_forcing_ratio=None,
        detach_anatomy_masks=False,
        finding_chunk_size=None,
        anatomy_targets=None,
        finding_targets=None,
        global_targets=None,
        anatomy_annotation_mask=None,
        anatomy_valid_mask=None,
        finding_valid_mask=None,
        global_valid_mask=None,
        return_loss=False,
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
            global_visual_tokens=vision_outputs["vision_tokens"],
            anatomy_queries=anatomy_queries,
            finding_queries=finding_queries,
            global_queries=global_queries,
            vision_outputs=vision_outputs,
            anatomy_query_mask=anatomy_query_mask,
            global_query_mask=global_query_mask,
            visual_token_mask=visual_token_mask,
            compatibility_mask=compatibility_mask,
            anatomy_mask_override=anatomy_mask_override,
            gt_anatomy_masks=gt_anatomy_masks,
            teacher_forcing_ratio=teacher_forcing_ratio,
            detach_anatomy_masks=detach_anatomy_masks,
            finding_chunk_size=finding_chunk_size,
            anatomy_targets=anatomy_targets,
            finding_targets=finding_targets,
            global_targets=global_targets,
            anatomy_annotation_mask=anatomy_annotation_mask,
            anatomy_valid_mask=anatomy_valid_mask,
            finding_valid_mask=finding_valid_mask,
            global_valid_mask=global_valid_mask,
            return_loss=return_loss,
        )

    def forward_hierarchy_from_features(
        self,
        visual_tokens,
        anatomy_queries,
        finding_queries,
        global_queries,
        global_visual_tokens=None,
        vision_outputs=None,
        anatomy_query_mask=None,
        global_query_mask=None,
        visual_token_mask=None,
        compatibility_mask=None,
        anatomy_mask_override=None,
        gt_anatomy_masks=None,
        teacher_forcing_ratio=None,
        detach_anatomy_masks=False,
        finding_chunk_size=None,
        anatomy_targets=None,
        finding_targets=None,
        global_targets=None,
        anatomy_annotation_mask=None,
        anatomy_valid_mask=None,
        finding_valid_mask=None,
        global_valid_mask=None,
        return_loss=False,
    ):
        self._validate_feature_counts(anatomy_queries, finding_queries, global_queries)
        if global_visual_tokens is None:
            raise ValueError(
                "global_visual_tokens is required and must include the CLS token."
            )
        if (
            global_visual_tokens.dim() != 3
            or global_visual_tokens.size(0) != visual_tokens.size(0)
            or global_visual_tokens.size(2) != visual_tokens.size(2)
        ):
            raise ValueError(
                "global_visual_tokens must match visual_tokens batch and hidden dimensions."
            )
        if global_visual_tokens.size(1) != visual_tokens.size(1) + 1:
            raise ValueError(
                "global_visual_tokens must contain CLS followed by all patch tokens."
            )
        anatomy = self.anatomy_head(
            anatomy_queries,
            visual_tokens,
            anatomy_query_mask=anatomy_query_mask,
            visual_token_mask=visual_token_mask,
        )
        anatomy_masks_used = self._resolve_anatomy_masks(
            anatomy.roi_masks,
            anatomy_mask_override,
            gt_anatomy_masks,
            teacher_forcing_ratio,
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
        global_queries_for_branch, global_visual_tokens = (
            self._prepare_radzero_global_inputs(
                global_queries,
                global_visual_tokens,
            )
        )
        finding_aggregation = self.finding_aggregator(
            global_queries_for_branch,
            finding.evidence_tokens,
            finding_scores=finding.finding_logits,
            pair_mask=finding.pair_mask,
            global_query_mask=global_query_mask,
        )
        global_token_mask = self._global_visual_token_mask(
            visual_token_mask, visual_tokens, global_visual_tokens
        )
        global_evidence = self.global_branch(
            global_queries_for_branch,
            global_visual_tokens,
            visual_token_mask=global_token_mask,
            global_query_mask=global_query_mask,
            temperature=self._radzero_attention_temperature(),
        )
        fusion = self.evidence_fusion(
            finding_aggregation.evidence_tokens,
            global_evidence.evidence_tokens,
            global_queries_for_branch,
            global_query_mask=global_query_mask,
        )
        if vision_outputs is None:
            vision_outputs = {"image_patch_tokens": visual_tokens}

        losses = None
        if return_loss:
            if anatomy_annotation_mask is not None and anatomy_valid_mask is not None:
                if not torch.equal(anatomy_annotation_mask, anatomy_valid_mask):
                    raise ValueError(
                        "anatomy_annotation_mask and legacy anatomy_valid_mask disagree."
                    )
            anatomy_supervision_mask = (
                anatomy_annotation_mask
                if anatomy_annotation_mask is not None
                else anatomy_valid_mask
            )
            losses = self.hierarchy_loss(
                anatomy.roi_mask_logits,
                finding.finding_logits,
                fusion.logits,
                anatomy_targets=anatomy_targets,
                finding_targets=finding_targets,
                global_targets=global_targets,
                anatomy_valid_mask=anatomy_supervision_mask,
                finding_valid_mask=finding_valid_mask,
                global_valid_mask=global_valid_mask,
                anatomy_query_mask=anatomy_query_mask,
                visual_token_mask=visual_token_mask,
                finding_pair_mask=finding.pair_mask,
                global_query_mask=global_query_mask,
            ).as_dict()

        return HierarchicalModelOutput(
            vision_outputs=vision_outputs,
            anatomy=anatomy,
            finding=finding,
            finding_aggregation=finding_aggregation,
            global_evidence=global_evidence,
            fusion=fusion,
            anatomy_masks_used=anatomy_masks_used,
            losses=losses,
        )

    def _prepare_radzero_global_inputs(self, queries, visual_tokens):
        radzero_loss = self.loss_fns["RadZeroLoss"]
        if radzero_loss.layer_norm is not None:
            queries = radzero_loss.layer_norm(queries)
            visual_tokens = radzero_loss.layer_norm(visual_tokens)
        return queries, visual_tokens

    def _radzero_attention_temperature(self):
        radzero_loss = self.loss_fns["RadZeroLoss"]
        log_temperature = (
            radzero_loss.attn_temperature
            if radzero_loss.attn_temperature is not None
            else radzero_loss.loss_temperature
        )
        return log_temperature.clamp(-6.907755, 0.0).exp()

    @staticmethod
    def _global_visual_token_mask(patch_mask, patch_tokens, global_tokens):
        if patch_mask is None or global_tokens.size(1) == patch_tokens.size(1):
            return patch_mask
        if global_tokens.size(1) != patch_tokens.size(1) + 1:
            raise ValueError(
                "Global visual tokens must be patch tokens with optional CLS."
            )
        cls_mask = torch.ones(
            patch_mask.size(0), 1, dtype=torch.bool, device=patch_mask.device
        )
        return torch.cat([cls_mask, patch_mask.to(dtype=torch.bool)], dim=1)

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

    def _resolve_anatomy_masks(
        self,
        predicted_masks,
        override,
        ground_truth_masks,
        teacher_forcing_ratio,
        detach,
    ):
        if override is not None and (
            ground_truth_masks is not None or teacher_forcing_ratio is not None
        ):
            raise ValueError(
                "anatomy_mask_override cannot be combined with teacher forcing."
            )
        if override is None:
            override = self.training_controller.select_anatomy_masks(
                predicted_masks,
                ground_truth_masks,
                training=self.training,
                ratio=teacher_forcing_ratio,
            )
        return self._select_anatomy_masks(predicted_masks, override, detach)

    def configure_hierarchy_training(self, stage=None):
        modules = self.training_controller.configure_model(self, stage=stage)
        self.train(self.training)
        return modules

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module_name in self.training_controller.ENCODER_MODULES:
                module = getattr(self, module_name)
                if not any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self

    def set_hierarchy_training_step(self, step):
        self.training_controller.set_step(step)

    def get_teacher_forcing_ratio(self, step=None):
        return self.training_controller.teacher_forcing_ratio(step)

    @staticmethod
    def _positive_int(config, name):
        value = config.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"hierarchy_config.{name} must be a positive integer.")
        return value
