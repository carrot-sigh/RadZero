import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from exp.cxr_pt.model.checkpoint import (
    CURRENT_LOSS_PREFIX,
    RELEASED_LOSS_PREFIX,
    load_radzero_state_dict,
    migrate_released_radzero_state_dict,
)
from exp.cxr_pt.model.hierarchical_modeling import HierarchicalCxrAlignModel
from exp.cxr_pt.model.modeling import CxrAlignModel


class HierarchyHarness(HierarchicalCxrAlignModel):
    def build_vision_model(self, config):
        return nn.Identity()

    def build_text_model(self, config):
        return nn.Identity()

    def build_align_transformer_model(self, config):
        return nn.Identity()

    def __init__(self, config):
        super().__init__(config)
        self.vision_forward_calls = 0
        self.text_forward_calls = 0

    def forward_vision_model(self, pixel_values):
        self.vision_forward_calls += 1
        cls_token = pixel_values[:, 0]
        patch_tokens = pixel_values[:, 1:]
        image_features = F.normalize(
            torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1), dim=-1
        )
        return {
            "vision_tokens": pixel_values,
            "image_cls_token": cls_token,
            "image_patch_tokens": patch_tokens,
            "image_features": image_features,
        }

    def forward_text_model(self, encoded_input):
        self.text_forward_calls += 1
        features = encoded_input["features"]
        return {
            "text_features_wo_l2_norm": features,
            "text_features": F.normalize(features, dim=-1),
        }


def make_config(
    hidden_dim=8,
    anatomies=3,
    findings=4,
    globals_=5,
    finding_roi_strategy="masked_attention",
    global_only=False,
    finding_connection_mode="roi_constraint",
):
    config = PretrainedConfig()
    config.pretrained_dir = None
    config.vision_config = SimpleNamespace()
    config.text_config = SimpleNamespace(use_text_projection=False)
    config.align_transformer_config = SimpleNamespace(hidden_size=hidden_dim)
    config.kwargs = {
        "loss": {
            "apply": ["RadZeroLoss"],
            "ratio": [1.0],
            "RadZeroLoss": {
                "hidden_dim": hidden_dim,
                "use_layer_norm": True,
                "sim_op": "cos",
            },
        },
        "compute_logits_type": "radzero",
        "module_to_update": [],
        "hierarchy_config": {
            "num_anatomies": anatomies,
            "num_findings": findings,
            "num_global_labels": globals_,
            "finding_chunk_size": 2,
            "aggregation_num_heads": 2,
            "finding_roi_strategy": finding_roi_strategy,
            "finding_connection_mode": finding_connection_mode,
            "pair_query_num_heads": 2,
            "pair_query_chunk_size": 5,
            "pair_query_mlp_dim": 4,
            "global_only": global_only,
            "feature_roi_size": 2,
            "finding_local_refiner_dim": 4,
        },
    }
    return config


class HierarchicalModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.batch_size = 2
        self.anatomies = 3
        self.findings = 4
        self.globals = 5
        self.tokens = 9
        self.hidden_dim = 8
        self.model = HierarchyHarness(
            make_config(self.hidden_dim, self.anatomies, self.findings, self.globals)
        )
        self.pixel_values = torch.randn(
            self.batch_size, self.tokens + 1, self.hidden_dim
        )
        anatomy_features = torch.randn(self.anatomies, 2 * self.hidden_dim)
        self.encoded_anatomy = {"features": anatomy_features}
        self.encoded_finding = {"features": torch.randn(self.findings, self.hidden_dim)}
        self.encoded_global = {"features": torch.randn(self.globals, self.hidden_dim)}

    def test_hierarchy_forward_reuses_encoders_and_returns_contract(self):
        output = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )

        self.assertEqual(self.model.vision_forward_calls, 1)
        self.assertEqual(self.model.text_forward_calls, 3)
        self.assertEqual(
            output.anatomy.roi_masks.shape,
            (self.batch_size, self.anatomies, self.tokens),
        )
        self.assertEqual(
            output.finding.evidence_tokens.shape,
            (
                self.batch_size,
                self.anatomies,
                self.findings,
                self.hidden_dim,
            ),
        )
        self.assertEqual(output.logits.shape, (self.batch_size, self.globals))
        self.assertIs(output.logits, output.fusion.logits)
        torch.testing.assert_close(
            output.vision_outputs["image_patch_tokens"],
            self.pixel_values[:, 1:],
        )

    def test_dynamic_pair_query_uses_override_and_ignores_roi_inputs(self):
        model = HierarchyHarness(
            make_config(finding_connection_mode="dynamic_pair_query")
        )
        anatomy_queries = self.encoded_anatomy["features"][:, self.hidden_dim :]
        finding_queries = self.encoded_finding["features"]
        global_queries = self.encoded_global["features"]
        override = torch.randn(self.batch_size, self.anatomies, self.hidden_dim)
        output = model.forward_hierarchy_from_features(
            self.pixel_values[:, 1:],
            anatomy_queries,
            finding_queries,
            global_queries,
            global_visual_tokens=self.pixel_values,
            anatomy_state_override=override,
        )
        changed = model.forward_hierarchy_from_features(
            self.pixel_values[:, 1:],
            anatomy_queries,
            finding_queries,
            global_queries,
            global_visual_tokens=self.pixel_values,
            anatomy_state_override=override + 1,
        )
        self.assertEqual(
            output.finding.attention_weights.shape,
            (self.batch_size, self.anatomies, self.findings, self.tokens),
        )
        self.assertFalse(
            torch.allclose(output.finding.evidence_tokens, changed.finding.evidence_tokens)
        )
        with self.assertRaisesRegex(ValueError, "do not accept ROI"):
            model.forward_hierarchy_from_features(
                self.pixel_values[:, 1:],
                anatomy_queries,
                finding_queries,
                global_queries,
                global_visual_tokens=self.pixel_values,
                anatomy_state_override=override,
                anatomy_mask_override=torch.ones(
                    self.batch_size, self.anatomies, self.tokens
                ),
            )

    def test_static_and_dynamic_modes_match_with_same_state_and_weights(self):
        torch.manual_seed(77)
        dynamic = HierarchyHarness(
            make_config(finding_connection_mode="dynamic_pair_query")
        )
        torch.manual_seed(77)
        static = HierarchyHarness(
            make_config(finding_connection_mode="static_pair_query")
        )
        static.load_state_dict(dynamic.state_dict())
        state = torch.randn(self.batch_size, self.anatomies, self.hidden_dim)
        anatomy_queries = self.encoded_anatomy["features"][:, self.hidden_dim :]
        arguments = dict(
            visual_tokens=self.pixel_values[:, 1:],
            anatomy_queries=anatomy_queries,
            finding_queries=self.encoded_finding["features"],
            global_queries=self.encoded_global["features"],
            global_visual_tokens=self.pixel_values,
            anatomy_state_override=state,
        )
        dynamic.eval()
        static.eval()
        dynamic_output = dynamic.forward_hierarchy_from_features(**arguments)
        static_output = static.forward_hierarchy_from_features(**arguments)
        torch.testing.assert_close(
            dynamic_output.finding.finding_logits,
            static_output.finding.finding_logits,
        )
        torch.testing.assert_close(dynamic_output.logits, static_output.logits)

    def test_pair_query_training_freezes_stage1_and_keeps_original_stage3_scope(self):
        model = HierarchyHarness(
            make_config(finding_connection_mode="dynamic_pair_query")
        )
        finding_modules = set(model.configure_hierarchy_training("finding"))
        self.assertEqual(finding_modules, {"pair_query_finding_head"})
        joint_modules = set(model.configure_hierarchy_training("joint"))
        self.assertEqual(
            joint_modules,
            {
                "pair_query_finding_head",
                "finding_aggregator",
                "global_branch",
                "evidence_fusion",
            },
        )
        self.assertFalse(any(p.requires_grad for p in model.anatomy_head.parameters()))
        self.assertFalse(model.anatomy_head.training)

    def test_roi_constraint_does_not_read_pair_query_module(self):
        self.model.eval()
        before = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )
        with torch.no_grad():
            for parameter in self.model.pair_query_finding_head.parameters():
                parameter.normal_(mean=100, std=10)
        after = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )
        torch.testing.assert_close(before.finding.finding_logits, after.finding.finding_logits)
        torch.testing.assert_close(before.logits, after.logits)

    def test_pair_query_checkpoint_round_trip_reproduces_logits(self):
        model = HierarchyHarness(
            make_config(finding_connection_mode="static_pair_query")
        )
        state = torch.randn(self.anatomies, self.hidden_dim)
        model.pair_query_finding_head.set_static_anatomy_state(state)
        model.eval()
        before = model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )
        restored = HierarchyHarness(
            make_config(finding_connection_mode="static_pair_query")
        )
        restored.load_state_dict(model.state_dict())
        restored.eval()
        after = restored.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )
        torch.testing.assert_close(before.finding.finding_logits, after.finding.finding_logits)
        torch.testing.assert_close(before.logits, after.logits)

    def test_feature_roi_output_flows_into_unchanged_stage3(self):
        model = HierarchyHarness(
            make_config(
                self.hidden_dim,
                self.anatomies,
                self.findings,
                self.globals,
                finding_roi_strategy="feature_roi",
            )
        )
        boxes = torch.tensor([0.0, 0.0, 3.0, 3.0]).repeat(
            self.batch_size, self.anatomies, 1
        )
        output = model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
            anatomy_boxes=boxes,
            anatomy_annotation_mask=torch.ones(
                self.batch_size, self.anatomies, dtype=torch.bool
            ),
        )

        self.assertEqual(
            output.finding.attention_weights.shape,
            (self.batch_size, self.anatomies, self.findings, 4),
        )
        self.assertEqual(
            output.finding_aggregation.evidence_tokens.shape,
            (self.batch_size, self.globals, self.hidden_dim),
        )
        self.assertEqual(output.fusion.logits.shape, (self.batch_size, self.globals))

    def test_feature_roi_inference_derives_boxes_from_predicted_masks(self):
        model = HierarchyHarness(
            make_config(
                self.hidden_dim,
                self.anatomies,
                self.findings,
                self.globals,
                finding_roi_strategy="feature_roi",
            )
        )

        output = model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )

        self.assertEqual(
            output.finding.attention_weights.shape,
            (self.batch_size, self.anatomies, self.findings, 4),
        )
        self.assertTrue(torch.isfinite(output.finding.finding_logits).all())

    def test_feature_roi_stage_trains_refiner_but_keeps_encoders_frozen(self):
        model = HierarchyHarness(make_config(finding_roi_strategy="feature_roi"))
        configured = set(model.configure_hierarchy_training("finding"))

        self.assertEqual(configured, {"finding_head", "finding_local_refiner"})
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.finding_local_refiner.parameters()
            )
        )
        for module_name in model.training_controller.ENCODER_MODULES:
            module = getattr(model, module_name)
            self.assertFalse(any(p.requires_grad for p in module.parameters()))
            self.assertFalse(module.training)

    def test_mask_override_and_detach_are_explicit(self):
        override = torch.rand(
            self.batch_size, self.anatomies, self.tokens, dtype=torch.float64
        )
        output = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
            anatomy_mask_override=override,
            detach_anatomy_masks=True,
        )

        torch.testing.assert_close(
            output.anatomy_masks_used, override.to(self.pixel_values.dtype)
        )
        self.assertFalse(output.anatomy_masks_used.requires_grad)

    def test_annotation_presence_does_not_change_inference(self):
        self.model.eval()
        first = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
            anatomy_annotation_mask=torch.ones(
                self.batch_size, self.anatomies, dtype=torch.bool
            ),
        )
        second = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
            anatomy_annotation_mask=torch.zeros(
                self.batch_size, self.anatomies, dtype=torch.bool
            ),
        )

        for first_tensor, second_tensor in (
            (first.anatomy.roi_masks, second.anatomy.roi_masks),
            (first.finding.finding_logits, second.finding.finding_logits),
            (first.finding.attention_weights, second.finding.attention_weights),
            (first.finding.evidence_tokens, second.finding.evidence_tokens),
            (
                first.global_evidence.evidence_tokens,
                second.global_evidence.evidence_tokens,
            ),
            (first.fusion.logits, second.fusion.logits),
        ):
            torch.testing.assert_close(first_tensor, second_tensor)

    def test_original_forward_and_compute_logits_are_inherited(self):
        self.assertIs(
            HierarchicalCxrAlignModel.compute_logits, CxrAlignModel.compute_logits
        )
        output = self.model(self.pixel_values, return_loss=False)
        self.assertEqual(output["vision_tokens"].shape, self.pixel_values.shape)

        hierarchy_output = self.model(
            self.pixel_values,
            encoded_anatomy_prompts=self.encoded_anatomy,
            encoded_finding_prompts=self.encoded_finding,
            encoded_global_prompts=self.encoded_global,
            return_loss=False,
        )
        self.assertEqual(hierarchy_output.logits.shape, (self.batch_size, self.globals))

    def test_prompt_count_validation_is_clear(self):
        invalid_findings = {"features": torch.randn(2, self.hidden_dim)}
        with self.assertRaisesRegex(ValueError, "Expected 4 finding prompts"):
            self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                invalid_findings,
                self.encoded_global,
            )

    def test_feature_api_requires_full_visual_tokens_with_cls(self):
        patch_tokens = self.pixel_values[:, 1:]
        anatomy_queries = self.encoded_anatomy["features"][:, self.hidden_dim :]
        finding_queries = self.encoded_finding["features"]
        global_queries = self.encoded_global["features"]

        with self.assertRaisesRegex(ValueError, "include the CLS token"):
            self.model.forward_hierarchy_from_features(
                patch_tokens,
                anatomy_queries,
                finding_queries,
                global_queries,
            )
        with self.assertRaisesRegex(ValueError, "CLS followed by all patch tokens"):
            self.model.forward_hierarchy_from_features(
                patch_tokens,
                anatomy_queries,
                finding_queries,
                global_queries,
                global_visual_tokens=patch_tokens,
            )

        output = self.model.forward_hierarchy_from_features(
            patch_tokens,
            anatomy_queries,
            finding_queries,
            global_queries,
            global_visual_tokens=self.pixel_values,
        )
        self.assertEqual(
            output.global_evidence.attention_weights.size(-1), self.tokens + 1
        )

    def test_hierarchy_loss_is_attached_and_backpropagates(self):
        anatomy_targets = torch.randint(
            0, 2, (self.batch_size, self.anatomies, self.tokens)
        ).float()
        finding_targets = torch.randint(
            0, 2, (self.batch_size, self.anatomies, self.findings)
        ).float()
        global_targets = torch.randint(0, 2, (self.batch_size, self.globals)).float()
        finding_valid = torch.ones_like(finding_targets, dtype=torch.bool)
        finding_valid[0, 0, 0] = False
        finding_targets[0, 0, 0] = torch.nan

        output = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
            anatomy_targets=anatomy_targets,
            finding_targets=finding_targets,
            global_targets=global_targets,
            finding_valid_mask=finding_valid,
            return_loss=True,
        )
        output.losses["loss"].backward()

        self.assertEqual(
            set(output.losses),
            {
                "loss",
                "anatomy_loss",
                "anatomy_bce_loss",
                "anatomy_dice_loss",
                "finding_loss",
                "global_loss",
            },
        )
        for gradient in (
            self.model.anatomy_head.log_roi_scale.grad,
            self.model.anatomy_head.roi_bias.grad,
            self.model.finding_head.finding_bias.grad,
            self.model.evidence_fusion.label_bias.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in self.model.anatomy_head.parameters()
            )
        )

    def test_global_branch_matches_original_radzero_vlcabs(self):
        output = self.model.forward_hierarchy(
            self.pixel_values,
            self.encoded_anatomy,
            self.encoded_finding,
            self.encoded_global,
        )
        queries = self.model._encode_prompt_bank(
            self.encoded_global, self.globals, "global"
        )
        queries, tokens = self.model._prepare_radzero_global_inputs(
            queries, self.pixel_values
        )
        reference = self.model.loss_fns["RadZeroLoss"].similarity_logit.vlcabs(
            queries,
            tokens,
            temperature=self.model._radzero_attention_temperature(),
        )

        torch.testing.assert_close(
            output.global_evidence.similarity_scores, reference.similarity_scores
        )
        torch.testing.assert_close(
            output.global_evidence.attention_weights, reference.attention_weights
        )
        torch.testing.assert_close(
            output.global_evidence.evidence_tokens, reference.evidence_tokens
        )
        torch.testing.assert_close(
            output.global_evidence.matching_logits, reference.matching_logits
        )
        self.assertEqual(
            output.global_evidence.attention_weights.size(-1), self.tokens + 1
        )

    def test_eval_forward_is_invariant_to_ground_truth_fields(self):
        self.model.eval()
        with torch.no_grad():
            reference = self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                self.encoded_finding,
                self.encoded_global,
            )
            perturbed = self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                self.encoded_finding,
                self.encoded_global,
                gt_anatomy_masks=torch.rand_like(reference.anatomy.roi_masks),
                anatomy_boxes=torch.rand(
                    self.batch_size, self.anatomies, 4, device=self.pixel_values.device
                ),
                anatomy_targets=torch.rand(
                    self.batch_size, self.anatomies, self.tokens,
                    device=self.pixel_values.device,
                ),
                finding_targets=torch.rand(
                    self.batch_size, self.anatomies, self.findings,
                    device=self.pixel_values.device,
                ),
                global_targets=torch.rand(
                    self.batch_size, self.globals, device=self.pixel_values.device
                ),
            )
        torch.testing.assert_close(
            reference.finding.finding_logits, perturbed.finding.finding_logits
        )
        torch.testing.assert_close(reference.fusion.logits, perturbed.fusion.logits)

    def test_hierarchy_loss_requires_at_least_one_target(self):
        with self.assertRaisesRegex(ValueError, "At least one hierarchy target"):
            self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                self.encoded_finding,
                self.encoded_global,
                return_loss=True,
            )


class GlobalOnlyPilotControlTest(unittest.TestCase):
    def test_global_only_freezes_finding_aggregator_and_neutralizes_branch(self):
        model = HierarchyHarness(make_config(global_only=True))
        modules = model.configure_hierarchy_training("global")
        self.assertNotIn("finding_aggregator", modules)
        self.assertTrue(model.global_only)
        self.assertTrue(all(not p.requires_grad for p in model.finding_aggregator.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.evidence_fusion.parameters()))


class CheckpointMigrationTest(unittest.TestCase):
    def test_released_loss_prefix_is_migrated(self):
        state_dict = {
            RELEASED_LOSS_PREFIX + "loss_temperature": torch.tensor([1.0]),
            "vision_model.weight": torch.tensor([2.0]),
        }
        migrated = migrate_released_radzero_state_dict(state_dict)

        self.assertIn(CURRENT_LOSS_PREFIX + "loss_temperature", migrated)
        self.assertNotIn(RELEASED_LOSS_PREFIX + "loss_temperature", migrated)
        self.assertIn("vision_model.weight", migrated)

    def test_loader_allows_only_new_hierarchy_parameters_to_be_missing(self):
        torch.manual_seed(43)
        source = HierarchyHarness(make_config())
        released_state = {}
        for key, value in source.state_dict().items():
            if key.startswith(source.pretrained_missing_key_prefixes):
                continue
            released_key = key.replace(CURRENT_LOSS_PREFIX, RELEASED_LOSS_PREFIX)
            released_state[released_key] = value.clone()

        torch.manual_seed(47)
        target = HierarchyHarness(make_config())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(released_state, checkpoint)
            incompatible = load_radzero_state_dict(target, checkpoint)

        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith(target.pretrained_missing_key_prefixes)
                for key in incompatible.missing_keys
            )
        )
        target_state = target.state_dict()
        source_state = source.state_dict()
        for key in source_state:
            if not key.startswith(source.pretrained_missing_key_prefixes):
                torch.testing.assert_close(target_state[key], source_state[key])


if __name__ == "__main__":
    unittest.main()
