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


def make_config(hidden_dim=8, anatomies=3, findings=4, globals_=5):
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
            "anatomy_projection_dim": 4,
            "finding_chunk_size": 2,
            "aggregation_num_heads": 2,
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
        self.tokens = 7
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

    def test_original_forward_and_compute_logits_are_inherited(self):
        self.assertIs(HierarchicalCxrAlignModel.forward, CxrAlignModel.forward)
        self.assertIs(
            HierarchicalCxrAlignModel.compute_logits, CxrAlignModel.compute_logits
        )
        output = self.model(self.pixel_values, return_loss=False)
        self.assertEqual(output["vision_tokens"].shape, self.pixel_values.shape)

    def test_prompt_count_validation_is_clear(self):
        invalid_findings = {"features": torch.randn(2, self.hidden_dim)}
        with self.assertRaisesRegex(ValueError, "Expected 4 finding prompts"):
            self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                invalid_findings,
                self.encoded_global,
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
            self.model.anatomy_head.query_projection.weight.grad,
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

    def test_hierarchy_loss_requires_at_least_one_target(self):
        with self.assertRaisesRegex(ValueError, "At least one hierarchy target"):
            self.model.forward_hierarchy(
                self.pixel_values,
                self.encoded_anatomy,
                self.encoded_finding,
                self.encoded_global,
                return_loss=True,
            )


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
