import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MPNetConfig, MPNetModel

from exp.cxr_pt.model.losses import RadZeroLoss, SimilarityLogit
from exp.cxr_pt.model.modeling import CxrAlignModel


def reference_similarity(queries, local_tokens, sim_op="dot", temperature=None):
    expanded_queries = queries.unsqueeze(0).expand(local_tokens.size(0), -1, -1)
    if sim_op == "cos":
        expanded_queries = F.normalize(expanded_queries, p=2, dim=-1)
        local_tokens = F.normalize(local_tokens, p=2, dim=-1)
        denominator = temperature
    else:
        denominator = math.sqrt(local_tokens.size(-1))
    scores = torch.bmm(expanded_queries, local_tokens.transpose(1, 2)) / denominator
    attention = F.softmax(scores, dim=-1)
    evidence = torch.matmul(attention, local_tokens)
    batch_query_logits = F.cosine_similarity(expanded_queries, evidence, dim=-1)
    return batch_query_logits.transpose(0, 1), scores


class ComputeLogitsHarness(nn.Module):
    compute_logits = CxrAlignModel.compute_logits

    def __init__(self, vision_tokens, query_features):
        super().__init__()
        hidden_dim = vision_tokens.size(-1)
        self.compute_logits_type = "radzero"
        self.loss_fns = nn.ModuleDict(
            {
                "RadZeroLoss": RadZeroLoss(
                    hidden_dim=hidden_dim,
                    use_vision_cls_token=False,
                    sim_op="cos",
                    use_layer_norm=False,
                )
            }
        )
        self.register_buffer("vision_tokens", vision_tokens)
        self.register_buffer("query_features", query_features)

    def forward_vision_model(self, pixel_values):
        return {"vision_tokens": self.vision_tokens}

    def forward_text_model(self, encoded_input):
        indices = encoded_input["input_ids"][:, 0]
        features = self.query_features[indices]
        return {
            "text_features_wo_l2_norm": features,
            "text_features": F.normalize(features, p=2, dim=-1),
        }


class SimilarityLogitRegressionTest(unittest.TestCase):
    def test_matches_reference_for_regular_batch(self):
        torch.manual_seed(7)
        queries = torch.randn(3, 8)
        local_tokens = torch.randn(2, 5, 8)

        logits, score_layers = SimilarityLogit(sim_op="dot")(
            queries, local_tokens, need_attn_weights=True
        )
        expected_logits, expected_scores = reference_similarity(queries, local_tokens)

        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(len(score_layers), 1)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(score_layers[0], expected_scores)

    def test_preserves_query_and_batch_dimensions(self):
        torch.manual_seed(11)
        module = SimilarityLogit(sim_op="dot")

        for batch_size, query_count in ((2, 3), (1, 3), (2, 1), (1, 1)):
            with self.subTest(batch_size=batch_size, query_count=query_count):
                queries = torch.randn(query_count, 8)
                local_tokens = torch.randn(batch_size, 5, 8)
                logits, _ = module(queries, local_tokens)
                self.assertEqual(logits.shape, (query_count, batch_size))


class ComputeLogitsRegressionTest(unittest.TestCase):
    def test_returns_batch_by_query_logits_for_singleton_dimensions(self):
        torch.manual_seed(19)

        for batch_size, query_count in ((2, 3), (1, 3), (2, 1), (1, 1)):
            with self.subTest(batch_size=batch_size, query_count=query_count):
                vision_tokens = torch.randn(batch_size, 5, 8)
                query_features = torch.randn(query_count, 8)
                model = ComputeLogitsHarness(vision_tokens, query_features)
                encoded_prompts = [
                    {
                        "input_ids": torch.arange(query_count).view(-1, 1),
                        "attention_mask": torch.ones(query_count, 1, dtype=torch.long),
                    }
                ]

                outputs = model.compute_logits(
                    pixel_values=torch.empty(batch_size, 0),
                    encoded_key_phrases=encoded_prompts,
                )
                patch_tokens = vision_tokens[:, 1:]
                temperature = model.loss_fns["RadZeroLoss"].loss_temperature.exp()
                expected_logits, expected_scores = reference_similarity(
                    query_features,
                    patch_tokens,
                    sim_op="cos",
                    temperature=temperature,
                )

                self.assertEqual(outputs["logits"].shape, (batch_size, query_count))
                self.assertEqual(
                    outputs["similarity_scores"].shape,
                    (batch_size, query_count, patch_tokens.size(1)),
                )
                torch.testing.assert_close(
                    outputs["logits"], expected_logits.transpose(0, 1) / temperature
                )
                torch.testing.assert_close(
                    outputs["similarity_scores"], expected_scores
                )


class TextEncodingRegressionTest(unittest.TestCase):
    def test_mpnet_mean_pooling_preserves_mixed_precision_dtype(self):
        text_model = MPNetModel(
            MPNetConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=16,
            )
        ).to(torch.bfloat16)
        harness = type("TextHarness", (), {})()
        harness.text_model = text_model
        harness.text_projector = None
        harness.config = type(
            "Config",
            (),
            {"text_config": type("TextConfig", (), {"use_cls_token": False})()},
        )()
        encoded = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }

        output = CxrAlignModel.forward_text_model(harness, encoded)

        self.assertEqual(output["text_features_wo_l2_norm"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
