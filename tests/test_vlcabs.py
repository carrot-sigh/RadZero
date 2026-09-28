import unittest

import torch
import torch.nn.functional as F

from exp.cxr_pt.model.losses import SimilarityLogit
from exp.cxr_pt.model.vlcabs import VLCABS, VLCABSOutput


class VLCABSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.queries = torch.randn(3, 8)
        self.local_tokens = torch.randn(2, 5, 8)

    def test_structured_output_matches_legacy_adapter(self):
        core_output = VLCABS(sim_op="dot")(self.queries, self.local_tokens)
        legacy_logits, legacy_scores = SimilarityLogit(sim_op="dot")(
            self.queries,
            self.local_tokens,
            need_attn_weights=True,
        )

        self.assertIsInstance(core_output, VLCABSOutput)
        self.assertEqual(core_output.similarity_scores.shape, (2, 3, 5))
        self.assertEqual(core_output.masked_similarity_scores.shape, (2, 3, 5))
        self.assertEqual(core_output.attention_weights.shape, (2, 3, 5))
        self.assertEqual(core_output.evidence_tokens.shape, (2, 3, 8))
        self.assertEqual(core_output.matching_logits.shape, (2, 3))
        torch.testing.assert_close(
            core_output.matching_logits.transpose(0, 1), legacy_logits
        )
        torch.testing.assert_close(core_output.similarity_scores, legacy_scores[0])

    def test_boolean_mask_excludes_tokens_before_softmax(self):
        spatial_mask = torch.tensor(
            [[True, False, True, False, True], [False, True, True, False, False]]
        )
        output = VLCABS(sim_op="dot")(
            self.queries, self.local_tokens, spatial_mask=spatial_mask
        )
        expanded_mask = spatial_mask.unsqueeze(1).expand_as(output.attention_weights)

        self.assertTrue(torch.all(output.attention_weights[~expanded_mask] == 0))
        torch.testing.assert_close(
            output.attention_weights.sum(dim=-1), torch.ones(2, 3)
        )
        torch.testing.assert_close(
            output.similarity_scores,
            VLCABS(sim_op="dot")(self.queries, self.local_tokens).similarity_scores,
        )

    def test_soft_mask_is_applied_as_attention_prior(self):
        queries = self.queries[:1]
        tokens = self.local_tokens[:1, :3]
        spatial_mask = torch.tensor([[1.0, 0.25, 0.0]])
        output = VLCABS(sim_op="dot")(queries, tokens, spatial_mask=spatial_mask)
        expected_scores = output.similarity_scores + torch.log(
            spatial_mask.unsqueeze(1).clamp_min(torch.finfo(torch.float32).tiny)
        )
        expected_scores[..., 2] = torch.finfo(torch.float32).min

        torch.testing.assert_close(output.masked_similarity_scores, expected_scores)
        torch.testing.assert_close(
            output.attention_weights, F.softmax(expected_scores, dim=-1)
        )
        self.assertEqual(output.attention_weights[..., 2].item(), 0.0)

    def test_empty_roi_falls_back_to_all_tokens(self):
        unmasked = VLCABS(sim_op="dot")(self.queries, self.local_tokens)
        empty_mask = torch.zeros(2, 3, 5, dtype=torch.bool)
        fallback = VLCABS(sim_op="dot", empty_mask_policy="all")(
            self.queries, self.local_tokens, spatial_mask=empty_mask
        )

        torch.testing.assert_close(
            fallback.attention_weights, unmasked.attention_weights
        )
        torch.testing.assert_close(fallback.evidence_tokens, unmasked.evidence_tokens)
        torch.testing.assert_close(fallback.matching_logits, unmasked.matching_logits)

    def test_empty_roi_can_raise(self):
        empty_mask = torch.zeros(2, 3, 5, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "empty query ROI"):
            VLCABS(sim_op="dot", empty_mask_policy="error")(
                self.queries, self.local_tokens, spatial_mask=empty_mask
            )

    def test_negative_soft_mask_is_rejected(self):
        invalid_mask = torch.ones(2, 3, 5)
        invalid_mask[0, 0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            VLCABS(sim_op="dot")(
                self.queries, self.local_tokens, spatial_mask=invalid_mask
            )

    def test_batched_queries_and_query_mask(self):
        batched_queries = torch.randn(2, 3, 8)
        query_mask = torch.tensor([[True, False, True], [False, True, True]])
        output = VLCABS(sim_op="dot")(
            batched_queries,
            self.local_tokens,
            query_mask=query_mask,
        )

        self.assertEqual(output.matching_logits.shape, (2, 3))
        self.assertTrue(torch.all(output.matching_logits[~query_mask] == 0))
        self.assertTrue(torch.all(output.attention_weights[~query_mask] == 0))
        self.assertTrue(torch.all(output.evidence_tokens[~query_mask] == 0))
        torch.testing.assert_close(
            output.attention_weights[query_mask].sum(dim=-1),
            torch.ones(int(query_mask.sum())),
        )

    def test_gradients_flow_through_queries_tokens_and_soft_mask(self):
        queries = self.queries.clone().requires_grad_()
        local_tokens = self.local_tokens.clone().requires_grad_()
        spatial_mask = torch.full((2, 3, 5), 0.5, requires_grad=True)
        output = VLCABS(sim_op="cos")(
            queries,
            local_tokens,
            spatial_mask=spatial_mask,
            temperature=torch.tensor(0.07),
        )

        loss = output.matching_logits.sum() + output.evidence_tokens.square().mean()
        loss.backward()

        for gradient in (queries.grad, local_tokens.grad, spatial_mask.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_bfloat16_masked_forward_is_finite(self):
        queries = self.queries.to(torch.bfloat16)
        local_tokens = self.local_tokens.to(torch.bfloat16)
        spatial_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        spatial_mask[..., -1] = False
        output = VLCABS(sim_op="cos")(
            queries,
            local_tokens,
            spatial_mask=spatial_mask,
            temperature=torch.tensor(0.07, dtype=torch.bfloat16),
        )

        for value in (
            output.similarity_scores,
            output.masked_similarity_scores,
            output.attention_weights,
            output.evidence_tokens,
            output.matching_logits,
        ):
            self.assertTrue(torch.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
