import unittest

import torch

from exp.cxr_pt.model.hierarchy import (
    AnatomyGroundingHead,
    AnatomyMaskedFindingHead,
    FindingEvidenceAggregator,
    GatedEvidenceFusion,
    GlobalEvidenceBranch,
)


class HierarchyModulesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.batch_size = 2
        self.anatomy_count = 3
        self.finding_count = 4
        self.global_count = 5
        self.token_count = 7
        self.hidden_dim = 8
        self.visual_tokens = torch.randn(
            self.batch_size, self.token_count, self.hidden_dim
        )
        self.anatomy_queries = torch.randn(self.anatomy_count, self.hidden_dim)
        self.finding_queries = torch.randn(self.finding_count, self.hidden_dim)
        self.global_queries = torch.randn(self.global_count, self.hidden_dim)

    def test_anatomy_head_outputs_masks_and_evidence(self):
        head = AnatomyGroundingHead(self.hidden_dim)
        anatomy_query_mask = torch.tensor([True, False, True])
        visual_token_mask = torch.tensor(
            [[True] * 6 + [False], [True] * 5 + [False, False]]
        )
        output = head(
            self.anatomy_queries,
            self.visual_tokens,
            anatomy_query_mask=anatomy_query_mask,
            visual_token_mask=visual_token_mask,
        )

        self.assertEqual(
            output.roi_mask_logits.shape,
            (self.batch_size, self.anatomy_count, self.token_count),
        )
        self.assertEqual(output.roi_masks.shape, output.roi_mask_logits.shape)
        self.assertEqual(
            output.evidence_tokens.shape,
            (self.batch_size, self.anatomy_count, self.hidden_dim),
        )
        self.assertEqual(
            output.matching_logits.shape,
            (self.batch_size, self.anatomy_count),
        )
        self.assertTrue(torch.all(output.roi_masks >= 0))
        self.assertTrue(torch.all(output.roi_masks <= 1))
        self.assertTrue(torch.all(output.roi_masks[:, 1] == 0))
        self.assertTrue(torch.all(output.attention_weights[:, 1] == 0))
        expanded_token_mask = visual_token_mask.unsqueeze(1).expand_as(output.roi_masks)
        self.assertTrue(torch.all(output.roi_masks[~expanded_token_mask] == 0))
        valid = expanded_token_mask & anatomy_query_mask.view(1, -1, 1)
        torch.testing.assert_close(
            output.roi_mask_logits[valid], output.similarity_scores[valid]
        )

    def test_anatomy_roi_refiner_consumes_shared_similarity_map(self):
        head = AnatomyGroundingHead(self.hidden_dim, num_anatomies=self.anatomy_count)
        visual_tokens = self.visual_tokens[:, :4]
        output = head(self.anatomy_queries, visual_tokens)
        scale = head.log_roi_scale.exp().view(1, -1, 1)
        bias = head.roi_bias.view(1, -1, 1)
        expected = head._roi_logits_from_similarity(
            output.similarity_scores, scale, bias
        )

        torch.testing.assert_close(output.roi_mask_logits, expected)

    def test_finding_head_chunking_is_numerically_equivalent(self):
        head = AnatomyMaskedFindingHead(
            self.hidden_dim,
            self.finding_count,
            finding_chunk_size=None,
        )
        anatomy_masks = torch.rand(
            self.batch_size, self.anatomy_count, self.token_count
        )
        anatomy_query_mask = torch.tensor([[True, True, False], [True, False, True]])
        compatibility_mask = torch.ones(
            self.anatomy_count, self.finding_count, dtype=torch.bool
        )
        compatibility_mask[0, 1] = False

        full = head(
            self.finding_queries,
            self.visual_tokens,
            anatomy_masks,
            anatomy_query_mask=anatomy_query_mask,
            compatibility_mask=compatibility_mask,
            finding_chunk_size=self.finding_count,
        )
        chunked = head(
            self.finding_queries,
            self.visual_tokens,
            anatomy_masks,
            anatomy_query_mask=anatomy_query_mask,
            compatibility_mask=compatibility_mask,
            finding_chunk_size=2,
        )

        self.assertEqual(
            full.evidence_tokens.shape,
            (
                self.batch_size,
                self.anatomy_count,
                self.finding_count,
                self.hidden_dim,
            ),
        )
        self.assertEqual(
            full.attention_weights.shape,
            (
                self.batch_size,
                self.anatomy_count,
                self.finding_count,
                self.token_count,
            ),
        )
        for full_value, chunked_value in (
            (full.similarity_scores, chunked.similarity_scores),
            (full.masked_similarity_scores, chunked.masked_similarity_scores),
            (full.attention_weights, chunked.attention_weights),
            (full.evidence_tokens, chunked.evidence_tokens),
            (full.matching_scores, chunked.matching_scores),
            (full.finding_logits, chunked.finding_logits),
        ):
            torch.testing.assert_close(full_value, chunked_value)
        self.assertTrue(torch.equal(full.pair_mask, chunked.pair_mask))
        self.assertTrue(torch.all(full.attention_weights[~full.pair_mask] == 0))
        self.assertTrue(torch.all(full.evidence_tokens[~full.pair_mask] == 0))
        self.assertTrue(torch.all(full.finding_logits[~full.pair_mask] == 0))
        torch.testing.assert_close(
            full.attention_weights[full.pair_mask].sum(dim=-1),
            torch.ones(int(full.pair_mask.sum())),
        )

    def test_aggregator_uses_null_evidence_when_no_pairs_are_valid(self):
        aggregator = FindingEvidenceAggregator(self.hidden_dim, num_heads=2)
        evidence = torch.randn(
            self.batch_size,
            self.anatomy_count,
            self.finding_count,
            self.hidden_dim,
        )
        scores = torch.randn(self.batch_size, self.anatomy_count, self.finding_count)
        pair_mask = torch.ones_like(scores, dtype=torch.bool)
        pair_mask[0] = False
        pair_mask[1, 0, 0] = False
        output = aggregator(
            self.global_queries,
            evidence,
            finding_scores=scores,
            pair_mask=pair_mask,
        )

        self.assertEqual(
            output.evidence_tokens.shape,
            (self.batch_size, self.global_count, self.hidden_dim),
        )
        self.assertEqual(
            output.attention_weights.shape,
            (
                self.batch_size,
                self.global_count,
                self.anatomy_count,
                self.finding_count,
            ),
        )
        self.assertTrue(torch.isfinite(output.evidence_tokens).all())
        self.assertTrue(torch.all(output.attention_weights[0] == 0))
        torch.testing.assert_close(
            output.null_attention[0], torch.ones(self.global_count)
        )
        self.assertTrue(torch.all(output.attention_weights[1, :, 0, 0] == 0))

    def test_global_branch_and_fusion_respect_query_mask(self):
        global_branch = GlobalEvidenceBranch(self.hidden_dim)
        fusion = GatedEvidenceFusion(self.hidden_dim, self.global_count)
        global_query_mask = torch.tensor([True, False, True, True, False])
        global_output = global_branch(
            self.global_queries,
            self.visual_tokens,
            global_query_mask=global_query_mask,
        )
        finding_evidence = torch.randn_like(global_output.evidence_tokens)
        fused = fusion(
            finding_evidence,
            global_output.evidence_tokens,
            self.global_queries,
            global_query_mask=global_query_mask,
        )

        self.assertEqual(fused.logits.shape, (self.batch_size, self.global_count))
        self.assertEqual(
            fused.fused_evidence.shape,
            (self.batch_size, self.global_count, self.hidden_dim),
        )
        valid_mask = global_query_mask.unsqueeze(0).expand(self.batch_size, -1)
        self.assertTrue(torch.all(fused.logits[~valid_mask] == 0))
        self.assertTrue(torch.all(fused.fused_evidence[~valid_mask] == 0))
        self.assertTrue(torch.all(fused.gates[~valid_mask] == 0))
        torch.testing.assert_close(
            fused.gates[valid_mask],
            torch.full((int(valid_mask.sum()), 1), 0.5),
        )

    def test_extreme_log_temperatures_are_bounded_and_finite(self):
        anatomy = AnatomyGroundingHead(self.hidden_dim)
        finding = AnatomyMaskedFindingHead(self.hidden_dim, self.finding_count)
        fusion = GatedEvidenceFusion(self.hidden_dim, self.global_count)
        anatomy_masks = torch.ones(
            self.batch_size, self.anatomy_count, self.token_count
        )
        finding_evidence = torch.randn(
            self.batch_size, self.global_count, self.hidden_dim
        )
        global_evidence = torch.randn_like(finding_evidence)

        for extreme in (-100.0, 100.0):
            anatomy.log_attention_temperature.fill_(extreme)
            finding.log_attention_temperature.data.fill_(extreme)
            finding.log_logit_temperature.data.fill_(extreme)
            fusion.log_logit_temperature.data.fill_(extreme)
            anatomy_output = anatomy(self.anatomy_queries, self.visual_tokens)
            finding_output = finding(
                self.finding_queries, self.visual_tokens, anatomy_masks
            )
            fusion_output = fusion(
                finding_evidence,
                global_evidence,
                self.global_queries,
            )
            for value in (
                anatomy_output.attention_weights,
                anatomy_output.roi_mask_logits,
                finding_output.attention_weights,
                finding_output.finding_logits,
                fusion_output.logits,
            ):
                self.assertTrue(torch.isfinite(value).all())

    def test_global_gradient_reaches_anatomy_head(self):
        visual_tokens = self.visual_tokens.clone().requires_grad_()
        anatomy_queries = self.anatomy_queries.clone().requires_grad_()
        finding_queries = self.finding_queries.clone().requires_grad_()
        global_queries = self.global_queries.clone().requires_grad_()
        anatomy_head = AnatomyGroundingHead(self.hidden_dim)
        finding_head = AnatomyMaskedFindingHead(
            self.hidden_dim,
            self.finding_count,
            finding_chunk_size=2,
        )
        aggregator = FindingEvidenceAggregator(self.hidden_dim, num_heads=2)
        global_branch = GlobalEvidenceBranch(self.hidden_dim)
        fusion = GatedEvidenceFusion(self.hidden_dim, self.global_count)

        anatomy = anatomy_head(anatomy_queries, visual_tokens)
        finding = finding_head(
            finding_queries,
            visual_tokens,
            anatomy.roi_masks,
        )
        finding_evidence = aggregator(
            global_queries,
            finding.evidence_tokens,
            finding_scores=finding.finding_logits,
            pair_mask=finding.pair_mask,
        )
        global_evidence = global_branch(global_queries, visual_tokens)
        fused = fusion(
            finding_evidence.evidence_tokens,
            global_evidence.evidence_tokens,
            global_queries,
        )
        loss = fused.logits.sum() + finding.finding_logits.mean()
        loss.backward()

        for gradient in (
            visual_tokens.grad,
            anatomy_queries.grad,
            finding_queries.grad,
            global_queries.grad,
            anatomy_head.log_roi_scale.grad,
            anatomy_head.roi_bias.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
