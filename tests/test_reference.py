"""Synthetic checks for the three published model families and their shared tensor contract."""

import unittest

import torch

from agu_hydra.models import CANDIDATES, MaskedFeatures, ModelConfig, ModelInputs, normalized_residual_loss
from agu_hydra.training.ablations import AblationSpec, build_ablation_model, project_inputs


def example_group(shape):
    values = torch.ones(shape)
    usable = torch.ones_like(values, dtype=torch.bool)
    return MaskedFeatures(values, usable, usable, torch.zeros_like(usable), torch.zeros_like(values))


def example_inputs():
    leads = torch.arange(1, 19, dtype=torch.float32).view(1, 18, 1).expand(1, 18, 5)
    return ModelInputs(example_group((1, 168, 1)), example_group((1, 18, 5)), example_group((1, 1)), leads, "03161000")


class ReferenceModelTest(unittest.TestCase):
    def test_full_input_shapes(self):
        for family in CANDIDATES:
            with self.subTest(family=family), torch.no_grad():
                model = build_ablation_model(ModelConfig("03161000", candidate=family)).eval()
                prediction = model(example_inputs())
                self.assertEqual(tuple(prediction.shape), (1, 18))
                self.assertTrue(torch.isfinite(prediction).all().item())

    def test_soil_removal_preserves_gauge_inputs_and_capacity(self):
        config = ModelConfig("03161000")
        original = example_inputs()
        removed = project_inputs(original, config, AblationSpec("no_soil"))
        self.assertTrue(torch.equal(original.history.values, removed.history.values))
        self.assertTrue(original.initial_state.usable.all())
        self.assertFalse(removed.initial_state.usable.any())
        self.assertTrue(torch.all(removed.initial_state.values == 0))
        full = build_ablation_model(config, "full")
        reduced = build_ablation_model(config, "no_soil")
        self.assertEqual([p.shape for p in full.parameters()], [p.shape for p in reduced.parameters()])

    def test_loss_uses_observed_ready_cells(self):
        pred = torch.tensor([[5., 20., 90.]])
        truth = torch.tensor([[3., 999., 5.]])
        loss = normalized_residual_loss(pred, truth, input_ready=torch.tensor([[True, True, False]]),
                                        label_loss_mask=torch.tensor([[True, False, True]]), site_scale=torch.tensor([2.]))
        self.assertEqual(loss.item(), 1.)
