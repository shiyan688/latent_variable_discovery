from __future__ import annotations

import unittest

import numpy as np
import torch

from lvs.backends.support_conditioned import (
    AttentiveConditionalRegressor,
    DeepSetEncoder,
    SupportModelConfig,
    predict_attentive_cnp,
    predict_attentive_reliability_selector,
    predict_deepsets_regressor,
    predict_q_encoder,
    predict_q_encoder_multistart,
    train_attentive_cnp,
    train_attentive_reliability_selector,
    train_deepsets_regressor,
    train_q_support_encoder,
)
from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.pipeline import (
    LatentQConfig,
    build_dataset_from_arrays,
    split_support_query_indices,
    train_latent_q_model,
)


class SupportConditionedTests(unittest.TestCase):
    @staticmethod
    def _data(labels: np.ndarray, rows_per_label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        repeated = np.repeat(labels, rows_per_label)
        position = np.tile(np.linspace(-1.0, 1.0, rows_per_label), len(labels))
        x = position.reshape(-1, 1).astype(np.float32)
        state = repeated.astype(np.float32) / 10.0
        y = (0.7 * position + state).astype(np.float32)
        return x, y, repeated

    def test_deepset_encoder_is_permutation_invariant(self) -> None:
        torch.manual_seed(4)
        encoder = DeepSetEncoder(3, 5, (7, 7))
        x = torch.randn(8, 2)
        y = torch.randn(8)
        permutation = torch.tensor([7, 3, 0, 5, 2, 6, 1, 4])
        torch.testing.assert_close(encoder(x, y), encoder(x[permutation], y[permutation]))

    def test_attentive_cnp_is_support_permutation_invariant(self) -> None:
        torch.manual_seed(8)
        model = AttentiveConditionalRegressor(2, 4, (7, 9), (11, 7))
        support_x = torch.randn(8, 2)
        support_y = torch.randn(8)
        query_x = torch.randn(5, 2)
        permutation = torch.tensor([7, 3, 0, 5, 2, 6, 1, 4])
        prediction, representation, _ = model(support_x, support_y, query_x)
        permuted_prediction, permuted_representation, _ = model(
            support_x[permutation], support_y[permutation], query_x
        )
        torch.testing.assert_close(prediction, permuted_prediction)
        torch.testing.assert_close(representation, permuted_representation)

    def test_direct_model_never_uses_query_targets_as_context(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 8)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 8)
        config = SupportModelConfig(
            representation_dim=3,
            epochs=4,
            support_ratio=0.5,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            seed=5,
            device="cpu",
        )
        artifacts = train_deepsets_regressor(train_x, train_y, train_labels, config)
        changed_y = test_y.copy()
        for label in np.unique(test_labels):
            indices = np.flatnonzero(test_labels == label)
            _, query = split_support_query_indices(
                indices, 0.5, mode="random", seed=7, label=label
            )
            changed_y[query] += 1000.0
        first = predict_deepsets_regressor(
            artifacts, test_x, test_y, test_labels, support_ratio=0.5, seed=7
        )
        second = predict_deepsets_regressor(
            artifacts, test_x, changed_y, test_labels, support_ratio=0.5, seed=7
        )
        np.testing.assert_allclose(first.predictions, second.predictions, atol=0.0, rtol=0.0)
        self.assertFalse(np.array_equal(first.targets, second.targets))

    def test_attentive_model_never_uses_query_targets_as_context(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 8)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 8)
        config = SupportModelConfig(
            representation_dim=3,
            epochs=4,
            support_ratio=0.5,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            seed=6,
            device="cpu",
        )
        artifacts = train_attentive_cnp(train_x, train_y, train_labels, config)
        changed_y = test_y.copy()
        for label in np.unique(test_labels):
            indices = np.flatnonzero(test_labels == label)
            _, query = split_support_query_indices(
                indices, 0.5, mode="random", seed=7, label=label
            )
            changed_y[query] -= 1000.0
        first = predict_attentive_cnp(
            artifacts, test_x, test_y, test_labels, support_ratio=0.5, seed=7
        )
        second = predict_attentive_cnp(
            artifacts, test_x, changed_y, test_labels, support_ratio=0.5, seed=7
        )
        np.testing.assert_allclose(first.predictions, second.predictions, atol=0.0, rtol=0.0)
        self.assertFalse(np.array_equal(first.targets, second.targets))

    def test_support_relative_attentive_model_is_leakage_free_and_bounded(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 10)
        train_y[3] = -1000.0
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 10)
        config = SupportModelConfig(
            representation_dim=3,
            epochs=4,
            support_ratio=0.4,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            target_coordinate_mode="support_robust",
            target_loss="smooth_l1",
            support_scale_floor_fraction=0.05,
            support_target_clip=8.0,
            standardized_output_bound=8.0,
            seed=10,
            device="cpu",
        )
        artifacts = train_attentive_cnp(train_x, train_y, train_labels, config)
        changed_y = test_y.copy()
        for label in np.unique(test_labels):
            indices = np.flatnonzero(test_labels == label)
            _, query = split_support_query_indices(
                indices, 0.4, mode="random", seed=12, label=label
            )
            changed_y[query] += 1_000_000.0
        first = predict_attentive_cnp(
            artifacts, test_x, test_y, test_labels, support_ratio=0.4, seed=12
        )
        second = predict_attentive_cnp(
            artifacts, test_x, changed_y, test_labels, support_ratio=0.4, seed=12
        )
        np.testing.assert_allclose(first.predictions, second.predictions, atol=0.0, rtol=0.0)
        self.assertFalse(np.array_equal(first.targets, second.targets))
        self.assertTrue(np.isfinite(first.predictions).all())
        for label, diagnostics in first.diagnostics_by_label.items():
            selected = first.predictions[first.labels == label]
            self.assertTrue(
                (selected >= diagnostics["prediction_physical_lower_bound"] - 1e-6).all()
            )
            self.assertTrue(
                (selected <= diagnostics["prediction_physical_upper_bound"] + 1e-6).all()
            )
            self.assertLessEqual(
                max(
                    abs(diagnostics["prediction_standardized_min"]),
                    abs(diagnostics["prediction_standardized_max"]),
                ),
                8.0 + 1e-6,
            )

    def test_support_relative_attentive_model_is_target_affine_equivariant(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 10)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 10)
        config = SupportModelConfig(
            representation_dim=3,
            epochs=4,
            support_ratio=0.4,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            target_coordinate_mode="support_robust",
            target_loss="smooth_l1",
            standardized_output_bound=8.0,
            seed=13,
            device="cpu",
        )
        first_artifacts = train_attentive_cnp(train_x, train_y, train_labels, config)
        first = predict_attentive_cnp(
            first_artifacts, test_x, test_y, test_labels, support_ratio=0.4, seed=14
        )
        second_artifacts = train_attentive_cnp(
            train_x, 3.0 * train_y + 5.0, train_labels, config
        )
        second = predict_attentive_cnp(
            second_artifacts,
            test_x,
            3.0 * test_y + 5.0,
            test_labels,
            support_ratio=0.4,
            seed=14,
        )
        np.testing.assert_allclose(
            second.predictions,
            3.0 * first.predictions + 5.0,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_attentive_reliability_selector_uses_only_external_support_targets(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 12)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 12)
        config = SupportModelConfig(
            representation_dim=3,
            epochs=4,
            support_ratio=0.5,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            seed=15,
            device="cpu",
        )
        artifacts = train_attentive_reliability_selector(
            train_x, train_y, train_labels, config
        )
        changed_y = test_y.copy()
        for label in np.unique(test_labels):
            indices = np.flatnonzero(test_labels == label)
            _, query = split_support_query_indices(
                indices, 0.5, mode="random", seed=16, label=label
            )
            changed_y[query] -= 1_000_000.0
        first = predict_attentive_reliability_selector(
            artifacts,
            test_x,
            test_y,
            test_labels,
            support_ratio=0.5,
            seed=16,
            selector_ratio=0.25,
            selector_min_rows=2,
        )
        second = predict_attentive_reliability_selector(
            artifacts,
            test_x,
            changed_y,
            test_labels,
            support_ratio=0.5,
            seed=16,
            selector_ratio=0.25,
            selector_min_rows=2,
        )
        np.testing.assert_allclose(first.predictions, second.predictions, atol=0.0, rtol=0.0)
        self.assertFalse(np.array_equal(first.targets, second.targets))
        for label in first.diagnostics_by_label:
            first_row = first.diagnostics_by_label[label]
            second_row = second.diagnostics_by_label[label]
            self.assertEqual(first_row["selected_bounded"], second_row["selected_bounded"])
            self.assertTrue(np.isfinite(first_row["global_selector_mae"]))
            self.assertTrue(np.isfinite(first_row["bounded_selector_mae"]))

    def test_q_encoder_refinement_is_finite_and_bounded(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 8)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 8)
        train_dataset = build_dataset_from_arrays(train_x, train_labels, train_y)
        latent_config = LatentQConfig(
            q_dim=2,
            epochs=4,
            batch_size=16,
            early_stop_enabled=False,
            device="cpu",
            verbose=False,
            seed=3,
        )
        latent = train_latent_q_model(
            train_dataset, build_torch_model_factory((12, 8)), latent_config
        )
        support_config = SupportModelConfig(
            representation_dim=2,
            epochs=4,
            support_ratio=0.5,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            seed=3,
            device="cpu",
        )
        encoder = train_q_support_encoder(
            train_x,
            train_y,
            train_labels,
            latent,
            support_config,
            alignment_weight=0.05,
        )
        result = predict_q_encoder(
            encoder,
            latent,
            test_x,
            test_y,
            test_labels,
            support_ratio=0.5,
            seed=9,
            refine_steps=4,
            refine_lr=0.05,
            trust_region_weight=0.01,
            clip_standard_deviations=2.0,
        )
        self.assertTrue(np.isfinite(result.initial.predictions).all())
        self.assertTrue(np.isfinite(result.refined.predictions).all())
        self.assertTrue(np.isfinite(result.refined.representations).all())
        train_q = latent.embedding.weight.detach().numpy()
        mean = train_q.mean(axis=0)
        std = np.maximum(train_q.std(axis=0), 0.05)
        self.assertTrue((result.refined.representations <= mean + 2.0 * std + 1e-6).all())
        self.assertTrue((result.refined.representations >= mean - 2.0 * std - 1e-6).all())
        self.assertEqual(len(result.refined.query_indices), len(result.refined.predictions))

    def test_encoder_multistart_uses_no_query_targets(self) -> None:
        train_x, train_y, train_labels = self._data(np.array([0, 1, 2, 3]), 8)
        test_x, test_y, test_labels = self._data(np.array([10, 11]), 8)
        train_dataset = build_dataset_from_arrays(train_x, train_labels, train_y)
        latent_config = LatentQConfig(
            q_dim=2,
            epochs=4,
            batch_size=16,
            calibration_steps=4,
            calibration_lr=0.05,
            calibration_ratio=0.5,
            calibration_split_mode="random",
            calibration_init_mode="prior_random",
            calibration_num_starts=2,
            calibration_selection_ratio=0.5,
            calibration_selection_min_rows=2,
            calibration_refine_steps=2,
            calibration_refine_only_after_selection=True,
            early_stop_enabled=False,
            device="cpu",
            verbose=False,
            seed=4,
        )
        latent = train_latent_q_model(
            train_dataset, build_torch_model_factory((12, 8)), latent_config
        )
        support_config = SupportModelConfig(
            representation_dim=2,
            epochs=4,
            support_ratio=0.5,
            entity_batch_size=2,
            encoder_hidden_sizes=(8, 8),
            decoder_hidden_sizes=(12, 8),
            seed=4,
            device="cpu",
        )
        encoder = train_q_support_encoder(
            train_x, train_y, train_labels, latent, support_config
        )
        changed_y = test_y.copy()
        for label in np.unique(test_labels):
            indices = np.flatnonzero(test_labels == label)
            _, query = split_support_query_indices(
                indices, 0.5, mode="random", seed=4, label=label
            )
            changed_y[query] += 1000.0
        first = predict_q_encoder_multistart(
            encoder, latent, test_x, test_y, test_labels, config=latent_config
        )
        second = predict_q_encoder_multistart(
            encoder, latent, test_x, changed_y, test_labels, config=latent_config
        )
        np.testing.assert_allclose(
            first.prediction.predictions,
            second.prediction.predictions,
            atol=0.0,
            rtol=0.0,
        )
        self.assertFalse(
            np.array_equal(first.prediction.targets, second.prediction.targets)
        )
        for diagnostics in first.prediction.diagnostics_by_label.values():
            self.assertIn(diagnostics["selected_extra_candidate"], (0.0, 1.0))
            self.assertEqual(diagnostics["extra_candidate_available"], 1.0)


if __name__ == "__main__":
    unittest.main()
