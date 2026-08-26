from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from lvs.backends.torch_mlp import run_torch_latent_q_from_arrays
from lvs.core import pipeline
from lvs.core.pipeline import CSVColumnConfig, LatentQConfig, OutputConfig
from lvs.workflows.batch import build_parser as build_batch_parser
from lvs.workflows.batch import namespace_to_batch_config, workflow_config_from_json
from lvs.workflows.single import generate_expression_dataset, workflow_config_to_json


class _TrackingLinear(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.trainable_flags: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.trainable_flags.append(all(parameter.requires_grad for parameter in self.parameters()))
        return self.linear(inputs)


class PipelineUnitTests(unittest.TestCase):
    def test_batch_config_round_trip_keeps_all_loss_options(self) -> None:
        parsed = build_batch_parser().parse_args(
            [
                "--latent-feature-stats-mode",
                "rich_rff_kme",
                "--latent-q-whitening-weight",
                "0.1",
                "--prediction-loss-type",
                "label_balanced_mse",
                "--label-split-mode",
                "disjoint",
                "--validation-label-count",
                "7",
                "--test-label-count",
                "9",
                "--validation-samples-per-label",
                "11",
                "--calibration-split-mode",
                "random",
                "--optimization-schedule",
                "alternating",
                "--theta-lr",
                "0.002",
                "--q-lr",
                "0.003",
                "--theta-steps-per-cycle",
                "2",
                "--q-steps-per-cycle",
                "3",
                "--loss-weighting",
                "gradnorm",
                "--gradnorm-record-trace",
            ]
        )
        config = namespace_to_batch_config(parsed).workflow_template
        restored = workflow_config_from_json(workflow_config_to_json(config))
        self.assertEqual(restored, config)

    def test_generator_receives_configured_disjoint_protocol(self) -> None:
        parsed = build_batch_parser().parse_args(
            [
                "--label-count", "5",
                "--label-split-mode", "disjoint",
                "--validation-label-count", "2",
                "--test-label-count", "3",
                "--validation-samples-per-label", "4",
                "--calibration-split-mode", "random",
            ]
        )
        config = namespace_to_batch_config(parsed).workflow_template
        with patch("lvs.workflows.single.sample_expression_dataset", return_value="generated") as generator:
            self.assertEqual(generate_expression_dataset(object(), config), "generated")
        self.assertEqual(
            generator.call_args.kwargs,
            {
                "label_count": 5,
                "validation_label_count": 2,
                "test_label_count": 3,
                "label_split_mode": "disjoint",
                "train_samples_per_label": 80,
                "validation_samples_per_label": 4,
                "test_samples_per_label": 30,
                "noise_std": 0.0,
                "seed": 42,
                "max_attempts_per_row": 200,
            },
        )
        self.assertEqual(config.calibration_split_mode, "random")

    def test_dynamic_weights_are_prediction_anchored_bounded_and_finite(self) -> None:
        config = LatentQConfig(
            loss_weighting="gradnorm",
            gradnorm_alpha=0.5,
            gradnorm_lr=1.0,
            gradnorm_min_weight=0.1,
            gradnorm_max_weight=10.0,
            latent_q_l2_weight=1.0,
        )
        weights = pipeline._static_loss_weights(config)
        pipeline._update_dynamic_loss_weights(
            weights,
            {"prediction": torch.tensor(4.0), "latent_q_l2": torch.tensor(0.01)},
            config,
        )
        self.assertEqual(weights["prediction"], 1.0)
        self.assertGreaterEqual(weights["latent_q_l2"], 0.1)
        self.assertLessEqual(weights["latent_q_l2"], 10.0)
        self.assertTrue(np.isfinite(weights["latent_q_l2"]))

    def test_alternating_counter_budget_and_freeze_semantics(self) -> None:
        rng = np.random.default_rng(2)
        labels = np.repeat(np.arange(3), 4)
        x = rng.normal(size=(12, 1)).astype(np.float32)
        y = (x[:, 0] + labels * 0.1).astype(np.float32)
        dataset = pipeline.build_dataset_from_arrays(x, labels, y)
        artifacts = pipeline.train_latent_q_model(
            dataset,
            lambda input_dim: _TrackingLinear(input_dim),
            LatentQConfig(
                q_dim=1,
                epochs=1,
                batch_size=6,
                optimization_schedule="alternating",
                theta_steps_per_cycle=2,
                q_steps_per_cycle=3,
                early_stop_enabled=False,
                device="cpu",
                verbose=False,
            ),
        )
        counters = artifacts.optimization_counters
        self.assertEqual(counters.theta_steps, 4)
        self.assertEqual(counters.q_steps, 6)
        self.assertEqual(counters.backward_passes, 10)
        self.assertEqual(counters.examples_processed, 60)
        self.assertEqual(artifacts.model.trainable_flags[:5], [True, True, False, False, False])
        self.assertTrue(artifacts.embedding.weight.requires_grad)
        self.assertIn("prediction", artifacts.train_history[-1].loss_components)

    def test_joint_default_has_original_one_backward_per_batch_budget(self) -> None:
        labels = np.repeat(np.arange(2), 4)
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = x[:, 0].copy()
        artifacts = pipeline.train_latent_q_model(
            pipeline.build_dataset_from_arrays(x, labels, y),
            lambda input_dim: torch.nn.Linear(input_dim, 1),
            LatentQConfig(q_dim=1, epochs=2, batch_size=4, early_stop_enabled=False, device="cpu", verbose=False),
        )
        counters = artifacts.optimization_counters
        self.assertEqual((counters.theta_steps, counters.q_steps, counters.backward_passes), (4, 4, 4))

    def test_prefix_q_training_uses_only_entity_prefix_for_q_phase(self) -> None:
        labels = np.repeat(np.arange(2), 10)
        x = np.tile(np.arange(10, dtype=np.float32), 2).reshape(-1, 1)
        y = (x[:, 0] + labels * 0.1).astype(np.float32)
        artifacts = pipeline.train_latent_q_model(
            pipeline.build_dataset_from_arrays(x, labels, y),
            lambda input_dim: _TrackingLinear(input_dim),
            LatentQConfig(
                q_dim=1,
                epochs=1,
                batch_size=20,
                optimization_schedule="alternating",
                q_training_split_mode="prefix",
                q_training_ratio=0.3,
                q_training_order_feature_index=0,
                latent_curve_continuity_weight=0.05,
                early_stop_enabled=False,
                device="cpu",
                verbose=False,
            ),
        )
        counters = artifacts.optimization_counters
        self.assertEqual((counters.theta_steps, counters.q_steps), (1, 1))
        self.assertEqual(counters.backward_passes, 2)
        self.assertEqual(counters.examples_processed, 26)
        self.assertEqual(artifacts.model.trainable_flags[:2], [False, True])

    def test_alternating_adaptive_weights_update_on_q_phase(self) -> None:
        labels = np.repeat(np.arange(3), 4)
        x = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
        y = (x[:, 0] + labels * 0.2).astype(np.float32)
        artifacts = pipeline.train_latent_q_model(
            pipeline.build_dataset_from_arrays(x, labels, y),
            lambda input_dim: torch.nn.Linear(input_dim, 1),
            LatentQConfig(
                q_dim=1, epochs=2, batch_size=6, optimization_schedule="alternating",
                loss_weighting="adaptive_loss_scale", latent_q_l2_weight=0.1,
                gradnorm_warmup_steps=0, gradnorm_interval=1, gradnorm_record_trace=True,
                early_stop_enabled=False, device="cpu", verbose=False,
            ),
        )
        self.assertTrue(artifacts.dynamic_weight_trace)
        self.assertTrue(all(entry["phase"] == "q" for entry in artifacts.dynamic_weight_trace))
        self.assertNotEqual(artifacts.dynamic_weight_trace[-1]["latent_q_l2"], 0.1)

    def test_joint_can_match_alternating_backward_budget(self) -> None:
        labels = np.repeat(np.arange(2), 4)
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = x[:, 0].copy()
        dataset = pipeline.build_dataset_from_arrays(x, labels, y)
        common = dict(q_dim=1, epochs=1, batch_size=4, early_stop_enabled=False, device="cpu", verbose=False)
        joint = pipeline.train_latent_q_model(
            dataset,
            lambda input_dim: torch.nn.Linear(input_dim, 1),
            LatentQConfig(**common, optimization_schedule="joint", joint_steps_per_cycle=2),
        )
        alternating = pipeline.train_latent_q_model(
            dataset,
            lambda input_dim: torch.nn.Linear(input_dim, 1),
            LatentQConfig(**common, optimization_schedule="alternating", theta_steps_per_cycle=1, q_steps_per_cycle=1),
        )
        self.assertEqual(joint.optimization_counters.backward_passes, alternating.optimization_counters.backward_passes)
        self.assertEqual(joint.optimization_counters.examples_processed, alternating.optimization_counters.examples_processed)

    def test_evaluate_many_reuses_one_training_artifact(self) -> None:
        train = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0], [0.0], [1.0]], [0, 0, 1, 1], [0.0, 1.0, 0.5, 1.5]
        )
        first = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0]], [2, 2], [1.0, 2.0]
        )
        second = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0]], [3, 3], [1.5, 2.5]
        )
        config = LatentQConfig(
            q_dim=1, epochs=1, batch_size=4, calibration_steps=2,
            calibration_ratio=0.5, early_stop_enabled=False, device="cpu", verbose=False,
        )
        artifacts = pipeline.train_latent_q_model(
            train, lambda input_dim: torch.nn.Linear(input_dim, 1), config
        )
        before = [parameter.detach().clone() for parameter in artifacts.model.parameters()]
        first_result = pipeline.evaluate_latent_q_pipeline(train, first, artifacts, config)
        second_result = pipeline.evaluate_latent_q_pipeline(train, second, artifacts, config)
        self.assertIs(first_result.training_artifacts, artifacts)
        self.assertIs(second_result.training_artifacts, artifacts)
        for parameter, snapshot in zip(artifacts.model.parameters(), before):
            self.assertTrue(torch.equal(parameter.detach(), snapshot))

    def test_singleton_label_is_rejected_before_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            pipeline.split_support_query_indices(np.array([4]), 0.5)

    def test_cached_hsic_feature_kernel_preserves_value_and_q_gradient(self) -> None:
        q = torch.tensor([[0.0], [1.0], [2.0]], requires_grad=True)
        stats = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        uncached = pipeline._latent_feature_hsic_penalty(q, stats)
        uncached_gradient = torch.autograd.grad(uncached, q, retain_graph=True)[0]
        kernel = pipeline._rbf_kernel_with_median_bandwidth(pipeline._standardize_columns(stats)).detach()
        cached = pipeline._latent_feature_hsic_penalty(q, stats, feature_kernel=kernel)
        cached_gradient = torch.autograd.grad(cached, q)[0]
        self.assertTrue(torch.allclose(cached, uncached))
        self.assertTrue(torch.allclose(cached_gradient, uncached_gradient))

    def test_curve_geometry_preserves_offset_and_amplitude(self) -> None:
        grid = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        targets = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 2.0])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        distances = pipeline._compute_label_curve_distance_matrix(
            grid, targets, labels, label_count=3, grid_size=8
        )
        self.assertGreater(float(distances[0, 1] - distances[0, 0]), 0.0)
        self.assertGreater(float(distances[0, 2] - distances[0, 0]), 0.0)

    def test_single_label_calibration_prior_is_finite_and_clears_model_grads(self) -> None:
        train = pipeline.build_dataset_from_arrays([[0.0], [1.0]], [1, 1], [0.0, 1.0])
        artifacts = pipeline.train_latent_q_model(
            train, lambda input_dim: torch.nn.Linear(input_dim, 1),
            LatentQConfig(q_dim=1, epochs=1, batch_size=2, early_stop_enabled=False, device="cpu", verbose=False),
        )
        test = pipeline.build_dataset_from_arrays([[0.0], [1.0], [2.0]], [2, 2, 2], [0.0, 1.0, 2.0])
        calibrated = pipeline.calibrate_latent_q_for_test_labels(
            test, artifacts,
            LatentQConfig(q_dim=1, epochs=1, calibration_steps=2, calibration_ratio=0.34,
                          calibration_q_prior_weight=1.0, device="cpu", verbose=False),
        )
        self.assertTrue(np.isfinite(calibrated.q_by_label[2]).all())
        self.assertTrue(all(parameter.grad is None for parameter in artifacts.model.parameters()))

    def test_inner_calibration_split_is_seeded_and_disjoint(self) -> None:
        indices = np.arange(10)
        first = pipeline._calibration_fit_selection_indices(
            indices, 0.3, seed=7, label="held-out"
        )
        second = pipeline._calibration_fit_selection_indices(
            indices, 0.3, seed=7, label="held-out"
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(first[2])
        self.assertFalse(np.intersect1d(first[0], first[1]).size)
        np.testing.assert_array_equal(np.sort(np.concatenate(first[:2])), indices)

    def test_inner_calibration_split_respects_minimum_support_rows(self) -> None:
        indices = np.arange(18)
        fit, selection, used = pipeline._calibration_fit_selection_indices(
            indices, 0.25, min_rows=32, seed=7, label="small-support"
        )
        self.assertFalse(used)
        np.testing.assert_array_equal(fit, indices)
        np.testing.assert_array_equal(selection, indices)

    def test_multistart_calibration_records_selection_diagnostics(self) -> None:
        train = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]],
            [0, 0, 0, 1, 1, 1],
            [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
        )
        base = LatentQConfig(
            q_dim=2, epochs=1, batch_size=6, early_stop_enabled=False,
            device="cpu", verbose=False,
        )
        artifacts = pipeline.train_latent_q_model(
            train, lambda input_dim: torch.nn.Linear(input_dim, 1), base
        )
        test = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0], [2.0], [3.0]], [2, 2, 2, 2], [1.0, 2.0, 3.0, 4.0]
        )
        config = LatentQConfig(
            q_dim=2, epochs=1, calibration_steps=2, calibration_ratio=0.5,
            calibration_init_mode="prior_random", calibration_num_starts=4,
            calibration_selection_ratio=0.5, calibration_refine_steps=1,
            early_stop_enabled=False, device="cpu", verbose=False,
        )
        result = pipeline.evaluate_latent_q_pipeline(train, test, artifacts, config)
        diagnostic = result.metrics
        self.assertEqual(diagnostic["calibration_num_starts"], 4)
        self.assertEqual(diagnostic["calibration_inner_selection_fraction"], 1.0)
        self.assertTrue(np.isfinite(diagnostic["calibration_candidate_q_dispersion_mean"]))
        self.assertTrue(np.isfinite(diagnostic["calibration_selection_loss_mean"]))

    def test_adaptive_calibration_skips_refinement_without_inner_split(self) -> None:
        train = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]],
            [0, 0, 0, 1, 1, 1],
            [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
        )
        base = LatentQConfig(
            q_dim=1, epochs=1, batch_size=6, early_stop_enabled=False,
            device="cpu", verbose=False,
        )
        artifacts = pipeline.train_latent_q_model(
            train, lambda input_dim: torch.nn.Linear(input_dim, 1), base
        )
        test = pipeline.build_dataset_from_arrays(
            [[0.0], [1.0], [2.0], [3.0]], [2, 2, 2, 2], [1.0, 2.0, 3.0, 4.0]
        )
        config = LatentQConfig(
            q_dim=1, epochs=1, calibration_steps=2, calibration_ratio=0.5,
            calibration_init_mode="prior_random", calibration_num_starts=2,
            calibration_selection_ratio=0.5, calibration_selection_min_rows=24,
            calibration_refine_steps=2, calibration_refine_only_after_selection=True,
            early_stop_enabled=False, device="cpu", verbose=False,
        )
        result = pipeline.evaluate_latent_q_pipeline(train, test, artifacts, config)
        self.assertEqual(result.metrics["calibration_inner_selection_fraction"], 0.0)
        self.assertEqual(result.metrics["calibration_refinement_fraction"], 0.0)

    def test_default_csv_columns_exclude_label_from_observed_features(self) -> None:
        config = CSVColumnConfig()
        self.assertEqual(config.label_col, 0)
        self.assertEqual(config.feature_cols, (1,))
        self.assertNotIn(config.label_col, config.feature_cols)

    def test_calibration_and_evaluation_indices_are_disjoint(self) -> None:
        indices = np.arange(5)
        calibration, evaluation = pipeline.split_calibration_and_eval_indices(indices, 0.4)
        np.testing.assert_array_equal(calibration, np.array([0, 1]))
        np.testing.assert_array_equal(evaluation, np.array([2, 3, 4]))
        self.assertFalse(np.intersect1d(calibration, evaluation).size)

    def test_label_balanced_mse_weights_groups_equally(self) -> None:
        predictions = torch.tensor([0.0, 2.0, 10.0])
        targets = torch.zeros(3)
        labels = torch.tensor([0, 0, 1])
        loss = pipeline._prediction_loss(
            predictions,
            targets,
            labels,
            loss_type="label_balanced_mse",
        )
        self.assertAlmostEqual(loss.item(), 51.0, places=6)

    def test_rich_rff_kme_is_deterministic_and_has_expected_shape(self) -> None:
        features = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])
        first = pipeline._compute_label_feature_stats(
            features,
            labels,
            label_count=2,
            mode="rich_rff_kme",
        )
        second = pipeline._compute_label_feature_stats(
            features,
            labels,
            label_count=2,
            mode="rich_rff_kme",
        )
        self.assertEqual(first.shape, (2, 120))
        self.assertTrue(torch.equal(first, second))

        with_empty_label = pipeline._compute_label_feature_stats(
            features,
            labels,
            label_count=3,
            mode="rich_rff_kme",
        )
        self.assertEqual(with_empty_label.shape, (3, 120))


class PipelineSmokeTests(unittest.TestCase):
    def test_torch_pipeline_runs_with_unseen_test_labels_on_cpu(self) -> None:
        rng = np.random.default_rng(7)

        def make_split(labels: np.ndarray, samples_per_label: int):
            repeated_labels = np.repeat(labels, samples_per_label)
            x = rng.uniform(-1.0, 1.0, size=(repeated_labels.size, 1)).astype(np.float32)
            latent = repeated_labels.astype(np.float32).reshape(-1, 1) / 10.0
            y = (1.5 * x[:, 0] + latent[:, 0]).astype(np.float32)
            return x, repeated_labels, y

        train_x, train_labels, train_y = make_split(np.array([1, 2, 3, 4]), 8)
        test_x, test_labels, test_y = make_split(np.array([11, 12]), 8)
        result = run_torch_latent_q_from_arrays(
            train_features=train_x,
            train_labels=train_labels,
            train_targets=train_y,
            test_features=test_x,
            test_labels=test_labels,
            test_targets=test_y,
            feature_names=("x",),
            config=LatentQConfig(
                q_dim=1,
                epochs=3,
                batch_size=16,
                calibration_steps=8,
                calibration_ratio=0.5,
                early_stop_enabled=False,
                device="cpu",
                verbose=False,
            ),
            output_config=OutputConfig(save_csv=False, save_plot=False),
            hidden_sizes=(16,),
        )
        self.assertEqual(result.train_q_matrix.shape, (32, 1))
        self.assertEqual(result.test_q_matrix.shape, (16, 1))
        self.assertEqual(result.eval_indices.size, 8)
        self.assertTrue(np.isfinite(result.metrics["test_r2"]))


if __name__ == "__main__":
    unittest.main()
