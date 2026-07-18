from __future__ import annotations

import unittest

import numpy as np
import torch

from lvs.backends.torch_mlp import run_torch_latent_q_from_arrays
from lvs.core import pipeline
from lvs.core.pipeline import CSVColumnConfig, LatentQConfig, OutputConfig
from lvs.workflows.batch import build_parser as build_batch_parser
from lvs.workflows.batch import namespace_to_batch_config, workflow_config_from_json
from lvs.workflows.single import workflow_config_to_json


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
            ]
        )
        config = namespace_to_batch_config(parsed).workflow_template
        restored = workflow_config_from_json(workflow_config_to_json(config))
        self.assertEqual(restored, config)

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
