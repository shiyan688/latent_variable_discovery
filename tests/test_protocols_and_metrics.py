from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from lvs.core.expression_library import (
    ExpressionRecord,
    build_expression_task,
    sample_expression_dataset,
    save_generated_expression_dataset,
)
from lvs.core.metrics import (
    alignment_metrics,
    apply_affine_alignment,
    apply_cca_alignment,
    canonical_correlations,
    effective_rank,
    fit_affine_alignment,
    fit_cca_alignment,
    grouped_rff_signatures,
    knn_overlap,
    local_distance_distortion,
    macro_prediction_metrics,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
    procrustes,
    reference_scaled_prediction_metrics,
    score_cca_alignment,
    trustworthiness_continuity,
)
from lvs.core.pipeline import split_support_query_indices
from lvs.workflows.single import WorkflowConfig, detect_target_scaling


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def task():
        return build_expression_task(
            ExpressionRecord(
                expression_id=1,
                raw_formula="y=x1+2*q1-q2",
                variable_mapping={},
                variable_ranges={"x1": (-1.0, 1.0), "q1": (-2.0, 2.0), "q2": (-3.0, 3.0)},
                formula_name="linear",
            )
        )

    def test_disjoint_generator_has_split_aware_truth_and_is_reproducible(self) -> None:
        kwargs = dict(
            label_count=3,
            validation_label_count=2,
            test_label_count=2,
            train_samples_per_label=4,
            validation_samples_per_label=3,
            test_samples_per_label=5,
            label_split_mode="disjoint",
            seed=17,
        )
        first = sample_expression_dataset(self.task(), **kwargs)
        second = sample_expression_dataset(self.task(), **kwargs)
        self.assertTrue(first.train_frame.equals(second.train_frame))
        self.assertTrue(first.validation_frame.equals(second.validation_frame))
        self.assertTrue(first.test_frame.equals(second.test_frame))
        self.assertTrue(first.latent_truth_frame.equals(second.latent_truth_frame))

        train_labels = set(first.train_frame.label)
        validation_labels = set(first.validation_frame.label)
        test_labels = set(first.test_frame.label)
        self.assertFalse(train_labels & validation_labels)
        self.assertFalse(train_labels & test_labels)
        self.assertFalse(validation_labels & test_labels)
        self.assertEqual(set(first.latent_truth_frame["split"]), {"train", "validation", "test"})

    def test_legacy_generator_keeps_shared_label_api(self) -> None:
        generated = sample_expression_dataset(
            self.task(), label_count=2, train_samples_per_label=2, test_samples_per_label=3
        )
        self.assertEqual(set(generated.train_frame.label), set(generated.test_frame.label))
        self.assertNotIn("split", generated.latent_truth_frame.columns)
        self.assertIsNone(generated.validation_frame)

    def test_validation_artifacts_round_trip_with_protocol_metadata(self) -> None:
        generated = sample_expression_dataset(
            self.task(),
            label_count=3,
            validation_label_count=2,
            test_label_count=4,
            train_samples_per_label=2,
            validation_samples_per_label=3,
            test_samples_per_label=2,
            label_split_mode="disjoint",
            seed=23,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            paths = save_generated_expression_dataset(
                generated,
                output_dir,
                validation_filename="held_out_validation.csv",
            )
            self.assertIn("generated_validation_csv", paths)
            self.assertEqual(paths["generated_validation_csv"].name, "held_out_validation.csv")
            pd.testing.assert_frame_equal(
                pd.read_csv(paths["generated_validation_csv"]),
                generated.validation_frame,
            )
            metadata = json.loads(paths["expression_metadata_json"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["train_label_count"], 3)
            self.assertEqual(metadata["validation_label_count"], 2)
            self.assertEqual(metadata["test_label_count"], 4)
            self.assertEqual(metadata["generator_protocol_version"], 2)
            self.assertEqual(metadata["label_count"], 9)

    def test_random_support_query_split_is_reproducible_and_seeded(self) -> None:
        indices = np.arange(20)
        first = split_support_query_indices(indices, 0.3, mode="random", seed=11, label="a")
        second = split_support_query_indices(indices, 0.3, mode="random", seed=11, label="a")
        changed = split_support_query_indices(indices, 0.3, mode="random", seed=12, label="a")
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(np.array_equal(first[0], changed[0]))
        self.assertFalse(np.intersect1d(*first).size)
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), indices)

    def test_prefix_split_is_legacy_compatible(self) -> None:
        support, query = split_support_query_indices(np.arange(5), 0.4)
        np.testing.assert_array_equal(support, [0, 1])
        np.testing.assert_array_equal(query, [2, 3, 4])

    def test_target_scaling_ignores_test_targets(self) -> None:
        config = WorkflowConfig(
            library_csv=None, expression_id=1, expression_name=None, label_count=1,
            validation_label_count=0, test_label_count=None, label_split_mode="shared",
            train_samples_per_label=1, validation_samples_per_label=None,
            test_samples_per_label=1, noise_std=0.0, seed=1,
            backend="torch", q_dim=1, output_root=None, max_attempts_per_row=1, epochs=1,
            batch_size=1, lr=0.1, auto_target_scale=True, target_scale_min_magnitude=1e-3,
            target_scale_desired_magnitude=1.0, cal_steps=1, cal_lr=0.1, cal_ratio=0.5,
            calibration_split_mode="prefix",
            early_stop_enabled=False, early_stop_r2_threshold=0.9, early_stop_patience=1,
            latent_feature_orthogonality_weight=0.0, latent_feature_orthogonality_type="pearson",
            latent_feature_stats_mode="mean_std", latent_curve_continuity_weight=0.0,
            latent_curve_continuity_grid_size=2, calibration_q_prior_weight=0.0,
            latent_q_l2_weight=0.0, prediction_loss_type="mse", latent_q_whitening_weight=0.0,
            latent_jacobian_disentanglement_weight=0.0, latent_q_canonicalization_mode="none",
            latent_q_smoothness_weight=0.0, latent_q_smoothness_epsilon=0.1,
            optimization_schedule="joint", theta_lr=None, q_lr=None,
            theta_steps_per_cycle=1, q_steps_per_cycle=1, loss_weighting="static",
            gradnorm_warmup_steps=0, gradnorm_interval=1, gradnorm_alpha=0.5,
            gradnorm_lr=0.025, gradnorm_min_weight=1e-3, gradnorm_max_weight=1e3,
            gradnorm_record_trace=False, device="cpu",
            quiet=True, hidden_sizes="2", kan_grid=2, kan_order=2,
        )
        train = np.array([1e-8, -2e-8])
        first = detect_target_scaling(train, np.array([1e9]), config)
        second = detect_target_scaling(train, np.array([1e-20]), config)
        self.assertEqual(first, second)
        self.assertTrue(first.applied)


class MetricTests(unittest.TestCase):
    def test_affine_alignment_fits_validation_and_applies_test(self) -> None:
        rng = np.random.default_rng(4)
        learned = rng.normal(size=(20, 2))
        coefficients = np.array([[2.0, -1.0], [0.5, 3.0]])
        intercept = np.array([4.0, -2.0])
        truth = learned @ coefficients + intercept
        alignment = fit_affine_alignment(learned[:12], truth[:12])
        aligned_test = apply_affine_alignment(alignment, learned[12:])
        np.testing.assert_allclose(aligned_test, truth[12:], atol=1e-10)
        self.assertAlmostEqual(alignment_metrics(truth[12:], aligned_test)["aligned_r2"], 1.0)

    def test_geometry_metrics_are_exact_under_similarity_transform(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.normal(size=(12, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        target = 2.5 * source @ rotation + 7.0
        aligned, disparity = procrustes(source, target)
        np.testing.assert_allclose(aligned, target, atol=1e-10)
        self.assertLess(disparity, 1e-20)
        geometry = pairwise_distance_metrics(source, target)
        self.assertAlmostEqual(geometry["distance_pearson"], 1.0)
        self.assertAlmostEqual(geometry["distance_spearman"], 1.0)
        self.assertAlmostEqual(knn_overlap(source, target, k=3), 1.0)

    def test_cca_alignment_fits_validation_and_scores_test(self) -> None:
        rng = np.random.default_rng(29)
        validation_left = rng.normal(size=(40, 3))
        validation_right = validation_left @ np.array(
            [[2.0, -1.0], [0.5, 3.0], [-2.0, 0.25]]
        )
        test_left = rng.normal(size=(18, 3))
        test_right = test_left @ np.array(
            [[2.0, -1.0], [0.5, 3.0], [-2.0, 0.25]]
        )

        alignment = fit_cca_alignment(validation_left, validation_right, n_components=2)
        fitted_parameters = tuple(
            value.copy()
            for value in (
                alignment.left_mean,
                alignment.left_scale,
                alignment.left_rotations,
                alignment.right_mean,
                alignment.right_scale,
                alignment.right_rotations,
            )
        )
        projected_left, projected_right = apply_cca_alignment(
            alignment, test_left, test_right
        )
        np.testing.assert_allclose(
            score_cca_alignment(alignment, test_left, test_right),
            np.ones(2),
            atol=1e-7,
        )
        score_cca_alignment(alignment, test_left * 100.0, rng.normal(size=(18, 2)))
        for actual, expected in zip(
            (
                alignment.left_mean,
                alignment.left_scale,
                alignment.left_rotations,
                alignment.right_mean,
                alignment.right_scale,
                alignment.right_rotations,
            ),
            fitted_parameters,
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(projected_left.shape, (18, 2))
        self.assertEqual(projected_right.shape, (18, 2))

    def test_knn_overlap_excludes_self_with_duplicate_points(self) -> None:
        left = np.zeros((3, 1))
        right = np.array([[0.0], [0.0], [1.0]])
        self.assertAlmostEqual(knn_overlap(left, right, k=1), 1.0)

    def test_cca_macro_and_effective_rank(self) -> None:
        rng = np.random.default_rng(3)
        latent = rng.normal(size=(30, 2))
        observed = np.column_stack([latent, latent[:, 0] + latent[:, 1]])
        correlations = canonical_correlations(observed, latent)
        np.testing.assert_allclose(correlations, np.ones(2), atol=1e-7)
        self.assertAlmostEqual(effective_rank(np.column_stack([latent[:, 0], np.zeros(30)])), 1.0)
        metrics = macro_prediction_metrics([0, 2, 10], [0, 0, 0], [1, 1, 2])
        self.assertAlmostEqual(metrics["macro_mse"], 51.0)

    def test_reference_scaled_prediction_metric_avoids_constant_label_pathology(self) -> None:
        metrics = reference_scaled_prediction_metrics(
            [1.0, 1.0, 3.0], [1.1, 0.9, 2.8], reference_scale=2.0
        )
        self.assertTrue(np.isfinite(metrics["reference_nrmse"]))
        self.assertAlmostEqual(metrics["reference_nrmse"], np.sqrt(0.02) / 2.0)

    def test_neighborhood_metrics_detect_perfect_geometry_and_permutation(self) -> None:
        reference = np.arange(12, dtype=float).reshape(-1, 1)
        learned = reference * 3.0 + 7.0
        perfect = trustworthiness_continuity(reference, learned, k=3)
        self.assertEqual(perfect, {"trustworthiness": 1.0, "continuity": 1.0, "knn_overlap": 1.0})
        distortion = local_distance_distortion(reference, learned, k=3)
        self.assertLess(distortion["local_log_distortion_p95"], 1e-12)

        permuted = learned[np.array([0, 11, 2, 9, 4, 7, 6, 5, 8, 3, 10, 1])]
        damaged = trustworthiness_continuity(reference, permuted, k=3)
        self.assertLess(damaged["trustworthiness"], 0.9)
        self.assertLess(damaged["continuity"], 0.9)
        curve = neighborhood_preservation_curve(reference, permuted, max_k=4)
        self.assertEqual([row["k"] for row in curve], [1.0, 2.0, 3.0, 4.0])

    def test_grouped_rff_signatures_are_deterministic_and_label_level(self) -> None:
        values = np.array([[0.0, 1.0], [0.1, 1.1], [2.0, 3.0], [2.1, 3.1]])
        labels = np.array(["a", "a", "b", "b"])
        first_labels, first = grouped_rff_signatures(values, labels, n_components=16, seed=9)
        second_labels, second = grouped_rff_signatures(values, labels, n_components=16, seed=9)
        np.testing.assert_array_equal(first_labels, ["a", "b"])
        np.testing.assert_array_equal(first_labels, second_labels)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
