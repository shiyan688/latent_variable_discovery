from __future__ import annotations

from argparse import Namespace

import pytest

from lvs.core.loss_presets import LOSS_PRESETS, get_loss_preset
from scripts.run_iclr_real_discovery import METHODS as REAL_METHODS
from scripts.run_iclr_real_discovery import _latent_config
from scripts.run_pdebench_burgers_latent_study import METHODS as PDE_METHODS
from scripts.run_pdebench_burgers_latent_study import _base_config as _pde_base_config


def _args() -> Namespace:
    return Namespace(
        q_dim=4,
        epochs=2,
        batch_size=32,
        cal_steps=3,
        support_ratio=0.3,
        support_split_mode="random",
        cal_init_mode="prior_random",
        cal_num_starts=4,
        cal_selection_ratio=0.25,
        cal_selection_min_rows=24,
        cal_refine_steps=50,
        cal_refine_only_after_selection=True,
        seed=7,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("name", "field", "expected"),
    [
        ("label_balanced_mse", "prediction_loss_type", "label_balanced_mse"),
        ("hsic", "latent_feature_orthogonality_weight", 0.05),
        ("continuity", "latent_curve_continuity_weight", 0.05),
        ("q_l2", "latent_q_l2_weight", 0.001),
        ("calibration_prior", "calibration_q_prior_weight", 0.01),
    ],
)
def test_single_component_presets_change_exactly_one_effective_field(
    name: str, field: str, expected: object
) -> None:
    baseline = LOSS_PRESETS["mse"].config_kwargs()
    candidate = LOSS_PRESETS[name].config_kwargs()
    changed = {key for key in baseline if baseline[key] != candidate[key]}
    assert changed == {field}
    assert candidate[field] == expected


def test_full_bundle_matches_historical_fixed_loss() -> None:
    preset = get_loss_preset("all_label_balanced")
    assert preset.prediction_loss_type == "label_balanced_mse"
    assert preset.latent_feature_orthogonality_type == "hsic"
    assert preset.latent_feature_stats_mode == "rich_rff_kme"
    assert preset.latent_feature_orthogonality_weight == 0.05
    assert preset.latent_curve_continuity_weight == 0.05
    assert preset.latent_q_l2_weight == 0.001
    assert preset.calibration_q_prior_weight == 0.01


def test_runner_carries_promoted_calibration_and_named_loss_into_config() -> None:
    config = _latent_config(_args(), REAL_METHODS["joint_hsic"])
    assert config.q_dim == 4
    assert config.calibration_init_mode == "prior_random"
    assert config.calibration_num_starts == 4
    assert config.calibration_selection_ratio == 0.25
    assert config.calibration_selection_min_rows == 24
    assert config.calibration_refine_steps == 50
    assert config.calibration_refine_only_after_selection is True
    assert config.latent_feature_orthogonality_weight == 0.05
    assert config.latent_curve_continuity_weight == 0.0


def test_dynamic_variant_changes_weighting_not_loss_bundle() -> None:
    fixed = REAL_METHODS["joint_fixed"]
    dynamic = REAL_METHODS["joint_dynamic"]
    assert fixed.loss_preset == dynamic.loss_preset == "all_label_balanced"
    assert fixed.weighting == "static"
    assert dynamic.weighting == "adaptive_loss_scale"


def test_matched_update_real_variants_use_one_joint_step() -> None:
    mse = _latent_config(_args(), REAL_METHODS["joint_mse_step1"])
    continuity = _latent_config(_args(), REAL_METHODS["joint_continuity_step1"])
    assert mse.joint_steps_per_cycle == 1
    assert continuity.joint_steps_per_cycle == 1
    assert mse.latent_curve_continuity_weight == 0.0
    assert continuity.latent_curve_continuity_weight == 0.05


def test_matched_update_pde_variants_use_one_joint_step() -> None:
    args = _args()
    args.method = "joint_mse_step1"
    mse = _pde_base_config(args, q_dim=16)
    args.method = "joint_continuity_step1"
    continuity = _pde_base_config(args, q_dim=16)
    assert PDE_METHODS["joint_mse_step1"].joint_steps_per_cycle == 1
    assert mse.joint_steps_per_cycle == 1
    assert continuity.joint_steps_per_cycle == 1
    assert mse.latent_curve_continuity_weight == 0.0
    assert continuity.latent_curve_continuity_weight == 0.05
