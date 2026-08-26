"""Named loss bundles used by the controlled latent-q ablation studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LossPreset:
    """Only the loss fields that vary in the component ablation."""

    prediction_loss_type: str = "mse"
    latent_feature_orthogonality_weight: float = 0.0
    latent_feature_orthogonality_type: str = "hsic"
    latent_feature_stats_mode: str = "rich_rff_kme"
    latent_curve_continuity_weight: float = 0.0
    latent_q_l2_weight: float = 0.0
    calibration_q_prior_weight: float = 0.0

    def config_kwargs(self) -> dict[str, Any]:
        return asdict(self)


LOSS_PRESETS: dict[str, LossPreset] = {
    "mse": LossPreset(),
    "label_balanced_mse": LossPreset(prediction_loss_type="label_balanced_mse"),
    "hsic": LossPreset(latent_feature_orthogonality_weight=0.05),
    "continuity": LossPreset(latent_curve_continuity_weight=0.05),
    "q_l2": LossPreset(latent_q_l2_weight=0.001),
    "calibration_prior": LossPreset(calibration_q_prior_weight=0.01),
    "hsic_continuity": LossPreset(
        latent_feature_orthogonality_weight=0.05,
        latent_curve_continuity_weight=0.05,
    ),
    "all_mse": LossPreset(
        latent_feature_orthogonality_weight=0.05,
        latent_curve_continuity_weight=0.05,
        latent_q_l2_weight=0.001,
        calibration_q_prior_weight=0.01,
    ),
    "all_label_balanced": LossPreset(
        prediction_loss_type="label_balanced_mse",
        latent_feature_orthogonality_weight=0.05,
        latent_curve_continuity_weight=0.05,
        latent_q_l2_weight=0.001,
        calibration_q_prior_weight=0.01,
    ),
}

# Dose-response and alternative-dependence presets are intentionally kept out
# of the runners' primary method lists.  The extended campaign opts into them
# explicitly after the component screen.
LOSS_PRESETS.update(
    {
        "hsic_w0p005": LossPreset(latent_feature_orthogonality_weight=0.005),
        "hsic_w0p01": LossPreset(latent_feature_orthogonality_weight=0.01),
        "hsic_w0p02": LossPreset(latent_feature_orthogonality_weight=0.02),
        "hsic_w0p10": LossPreset(latent_feature_orthogonality_weight=0.10),
        "hsic_w0p20": LossPreset(latent_feature_orthogonality_weight=0.20),
        "continuity_w0p005": LossPreset(latent_curve_continuity_weight=0.005),
        "continuity_w0p01": LossPreset(latent_curve_continuity_weight=0.01),
        "continuity_w0p02": LossPreset(latent_curve_continuity_weight=0.02),
        "continuity_w0p10": LossPreset(latent_curve_continuity_weight=0.10),
        "continuity_w0p20": LossPreset(latent_curve_continuity_weight=0.20),
        "q_l2_w0p0001": LossPreset(latent_q_l2_weight=0.0001),
        "q_l2_w0p0003": LossPreset(latent_q_l2_weight=0.0003),
        "q_l2_w0p003": LossPreset(latent_q_l2_weight=0.003),
        "q_l2_w0p01": LossPreset(latent_q_l2_weight=0.01),
        "calibration_prior_w0p001": LossPreset(calibration_q_prior_weight=0.001),
        "calibration_prior_w0p003": LossPreset(calibration_q_prior_weight=0.003),
        "calibration_prior_w0p03": LossPreset(calibration_q_prior_weight=0.03),
        "calibration_prior_w0p10": LossPreset(calibration_q_prior_weight=0.10),
        "orth_pearson": LossPreset(
            latent_feature_orthogonality_weight=0.05,
            latent_feature_orthogonality_type="pearson",
        ),
        "orth_nhsic": LossPreset(
            latent_feature_orthogonality_weight=0.05,
            latent_feature_orthogonality_type="nhsic",
        ),
        "orth_distance_correlation": LossPreset(
            latent_feature_orthogonality_weight=0.05,
            latent_feature_orthogonality_type="distance_correlation",
        ),
        "orth_propensity": LossPreset(
            latent_feature_orthogonality_weight=0.05,
            latent_feature_orthogonality_type="propensity",
        ),
        "orth_adversarial": LossPreset(
            latent_feature_orthogonality_weight=0.05,
            latent_feature_orthogonality_type="adversarial",
        ),
    }
)

LOSS_SWEEP_METHOD_PRESETS: dict[str, str] = {
    "joint_hsic_w005": "hsic_w0p005",
    "joint_hsic_w01": "hsic_w0p01",
    "joint_hsic_w02": "hsic_w0p02",
    "joint_hsic_w10": "hsic_w0p10",
    "joint_hsic_w20": "hsic_w0p20",
    "joint_cont_w005": "continuity_w0p005",
    "joint_cont_w01": "continuity_w0p01",
    "joint_cont_w02": "continuity_w0p02",
    "joint_cont_w10": "continuity_w0p10",
    "joint_cont_w20": "continuity_w0p20",
    "joint_ql2_w0001": "q_l2_w0p0001",
    "joint_ql2_w0003": "q_l2_w0p0003",
    "joint_ql2_w003": "q_l2_w0p003",
    "joint_ql2_w01": "q_l2_w0p01",
    "joint_calprior_w001": "calibration_prior_w0p001",
    "joint_calprior_w003": "calibration_prior_w0p003",
    "joint_calprior_w03": "calibration_prior_w0p03",
    "joint_calprior_w10": "calibration_prior_w0p10",
    "joint_orth_pearson": "orth_pearson",
    "joint_orth_nhsic": "orth_nhsic",
    "joint_orth_dcor": "orth_distance_correlation",
    "joint_orth_propensity": "orth_propensity",
    "joint_orth_adversarial": "orth_adversarial",
}


def get_loss_preset(name: str) -> LossPreset:
    try:
        return LOSS_PRESETS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown loss preset {name!r}; expected one of {sorted(LOSS_PRESETS)}."
        ) from error
