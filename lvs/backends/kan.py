from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

import torch.nn as nn

from lvs.core.pipeline import (
    CSVColumnConfig,
    LatentQConfig,
    LatentQDataset,
    LatentQPipelineResult,
    OutputConfig,
    add_common_cli_arguments,
    build_dataset_from_arrays,
    load_csv_dataset,
    namespace_to_shared_configs,
    run_latent_q_pipeline,
)

DEFAULT_PLOT_OUTPUT_NAME = "fit_vs_real_latent_q_kan.png"
DEFAULT_PLOT_TITLE = "Fit vs Real Curve by Selected Feature (KAN, Test Split)"


def build_kan_model_factory(
    *,
    grid: int = 5,
    spline_order: int = 3,
    seed: int = 42,
    checkpoint_dir: Optional[Path] = None,
):
    if grid <= 0:
        raise ValueError("grid must be a positive integer.")
    if spline_order <= 0:
        raise ValueError("spline_order must be a positive integer.")
    resolved_checkpoint_dir = checkpoint_dir or _build_default_checkpoint_dir()
    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def factory(input_dim: int) -> nn.Module:
        try:
            from kan import KAN
        except ImportError as exc:
            raise ImportError(
                "KAN backend is not available. Install the `pykan` package before running q_optimize_kan.py "
                f"(its import name is `kan`). Original import error: {exc}"
            ) from exc
        return KAN(
            width=[input_dim, 8, 1],
            grid=grid,
            k=spline_order,
            seed=seed,
            ckpt_path=str(resolved_checkpoint_dir),
            auto_save=True,
        )

    return factory


def run_kan_latent_q(
    train_dataset: LatentQDataset,
    test_dataset: LatentQDataset,
    *,
    config: Optional[LatentQConfig] = None,
    output_config: Optional[OutputConfig] = None,
    kan_grid: int = 5,
    kan_order: int = 3,
) -> LatentQPipelineResult:
    resolved_config = config or LatentQConfig()
    checkpoint_dir = _resolve_kan_checkpoint_dir(output_config)
    model_factory = build_kan_model_factory(
        grid=kan_grid,
        spline_order=kan_order,
        seed=resolved_config.seed,
        checkpoint_dir=checkpoint_dir,
    )
    return run_latent_q_pipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_factory=model_factory,
        config=resolved_config,
        output_config=output_config,
    )


def run_kan_latent_q_from_csv(
    train_csv: Path | str,
    test_csv: Path | str,
    *,
    column_config: Optional[CSVColumnConfig] = None,
    config: Optional[LatentQConfig] = None,
    output_config: Optional[OutputConfig] = None,
    kan_grid: int = 5,
    kan_order: int = 3,
) -> LatentQPipelineResult:
    resolved_column_config = column_config or CSVColumnConfig()
    train_dataset = load_csv_dataset(train_csv, resolved_column_config)
    test_dataset = load_csv_dataset(test_csv, resolved_column_config)
    return run_kan_latent_q(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_config=output_config,
        kan_grid=kan_grid,
        kan_order=kan_order,
    )


def run_kan_latent_q_from_arrays(
    *,
    train_features,
    train_labels,
    train_targets,
    test_features,
    test_labels,
    test_targets,
    feature_names: Optional[Sequence[str]] = None,
    label_name: str = "label",
    target_name: str = "target",
    config: Optional[LatentQConfig] = None,
    output_config: Optional[OutputConfig] = None,
    kan_grid: int = 5,
    kan_order: int = 3,
) -> LatentQPipelineResult:
    train_dataset = build_dataset_from_arrays(
        features=train_features,
        labels=train_labels,
        targets=train_targets,
        feature_names=feature_names,
        label_name=label_name,
        target_name=target_name,
    )
    test_dataset = build_dataset_from_arrays(
        features=test_features,
        labels=test_labels,
        targets=test_targets,
        feature_names=feature_names,
        label_name=label_name,
        target_name=target_name,
    )
    return run_kan_latent_q(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_config=output_config,
        kan_grid=kan_grid,
        kan_order=kan_order,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent q optimization pipeline powered by a KAN regressor.")
    add_common_cli_arguments(
        parser,
        default_plot_output_name=DEFAULT_PLOT_OUTPUT_NAME,
        default_plot_title=DEFAULT_PLOT_TITLE,
    )
    parser.add_argument("--kan-grid", "--kan_grid", dest="kan_grid", type=int, default=5, help="KAN grid size.")
    parser.add_argument(
        "--kan-order",
        "--kan-k",
        "--kan_k",
        dest="kan_order",
        type=int,
        default=3,
        help="KAN spline order.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    column_config, pipeline_config, output_config = namespace_to_shared_configs(args)

    result = run_kan_latent_q_from_csv(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        column_config=column_config,
        config=pipeline_config,
        output_config=output_config,
        kan_grid=args.kan_grid,
        kan_order=args.kan_order,
    )

    print(f"Test R2: {result.metrics['test_r2']:.6f}")
    print(f"Test MSE: {result.metrics['test_mse']:.6f}")
    for output_name, output_path in result.saved_paths.items():
        print(f"Saved {output_name}: {output_path}")


def _resolve_kan_checkpoint_dir(output_config: Optional[OutputConfig]) -> Path:
    if output_config is None:
        return _build_default_checkpoint_dir()
    return Path(output_config.output_dir) / "kan_checkpoints" / f"kan_run_{uuid4().hex[:12]}"


def _build_default_checkpoint_dir() -> Path:
    return Path("model") / f"kan_run_{uuid4().hex[:12]}"


if __name__ == "__main__":
    main()


__all__ = [
    "build_kan_model_factory",
    "build_parser",
    "run_kan_latent_q",
    "run_kan_latent_q_from_arrays",
    "run_kan_latent_q_from_csv",
]
