from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from lvs.core.pipeline import (
    CSVColumnConfig,
    LatentQConfig,
    LatentQDataset,
    LatentQPipelineResult,
    OutputConfig,
    add_common_cli_arguments,
    build_dataset_from_arrays,
    evaluate_latent_q_pipeline,
    load_csv_dataset,
    namespace_to_shared_configs,
    run_latent_q_pipeline,
)

DEFAULT_HIDDEN_SIZES = (128, 64)
DEFAULT_PLOT_OUTPUT_NAME = "fit_vs_real_latent_q_torch.png"
DEFAULT_PLOT_TITLE = "Fit vs Real Curve by Selected Feature (Torch, Test Split)"


class TorchMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def parse_hidden_sizes(raw_value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    hidden_sizes = tuple(int(part) for part in parts if part)
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty.")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Each hidden layer size must be a positive integer.")
    return hidden_sizes


def build_torch_model_factory(hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES):
    resolved_hidden_sizes = tuple(hidden_sizes)
    if not resolved_hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty.")
    if any(size <= 0 for size in resolved_hidden_sizes):
        raise ValueError("Each hidden layer size must be a positive integer.")

    def factory(input_dim: int) -> nn.Module:
        return TorchMLPRegressor(input_dim=input_dim, hidden_sizes=resolved_hidden_sizes)

    return factory


def run_torch_latent_q(
    train_dataset: LatentQDataset,
    test_dataset: LatentQDataset,
    *,
    config: Optional[LatentQConfig] = None,
    output_config: Optional[OutputConfig] = None,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
) -> LatentQPipelineResult:
    resolved_config = config or LatentQConfig()
    model_factory = build_torch_model_factory(hidden_sizes)
    return run_latent_q_pipeline(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_factory=model_factory,
        config=resolved_config,
        output_config=output_config,
    )


def run_torch_latent_q_from_csv(
    train_csv: Path | str,
    test_csv: Path | str,
    *,
    column_config: Optional[CSVColumnConfig] = None,
    config: Optional[LatentQConfig] = None,
    output_config: Optional[OutputConfig] = None,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
) -> LatentQPipelineResult:
    resolved_column_config = column_config or CSVColumnConfig()
    train_dataset = load_csv_dataset(train_csv, resolved_column_config)
    test_dataset = load_csv_dataset(test_csv, resolved_column_config)
    return run_torch_latent_q(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_config=output_config,
        hidden_sizes=hidden_sizes,
    )


def run_torch_latent_q_from_arrays(
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
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
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
    return run_torch_latent_q(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
        output_config=output_config,
        hidden_sizes=hidden_sizes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent q optimization pipeline powered by a Torch MLP regressor.")
    add_common_cli_arguments(
        parser,
        default_plot_output_name=DEFAULT_PLOT_OUTPUT_NAME,
        default_plot_title=DEFAULT_PLOT_TITLE,
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes for the Torch MLP regressor.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    column_config, pipeline_config, output_config = namespace_to_shared_configs(args)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    result = run_torch_latent_q_from_csv(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        column_config=column_config,
        config=pipeline_config,
        output_config=output_config,
        hidden_sizes=hidden_sizes,
    )

    print(f"Test R2: {result.metrics['test_r2']:.6f}")
    print(f"Test MSE: {result.metrics['test_mse']:.6f}")
    for output_name, output_path in result.saved_paths.items():
        print(f"Saved {output_name}: {output_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_HIDDEN_SIZES",
    "TorchMLPRegressor",
    "build_parser",
    "build_torch_model_factory",
    "parse_hidden_sizes",
    "run_torch_latent_q",
    "run_torch_latent_q_from_arrays",
    "run_torch_latent_q_from_csv",
]
