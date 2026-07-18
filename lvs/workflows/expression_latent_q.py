from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from lvs.core.expression_library import (
    describe_expression_support,
    load_expression_library,
    sample_expression_dataset,
    save_generated_expression_dataset,
    select_expression_task,
)

DEFAULT_LIBRARY_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "latent_variable_expressions.csv"

if TYPE_CHECKING:
    from lvs.core.pipeline import LatentQConfig, OutputConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a latent-variable expression template, generate synthetic train/test data, "
            "and run the latent-q pipeline directly without the run_nn baseline."
        )
    )
    parser.add_argument(
        "--library-csv",
        type=Path,
        default=DEFAULT_LIBRARY_CSV,
        help="Path to the latent-variable expression library CSV.",
    )
    parser.add_argument(
        "--list-expressions",
        action="store_true",
        help="List supported and unsupported expression templates, then exit.",
    )
    parser.add_argument("--expression-id", type=int, default=None, help="Expression ID to run.")
    parser.add_argument("--expression-name", type=str, default=None, help="Exact expression name to run.")
    parser.add_argument("--label-count", type=int, default=50, help="Number of labels / groups to sample.")
    parser.add_argument(
        "--train-samples-per-label",
        type=int,
        default=80,
        help="Number of train samples generated for each label.",
    )
    parser.add_argument(
        "--test-samples-per-label",
        type=int,
        default=30,
        help="Number of test samples generated for each label.",
    )
    parser.add_argument("--noise-std", type=float, default=0.0, help="Optional Gaussian noise std added to target.")
    parser.add_argument(
        "--max-attempts-per-row",
        type=int,
        default=200,
        help="Max resampling attempts per generated row when the formula hits an invalid domain.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for both data generation and model training.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("torch", "kan"),
        default="torch",
        help="Latent-q model backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("expression_run"),
        help="Directory used for generated datasets, latent-q outputs, and plots.",
    )
    parser.add_argument(
        "--generated-train-name",
        type=str,
        default="generated_train.csv",
        help="Generated training CSV filename.",
    )
    parser.add_argument(
        "--generated-test-name",
        type=str,
        default="generated_test.csv",
        help="Generated test CSV filename.",
    )
    parser.add_argument(
        "--latent-truth-name",
        type=str,
        default="latent_truth.csv",
        help="Generated latent ground-truth CSV filename.",
    )
    parser.add_argument(
        "--expression-metadata-name",
        type=str,
        default="expression_metadata.json",
        help="Generated metadata JSON filename.",
    )
    parser.add_argument(
        "--skip-generated-save",
        action="store_true",
        help="Skip saving generated train/test/metadata files.",
    )
    parser.add_argument("--q-dim", "--q_dim", dest="q_dim", type=int, default=2, help="Latent q dimension.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--cal-steps", type=int, default=200, help="Test-time q calibration steps.")
    parser.add_argument("--cal-lr", type=float, default=0.05, help="Test-time q calibration learning rate.")
    parser.add_argument("--cal-ratio", type=float, default=0.3, help="Per-label calibration ratio on test set.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device such as cpu, cuda, or cuda:0. Defaults to auto detection.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable epoch-level logging.")
    parser.add_argument(
        "--train-output-name",
        type=str,
        default="train_with_q.csv",
        help="Output CSV name for the training set augmented with q.",
    )
    parser.add_argument(
        "--test-output-name",
        type=str,
        default="test_with_q.csv",
        help="Output CSV name for the test set augmented with q.",
    )
    parser.add_argument(
        "--plot-output-name",
        type=str,
        default=None,
        help="Output plot filename. Defaults to a backend-specific name.",
    )
    parser.add_argument("--skip-save", action="store_true", help="Skip writing latent-q train/test CSV outputs.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip writing the latent-q plot.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes for the Torch backend.",
    )
    parser.add_argument("--kan-grid", type=int, default=5, help="KAN grid size.")
    parser.add_argument("--kan-order", type=int, default=3, help="KAN spline order.")
    return parser


def parse_hidden_sizes(raw_value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    hidden_sizes = tuple(int(part) for part in parts if part)
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty.")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Each hidden layer size must be a positive integer.")
    return hidden_sizes


def list_expressions(library_csv: Path | str) -> None:
    records = load_expression_library(library_csv)
    descriptions = describe_expression_support(records)
    for item in descriptions:
        status = item["status"]
        if status == "supported":
            print(
                f"[supported] id={item['expression_id']:>2} "
                f"name={item['formula_name']} "
                f"x={item['observed_feature_variables']} "
                f"q={item['latent_variables']}"
            )
        else:
            print(
                f"[unsupported] id={item['expression_id']:>2} "
                f"name={item['formula_name']} "
                f"reason={item['reason']}"
            )


def build_pipeline_config(args: argparse.Namespace):
    from lvs.core.pipeline import LatentQConfig

    return LatentQConfig(
        q_dim=args.q_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        calibration_steps=args.cal_steps,
        calibration_lr=args.cal_lr,
        calibration_ratio=args.cal_ratio,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
    )


def build_output_config(args: argparse.Namespace):
    from lvs.core.pipeline import OutputConfig

    default_plot_name = (
        "column2_fit_vs_real_latent_q_torch.png"
        if args.backend == "torch"
        else "column2_fit_vs_real_latent_q_kan.png"
    )
    plot_output_name = args.plot_output_name or default_plot_name
    return OutputConfig(
        output_dir=args.output_dir,
        train_output_name=args.train_output_name,
        test_output_name=args.test_output_name,
        plot_output_name=plot_output_name,
        save_csv=not args.skip_save,
        save_plot=not args.skip_plot,
        plot_feature_index=0,
        plot_title=f"Fit vs Real Curve by Selected Feature ({args.backend.upper()}, Expression Workflow)",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_expressions:
        list_expressions(args.library_csv)
        return

    records = load_expression_library(args.library_csv)
    task = select_expression_task(
        records,
        expression_id=args.expression_id,
        formula_name=args.expression_name,
    )

    generated_dataset = sample_expression_dataset(
        task,
        label_count=args.label_count,
        train_samples_per_label=args.train_samples_per_label,
        test_samples_per_label=args.test_samples_per_label,
        noise_std=args.noise_std,
        seed=args.seed,
        max_attempts_per_row=args.max_attempts_per_row,
    )

    saved_paths: dict[str, Path] = {}
    if not args.skip_generated_save:
        saved_paths.update(
            save_generated_expression_dataset(
                generated_dataset,
                args.output_dir,
                train_filename=args.generated_train_name,
                test_filename=args.generated_test_name,
                latent_truth_filename=args.latent_truth_name,
                metadata_filename=args.expression_metadata_name,
                include_header=True,
            )
        )

    pipeline_config = build_pipeline_config(args)
    output_config = build_output_config(args)

    if args.backend == "torch":
        from lvs.backends.torch_mlp import run_torch_latent_q_from_arrays

        result = run_torch_latent_q_from_arrays(
            train_features=generated_dataset.train_frame.loc[:, list(task.feature_columns)].to_numpy(),
            train_labels=generated_dataset.train_frame["label"].to_numpy(),
            train_targets=generated_dataset.train_frame["target"].to_numpy(),
            test_features=generated_dataset.test_frame.loc[:, list(task.feature_columns)].to_numpy(),
            test_labels=generated_dataset.test_frame["label"].to_numpy(),
            test_targets=generated_dataset.test_frame["target"].to_numpy(),
            feature_names=task.feature_columns,
            label_name="label",
            target_name="target",
            config=pipeline_config,
            output_config=output_config,
            hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        )
    else:
        from lvs.backends.kan import run_kan_latent_q_from_arrays

        result = run_kan_latent_q_from_arrays(
            train_features=generated_dataset.train_frame.loc[:, list(task.feature_columns)].to_numpy(),
            train_labels=generated_dataset.train_frame["label"].to_numpy(),
            train_targets=generated_dataset.train_frame["target"].to_numpy(),
            test_features=generated_dataset.test_frame.loc[:, list(task.feature_columns)].to_numpy(),
            test_labels=generated_dataset.test_frame["label"].to_numpy(),
            test_targets=generated_dataset.test_frame["target"].to_numpy(),
            feature_names=task.feature_columns,
            label_name="label",
            target_name="target",
            config=pipeline_config,
            output_config=output_config,
            kan_grid=args.kan_grid,
            kan_order=args.kan_order,
        )

    saved_paths.update(result.saved_paths)

    print(f"Expression ID: {task.expression_id}")
    print(f"Expression Name: {task.formula_name}")
    print(f"Formula: {task.raw_formula}")
    print(f"Observed feature variables: {list(task.observed_feature_variables)}")
    print(f"Latent variables: {list(task.latent_variables)}")
    print(f"Generated feature columns: {list(task.feature_columns)}")
    print(f"Test R2: {result.metrics['test_r2']:.6f}")
    print(f"Test MSE: {result.metrics['test_mse']:.6f}")
    for output_name, output_path in saved_paths.items():
        print(f"Saved {output_name}: {output_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "build_parser",
    "build_output_config",
    "build_pipeline_config",
    "list_expressions",
    "main",
    "parse_hidden_sizes",
]
