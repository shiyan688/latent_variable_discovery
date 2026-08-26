#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


ORTH_TYPES = ("pearson", "hsic", "nhsic", "distance_correlation", "adversarial", "propensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run torch-only orthogonality-loss type scans on expression and application datasets."
    )
    parser.add_argument("--phase", choices=("expression", "application", "both"), default="both")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--orth-types", default=",".join(ORTH_TYPES))
    parser.add_argument("--orth-stats-modes", default="mean_std")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cal-steps", type=int, default=1200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--orth-weight", type=float, default=0.05)
    parser.add_argument("--continuity-weight", type=float, default=0.05)
    parser.add_argument("--cal-q-prior-weight", type=float, default=0.01)
    parser.add_argument("--latent-q-l2-weight", type=float, default=0.0)
    parser.add_argument("--prediction-loss-type", choices=("mse", "label_balanced_mse"), default="mse")
    parser.add_argument("--expression-cal-ratio", type=float, default=0.5)
    parser.add_argument("--expression-q-dims", default="1,2,3")
    parser.add_argument("--only-expression-ids", default=None)
    parser.add_argument("--application-q-dims", default="1,2,3,4")
    parser.add_argument("--application-cal-ratios", default="0.2,0.3,0.5")
    parser.add_argument("--application-cal-q-prior-weights", default="0.01,0.03")
    parser.add_argument(
        "--application-prepared-summary",
        type=Path,
        default=PROJECT_ROOT / "data" / "application_reviewer_clean" / "prepared_datasets.json",
    )
    parser.add_argument("--application-datasets", nargs="*", default=None)
    parser.add_argument("--quiet", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root) if args.output_root else PROJECT_ROOT / "runs" / f"orth_loss_type_scan_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.log"
    orth_types = parse_orth_types(args.orth_types)
    orth_stats_modes = parse_stats_modes(args.orth_stats_modes)
    append_status(
        status_path,
        f"start phase={args.phase} orth_types={','.join(orth_types)} stats_modes={','.join(orth_stats_modes)}",
    )

    if args.phase in {"expression", "both"}:
        run_expression_scan(args, output_root, status_path, orth_types, orth_stats_modes)
    if args.phase in {"application", "both"}:
        run_application_scan(args, output_root, status_path, orth_types, orth_stats_modes)

    append_status(status_path, "done")


def run_expression_scan(
    args: argparse.Namespace,
    output_root: Path,
    status_path: Path,
    orth_types: list[str],
    orth_stats_modes: list[str],
) -> None:
    expression_root = output_root / "expressions"
    for orth_type in orth_types:
        for stats_mode in orth_stats_modes:
            method_name = f"{orth_type}_{stats_mode}"
            method_root = expression_root / method_name
            cmd = [
                python_bin(),
                str(PROJECT_ROOT / "scripts" / "run_expression_qdim_torch_grid.py"),
                "--library-csv",
                str(PROJECT_ROOT / "data" / "latent_variable_expressions.csv"),
                "--output-root",
                str(method_root),
                "--gpus",
                args.gpus,
                "--max-parallel",
                str(args.max_parallel),
                "--q-dims",
                args.expression_q_dims,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--cal-steps",
                str(args.cal_steps),
                "--cal-lr",
                str(args.cal_lr),
                "--cal-ratio",
                str(args.expression_cal_ratio),
                "--orth-weight",
                str(args.orth_weight),
                "--orth-type",
                orth_type,
                "--orth-stats-mode",
                stats_mode,
                "--continuity-weight",
                str(args.continuity_weight),
                "--cal-q-prior-weight",
                str(args.cal_q_prior_weight),
                "--latent-q-l2-weight",
                str(args.latent_q_l2_weight),
                "--prediction-loss-type",
                args.prediction_loss_type,
                "--hidden-sizes",
                args.hidden_sizes,
                "--quiet",
            ]
            if args.only_expression_ids:
                cmd.extend(["--only-expression-ids", args.only_expression_ids])
            append_status(status_path, f"launch expression orth_type={orth_type} stats_mode={stats_mode}")
            returncode = call_with_log(cmd, output_root / f"expression_{method_name}.log", env=gpu_env(args.gpus))
            append_status(
                status_path,
                f"finished expression orth_type={orth_type} stats_mode={stats_mode} rc={returncode}",
            )
            if returncode != 0:
                raise SystemExit(returncode)


def run_application_scan(
    args: argparse.Namespace,
    output_root: Path,
    status_path: Path,
    orth_types: list[str],
    orth_stats_modes: list[str],
) -> None:
    app_root = output_root / "applications"
    gpu_list = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    app_gpu = gpu_list[-1] if gpu_list else None
    for orth_type in orth_types:
        for stats_mode in orth_stats_modes:
            for cal_ratio in parse_float_list(args.application_cal_ratios):
                method_name = f"{orth_type}_{stats_mode}"
                method_root = app_root / method_name / f"calratio_{cal_ratio:g}"
                cmd = [
                    python_bin(),
                    str(PROJECT_ROOT / "scripts" / "run_application_torch_ablation.py"),
                    "--prepared-summary",
                    str(resolve(args.application_prepared_summary)),
                    "--output-root",
                    str(method_root),
                    "--q-dims",
                    args.application_q_dims,
                    "--orth-weights",
                    str(args.orth_weight),
                    "--orth-types",
                    orth_type,
                    "--orth-stats-modes",
                    stats_mode,
                    "--continuity-weights",
                    str(args.continuity_weight),
                    "--cal-q-prior-weights",
                    args.application_cal_q_prior_weights,
                    "--latent-q-l2-weights",
                    str(args.latent_q_l2_weight),
                    "--prediction-loss-types",
                    args.prediction_loss_type,
                    "--hidden-sizes-list",
                    args.hidden_sizes,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--cal-steps",
                    str(args.cal_steps),
                    "--cal-lr",
                    str(args.cal_lr),
                    "--cal-ratio",
                    str(cal_ratio),
                    "--quiet",
                ]
                if args.application_datasets:
                    cmd.extend(["--datasets", *args.application_datasets])
                append_status(
                    status_path,
                    f"launch application orth_type={orth_type} stats_mode={stats_mode} cal_ratio={cal_ratio:g}",
                )
                returncode = call_with_log(
                    cmd,
                    output_root / f"application_{method_name}_calratio_{cal_ratio:g}.log",
                    env=gpu_env(app_gpu) if app_gpu is not None else None,
                )
                append_status(
                    status_path,
                    f"finished application orth_type={orth_type} stats_mode={stats_mode} cal_ratio={cal_ratio:g} rc={returncode}",
                )
                if returncode != 0:
                    raise SystemExit(returncode)


def call_with_log(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[start] {datetime.now().isoformat(timespec='seconds')}\n")
        log.write("[command] " + " ".join(cmd) + "\n")
        log.flush()
        returncode = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"[finish] {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"[returncode] {returncode}\n")
    return returncode


def gpu_env(gpus: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_orth_types(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(ORTH_TYPES))
    if unknown:
        raise ValueError(f"Unsupported orth types: {', '.join(unknown)}")
    return values


def parse_stats_modes(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    allowed = {"mean_std", "rich", "rff_kme", "rich_rff_kme"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported stats modes: {', '.join(unknown)}")
    return values


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def append_status(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def python_bin() -> str:
    return sys.executable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
