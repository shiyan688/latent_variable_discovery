#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Setting:
    name: str
    orth_weight: float
    continuity_weight: float
    cal_q_prior_weight: float
    cal_ratio: float


DEFAULT_SETTINGS = [
    Setting("mse_only_cal03_prior001", 0.0, 0.0, 0.01, 0.3),
    Setting("orth_only_cal03_prior001", 0.05, 0.0, 0.01, 0.3),
    Setting("cont_only_cal03_prior001", 0.0, 0.05, 0.01, 0.3),
    Setting("orth_cont_cal03_prior003", 0.05, 0.05, 0.03, 0.3),
    Setting("orth_cont_cal05_prior001", 0.05, 0.05, 0.01, 0.5),
    Setting("orth_cont_cal05_prior003", 0.05, 0.05, 0.03, 0.5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run torch-only expression loss/parameter ablation scan.")
    parser.add_argument("--library-csv", type=Path, default=PROJECT_ROOT / "data" / "latent_variable_expressions.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "expressions_qdim123_ablation_scan")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--q-dims", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cal-steps", type=int, default=1200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--only-settings", default=None, help="Comma-separated setting names to run.")
    parser.add_argument("--only-expression-ids", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.log"
    settings = select_settings(args.only_settings)

    append_status(status_path, f"start settings={','.join(setting.name for setting in settings)}")
    all_results: list[dict[str, Any]] = []
    for setting in settings:
        setting_root = output_root / setting.name
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_expression_qdim_torch_grid.py"),
            "--library-csv",
            str(resolve(args.library_csv)),
            "--output-root",
            str(setting_root),
            "--gpus",
            args.gpus,
            "--max-parallel",
            str(args.max_parallel),
            "--q-dims",
            args.q_dims,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--cal-steps",
            str(args.cal_steps),
            "--cal-lr",
            str(args.cal_lr),
            "--cal-ratio",
            str(setting.cal_ratio),
            "--orth-weight",
            str(setting.orth_weight),
            "--continuity-weight",
            str(setting.continuity_weight),
            "--cal-q-prior-weight",
            str(setting.cal_q_prior_weight),
            "--hidden-sizes",
            args.hidden_sizes,
            "--quiet",
        ]
        if args.only_expression_ids:
            cmd.extend(["--only-expression-ids", args.only_expression_ids])

        append_status(status_path, f"launch setting={setting.name}")
        log_path = output_root / f"{setting.name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[start] {datetime.now().isoformat(timespec='seconds')}\n")
            log.write("[command] " + " ".join(cmd) + "\n")
            log.flush()
            returncode = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"[finish] {datetime.now().isoformat(timespec='seconds')}\n")
            log.write(f"[returncode] {returncode}\n")
        result = {"setting": setting.name, "returncode": returncode}
        all_results.append(result)
        append_status(status_path, f"finished setting={setting.name} rc={returncode}")
        if returncode != 0:
            break

    (output_root / "ablation_launcher_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    append_status(status_path, "done")


def select_settings(raw: str | None) -> list[Setting]:
    if not raw:
        return DEFAULT_SETTINGS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    selected = [setting for setting in DEFAULT_SETTINGS if setting.name in wanted]
    missing = sorted(wanted - {setting.name for setting in selected})
    if missing:
        raise ValueError(f"Unknown settings: {', '.join(missing)}")
    return selected


def append_status(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
