#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Finding:
    severity: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit application dataset inputs for physical and reviewer-facing validity.")
    parser.add_argument(
        "--prepared-summary",
        action="append",
        type=Path,
        default=[],
        help="Prepared dataset summary JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "application_input_audits" / f"{datetime.now():%Y%m%d_%H%M%S}_input_audit.md",
    )
    parser.add_argument("--corr-threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = args.prepared_summary or [
        PROJECT_ROOT / "data" / "application" / "prepared_datasets.json",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
        PROJECT_ROOT / "data" / "application_full_features_sample_label" / "prepared_datasets.json",
    ]
    records: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in summaries:
        summary_path = resolve_path(summary_path)
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset_records = payload.get("datasets", payload) if isinstance(payload, dict) else payload
        for record in dataset_records:
            records.append((summary_path, record))

    lines = [
        "# Application Input Audit",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Policy:",
        "",
        "- `label` is allowed only as a curve/sample identifier used to assign or calibrate latent q; it must not appear in `feature_columns`.",
        "- Features must be available at the prediction time implied by the task.",
        "- Same-cycle target proxies are leakage unless explicitly declared as an upper-bound reconstruction experiment.",
        "- Smoke-test representations are not valid main-paper baselines.",
        "",
    ]

    for summary_path, record in records:
        lines.extend(audit_record(summary_path, record, args.corr_threshold))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


def audit_record(summary_path: Path, record: dict[str, Any], corr_threshold: float) -> list[str]:
    name = str(record["name"])
    feature_columns = list(record.get("feature_columns", []))
    target_column = str(record.get("target_column", "target"))
    label_column = str(record.get("label_column", "label"))
    train_csv = resolve_path(Path(record["train_csv"]))
    test_csv = resolve_path(Path(record["test_csv"]))
    findings: list[Finding] = []

    if label_column in feature_columns or "label" in feature_columns:
        findings.append(Finding("fail", "`label` appears in feature columns. This would make the hidden variable non-hidden."))
    if target_column in feature_columns or "target" in feature_columns:
        findings.append(Finding("fail", "`target` appears in feature columns."))

    train = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    test = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()
    if train.empty or test.empty:
        findings.append(Finding("fail", "Train/test CSV missing or empty."))
    else:
        findings.extend(generic_numeric_checks(train, feature_columns, target_column, corr_threshold))

    findings.extend(domain_checks(name, feature_columns, train))
    verdict = overall_verdict(findings)

    lines = [
        f"## {name}",
        "",
        f"- Summary: `{relative(summary_path)}`",
        f"- Train CSV: `{relative(train_csv)}`",
        f"- Test CSV: `{relative(test_csv)}`",
        f"- Features ({len(feature_columns)}): `{', '.join(feature_columns[:18])}{'...' if len(feature_columns) > 18 else ''}`",
        f"- Verdict: **{verdict}**",
        "",
        "| Severity | Finding |",
        "|---|---|",
    ]
    if findings:
        for finding in findings:
            lines.append(f"| {finding.severity} | {finding.message} |")
    else:
        lines.append("| pass | No immediate input validity issues detected by this audit. |")
    lines.append("")
    return lines


def generic_numeric_checks(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    corr_threshold: float,
) -> list[Finding]:
    findings: list[Finding] = []
    if target_column not in train.columns:
        findings.append(Finding("fail", f"Target column `{target_column}` is missing."))
        return findings
    target = pd.to_numeric(train[target_column], errors="coerce")
    for column in feature_columns:
        if column not in train.columns:
            findings.append(Finding("fail", f"Feature `{column}` is missing from train CSV."))
            continue
        values = pd.to_numeric(train[column], errors="coerce")
        if values.notna().sum() < 3 or target.notna().sum() < 3:
            continue
        if values.nunique(dropna=True) <= 1:
            findings.append(Finding("warn", f"Feature `{column}` is constant in train split."))
            continue
        corr = values.corr(target)
        if pd.notna(corr) and abs(float(corr)) >= corr_threshold:
            findings.append(
                Finding(
                    "warn",
                    f"Feature `{column}` has high absolute Pearson correlation with target in train split: {corr:.4f}. Check for target proxy/leakage.",
                )
            )
    return findings


def domain_checks(name: str, feature_columns: list[str], train: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    feature_set = set(feature_columns)

    if name.startswith("starry_te_"):
        if feature_columns == ["temperature"]:
            findings.append(
                Finding(
                    "warn",
                    "StarryData2 temperature-only input is a legacy weak baseline. It is physically interpretable, but reviewers will expect composition descriptors for material identity.",
                )
            )
        elif "temperature" in feature_set and (
            any(column.startswith("elem_") for column in feature_columns)
            or any(column.startswith("comp_") for column in feature_columns)
        ):
            findings.append(
                Finding(
                    "pass",
                    "StarryData2 inputs are physically meaningful: temperature plus composition-derived descriptors. For stronger reviewer acceptance, compare against Magpie/matminer-style descriptors when available.",
                )
            )
        else:
            findings.append(Finding("warn", "StarryData2 input should include temperature and material composition descriptors."))
        if not train.empty and "label" in train.columns:
            label_sample = str(train["label"].iloc[0])
            if not label_sample.isdigit():
                findings.append(
                    Finding(
                        "warn",
                        "StarryData2 labels look like composition strings. That groups multiple experiments by formula and is less aligned with sample-level hidden intrinsic variables.",
                    )
                )
            else:
                findings.append(Finding("pass", "StarryData2 label looks like a sample identifier; it is used only for latent q grouping."))

    if name.startswith("battery_matr"):
        if "q_charge" in feature_set:
            findings.append(
                Finding(
                    "fail",
                    "`q_charge` is same-cycle charge capacity and is a direct proxy for discharge capacity target. Use only as an explicitly labeled leakage upper bound.",
                )
            )
        same_cycle = sorted(feature_set & {"ir", "t_avg", "t_max", "t_min", "charge_time"})
        if same_cycle:
            findings.append(
                Finding(
                    "warn",
                    f"Same-cycle diagnostic features `{', '.join(same_cycle)}` are valid only if the task is post-cycle capacity reconstruction. For prospective prediction, restrict to protocol and early-cycle summaries.",
                )
            )
        protocol = sorted(feature_set & {"charge_c_rate", "charge_percent", "discharge_c_rate"})
        if protocol:
            findings.append(Finding("pass", f"Protocol features `{', '.join(protocol)}` are reviewer-friendly: known before cycling."))
        if "cycle" in feature_set:
            findings.append(Finding("pass", "`cycle` is a meaningful curve coordinate for capacity fade."))

    if name.startswith("oc20"):
        if set(feature_columns) <= {"frame", "natoms"}:
            findings.append(
                Finding(
                    "fail",
                    "OC20 with only `frame,natoms` is a smoke test, not a physically sufficient catalyst representation. Main experiments need atomic numbers, positions, cell/adsorbate graph, energy and forces.",
                )
            )

    if name.startswith("matbench"):
        findings.append(
            Finding(
                "warn",
                "Standard Matbench tasks are one-material/one-property supervised tasks, not repeated curves per label. Use a separate supervised baseline or reformulate before latent-q comparison.",
            )
        )

    return findings


def overall_verdict(findings: list[Finding]) -> str:
    if any(finding.severity == "fail" for finding in findings):
        return "not main-paper valid"
    if any(finding.severity == "warn" for finding in findings):
        return "usable with caveats / baseline only"
    return "main-paper candidate"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
