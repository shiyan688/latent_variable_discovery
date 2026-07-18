#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANGE_LINE = re.compile(r"^\s*([A-Za-z]\d+)\s*:\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an expression library copy with standardized observed x ranges.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "latent_variable_expressions.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "latent_variable_expressions_xrange10.csv",
    )
    parser.add_argument(
        "--signed-policy",
        choices=["preserve_negative", "all_positive", "all_signed"],
        default="preserve_negative",
        help=(
            "How to choose x ranges. preserve_negative maps originally nonnegative x to [0,10] "
            "and x ranges that include negatives to [-10,10]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    range_col = frame.columns[3]
    frame[range_col] = frame[range_col].map(lambda value: normalize_ranges(str(value), args.signed_policy))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output_path)


def normalize_ranges(raw: str, signed_policy: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        match = RANGE_LINE.match(line)
        if not match:
            lines.append(line)
            continue
        symbol, lower_text, upper_text = match.groups()
        if symbol.lower().startswith("x"):
            lower, upper = float(lower_text), float(upper_text)
            if signed_policy == "all_positive":
                new_lower, new_upper = 0.0, 10.0
            elif signed_policy == "all_signed":
                new_lower, new_upper = -10.0, 10.0
            elif lower < 0 < upper or upper <= 0:
                new_lower, new_upper = -10.0, 10.0
            else:
                new_lower, new_upper = 0.0, 10.0
            lines.append(f"{symbol}: [{format_bound(new_lower)}, {format_bound(new_upper)}]")
        else:
            lines.append(line)
    return "\n".join(lines)


def format_bound(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    main()
