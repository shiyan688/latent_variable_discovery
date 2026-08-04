from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SUPPORTED_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "exp": math.exp,
    "fabs": math.fabs,
    "floor": math.floor,
    "log": math.log,
    "pow": math.pow,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
SYMBOL_PATTERN = re.compile(r"\b(?:x\d+|q\d+|y)\b")
SCALAR_LHS_PATTERN = re.compile(r"^(?:y|x\d+)$")
VARIABLE_CALL_PATTERN = re.compile(r"\b(?:x\d+|q\d+|y)\s*\(")
RANGE_PATTERN = re.compile(r"^\s*([A-Za-z]\d+)\s*:\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]\s*$")
MAP_PATTERN = re.compile(r"^\s*([A-Za-z]\d+)\s*=\s*(.+?)\s*$")


@dataclass(frozen=True)
class ExpressionRecord:
    expression_id: int
    raw_formula: str
    variable_mapping: dict[str, str]
    variable_ranges: dict[str, tuple[float, float]]
    formula_name: str


@dataclass(frozen=True)
class ExpressionTask:
    expression_id: int
    formula_name: str
    raw_formula: str
    normalized_formula: str
    lhs_variable: str
    rhs_expression: str
    target_variable: str
    observed_feature_variables: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_column_mapping: dict[str, str]
    latent_variables: tuple[str, ...]
    ground_truth_latent_dim: int
    variable_mapping: dict[str, str]
    variable_ranges: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class GeneratedExpressionDataset:
    task: ExpressionTask
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    latent_truth_frame: pd.DataFrame
    ground_truth_latent_dim: int


def load_expression_library(csv_path: Path | str) -> list[ExpressionRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Expression library CSV does not exist: {path}")

    frame = pd.read_csv(path)
    if frame.shape[1] < 5:
        raise ValueError("Expression library CSV must have at least 5 columns.")

    records: list[ExpressionRecord] = []
    for row in frame.itertuples(index=False, name=None):
        expression_id = int(row[0])
        raw_formula = str(row[1]).strip()
        variable_mapping = _parse_variable_mapping(str(row[2]))
        variable_ranges = _parse_variable_ranges(str(row[3]))
        formula_name = str(row[4]).strip()
        records.append(
            ExpressionRecord(
                expression_id=expression_id,
                raw_formula=raw_formula,
                variable_mapping=variable_mapping,
                variable_ranges=variable_ranges,
                formula_name=formula_name,
            )
        )
    return records


def build_expression_task(record: ExpressionRecord) -> ExpressionTask:
    normalized_formula = _normalize_formula(record.raw_formula)
    _validate_formula_support(normalized_formula)

    lhs_expression, rhs_expression = _split_equation(normalized_formula)
    if not SCALAR_LHS_PATTERN.fullmatch(lhs_expression):
        raise ValueError(
            f"Expression {record.expression_id} has an unsupported left-hand side: {lhs_expression!r}. "
            "Only scalar forms like 'y = ...' or 'x1 = ...' are currently supported."
        )

    _validate_rhs_expression(rhs_expression)
    symbols = sorted(set(SYMBOL_PATTERN.findall(rhs_expression)), key=_sort_symbol_key)
    observed_feature_variables = tuple(symbol for symbol in symbols if symbol.startswith("x"))
    latent_variables = tuple(symbol for symbol in symbols if symbol.startswith("q"))

    if not latent_variables:
        raise ValueError(
            f"Expression {record.expression_id} does not contain any latent variable q* and is not suitable "
            "for the current latent-q workflow."
        )
    if not observed_feature_variables:
        raise ValueError(
            f"Expression {record.expression_id} does not expose any observable x* variable on the right-hand side."
        )

    required_ranges = [*observed_feature_variables, *latent_variables]
    missing_ranges = [symbol for symbol in required_ranges if symbol not in record.variable_ranges]
    if missing_ranges:
        raise ValueError(
            f"Expression {record.expression_id} is missing variable ranges for: {', '.join(missing_ranges)}."
        )

    feature_columns = observed_feature_variables
    feature_column_mapping = {feature_name: feature_name for feature_name in observed_feature_variables}
    target_variable = lhs_expression

    return ExpressionTask(
        expression_id=record.expression_id,
        formula_name=record.formula_name,
        raw_formula=record.raw_formula,
        normalized_formula=normalized_formula,
        lhs_variable=lhs_expression,
        rhs_expression=rhs_expression,
        target_variable=target_variable,
        observed_feature_variables=observed_feature_variables,
        feature_columns=feature_columns,
        feature_column_mapping=feature_column_mapping,
        latent_variables=latent_variables,
        ground_truth_latent_dim=len(latent_variables),
        variable_mapping=dict(record.variable_mapping),
        variable_ranges=dict(record.variable_ranges),
    )


def describe_expression_support(records: list[ExpressionRecord]) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for record in records:
        try:
            task = build_expression_task(record)
            descriptions.append(
                {
                    "expression_id": record.expression_id,
                    "formula_name": record.formula_name,
                    "status": "supported",
                    "reason": "",
                    "observed_feature_variables": list(task.observed_feature_variables),
                    "latent_variables": list(task.latent_variables),
                }
            )
        except ValueError as exc:
            descriptions.append(
                {
                    "expression_id": record.expression_id,
                    "formula_name": record.formula_name,
                    "status": "unsupported",
                    "reason": str(exc),
                    "observed_feature_variables": [],
                    "latent_variables": [],
                }
            )
    return descriptions


def select_expression_task(
    records: list[ExpressionRecord],
    *,
    expression_id: Optional[int] = None,
    formula_name: Optional[str] = None,
) -> ExpressionTask:
    if expression_id is None and formula_name is None:
        raise ValueError("Either expression_id or formula_name must be provided.")

    matched_record: Optional[ExpressionRecord] = None
    for record in records:
        if expression_id is not None and record.expression_id == expression_id:
            matched_record = record
            break
        if formula_name is not None and record.formula_name == formula_name:
            matched_record = record
            break

    if matched_record is None:
        identifier = f"id={expression_id}" if expression_id is not None else f"name={formula_name!r}"
        raise ValueError(f"No expression record matched {identifier}.")

    return build_expression_task(matched_record)


def sample_expression_dataset(
    task: ExpressionTask,
    *,
    label_count: int,
    train_samples_per_label: int,
    test_samples_per_label: int,
    noise_std: float = 0.0,
    seed: int = 42,
    max_attempts_per_row: int = 200,
) -> GeneratedExpressionDataset:
    if label_count <= 0:
        raise ValueError("label_count must be a positive integer.")
    if train_samples_per_label <= 0:
        raise ValueError("train_samples_per_label must be a positive integer.")
    if test_samples_per_label <= 0:
        raise ValueError("test_samples_per_label must be a positive integer.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if max_attempts_per_row <= 0:
        raise ValueError("max_attempts_per_row must be a positive integer.")

    rng = np.random.default_rng(seed)
    latent_truth_rows: list[dict[str, float]] = []
    train_rows: list[dict[str, float]] = []
    test_rows: list[dict[str, float]] = []

    for label in range(1, label_count + 1):
        latent_assignment = {
            latent_var: _sample_uniform(task.variable_ranges[latent_var], rng)
            for latent_var in task.latent_variables
        }
        latent_truth_rows.append({"label": label, **latent_assignment})
        train_rows.extend(
            _sample_split_rows(
                label=label,
                task=task,
                latent_assignment=latent_assignment,
                sample_count=train_samples_per_label,
                noise_std=noise_std,
                rng=rng,
                max_attempts_per_row=max_attempts_per_row,
            )
        )
        test_rows.extend(
            _sample_split_rows(
                label=label,
                task=task,
                latent_assignment=latent_assignment,
                sample_count=test_samples_per_label,
                noise_std=noise_std,
                rng=rng,
                max_attempts_per_row=max_attempts_per_row,
            )
        )

    train_frame = pd.DataFrame(train_rows, columns=["label", *task.feature_columns, "target"])
    test_frame = pd.DataFrame(test_rows, columns=["label", *task.feature_columns, "target"])
    latent_truth_frame = pd.DataFrame(latent_truth_rows, columns=["label", *task.latent_variables])

    return GeneratedExpressionDataset(
        task=task,
        train_frame=train_frame,
        test_frame=test_frame,
        latent_truth_frame=latent_truth_frame,
        ground_truth_latent_dim=task.ground_truth_latent_dim,
    )


def save_generated_expression_dataset(
    dataset: GeneratedExpressionDataset,
    output_dir: Path | str,
    *,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
    latent_truth_filename: str = "latent_truth.csv",
    metadata_filename: str = "expression_metadata.json",
    include_header: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / train_filename
    test_path = output_path / test_filename
    latent_truth_path = output_path / latent_truth_filename
    metadata_path = output_path / metadata_filename

    dataset.train_frame.to_csv(train_path, index=False, header=include_header)
    dataset.test_frame.to_csv(test_path, index=False, header=include_header)
    dataset.latent_truth_frame.to_csv(latent_truth_path, index=False)
    metadata_path.write_text(
        json.dumps(dataset_metadata(dataset), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "generated_train_csv": train_path,
        "generated_test_csv": test_path,
        "latent_truth_csv": latent_truth_path,
        "expression_metadata_json": metadata_path,
    }


def dataset_metadata(dataset: GeneratedExpressionDataset) -> dict[str, Any]:
    task = dataset.task
    return {
        "expression_id": task.expression_id,
        "formula_name": task.formula_name,
        "raw_formula": task.raw_formula,
        "normalized_formula": task.normalized_formula,
        "rhs_expression": task.rhs_expression,
        "target_variable": task.target_variable,
        "observed_feature_variables": list(task.observed_feature_variables),
        "feature_columns": list(task.feature_columns),
        "feature_column_mapping": dict(task.feature_column_mapping),
        "latent_variables": list(task.latent_variables),
        "ground_truth_latent_dim": int(task.ground_truth_latent_dim),
        "variable_mapping": dict(task.variable_mapping),
        "variable_ranges": {
            symbol: [bounds[0], bounds[1]] for symbol, bounds in task.variable_ranges.items()
        },
        "train_row_count": int(len(dataset.train_frame)),
        "test_row_count": int(len(dataset.test_frame)),
        "label_count": int(dataset.latent_truth_frame["label"].nunique()),
    }


def _sample_split_rows(
    *,
    label: int,
    task: ExpressionTask,
    latent_assignment: dict[str, float],
    sample_count: int,
    noise_std: float,
    rng: np.random.Generator,
    max_attempts_per_row: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _ in range(sample_count):
        row = _sample_single_row(
            label=label,
            task=task,
            latent_assignment=latent_assignment,
            noise_std=noise_std,
            rng=rng,
            max_attempts=max_attempts_per_row,
        )
        rows.append(row)
    return rows


def _sample_single_row(
    *,
    label: int,
    task: ExpressionTask,
    latent_assignment: dict[str, float],
    noise_std: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> dict[str, float]:
    for _ in range(max_attempts):
        observed_assignment = {
            observed_var: _sample_uniform(task.variable_ranges[observed_var], rng)
            for observed_var in task.observed_feature_variables
        }
        evaluation_context = {**latent_assignment, **observed_assignment}
        try:
            target_value = evaluate_scalar_expression(task.rhs_expression, evaluation_context)
        except Exception:
            continue
        if not np.isfinite(target_value):
            continue
        if noise_std > 0:
            target_value += float(rng.normal(loc=0.0, scale=noise_std))

        row: dict[str, float] = {"label": int(label)}
        for observed_var, feature_column in task.feature_column_mapping.items():
            row[feature_column] = float(observed_assignment[observed_var])
        row["target"] = float(target_value)
        return row

    raise RuntimeError(
        f"Failed to generate a valid sample for expression {task.expression_id} after {max_attempts} attempts. "
        "The variable ranges may be incompatible with the formula domain."
    )


def evaluate_scalar_expression(expression: str, values: dict[str, float]) -> float:
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.Name):
            if node.id in values:
                return float(values[node.id])
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
            raise ValueError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator.")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("Unsupported binary operator.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are supported.")
            if node.func.id not in SUPPORTED_FUNCTIONS:
                raise ValueError(f"Function {node.func.id!r} is not supported.")
            args = [_eval(argument) for argument in node.args]
            return float(SUPPORTED_FUNCTIONS[node.func.id](*args))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    return float(_eval(tree))


def _normalize_formula(raw_formula: str) -> str:
    normalized = raw_formula.strip()
    normalized = normalized.replace("·", "*")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("×", "*")
    normalized = normalized.replace("÷", "/")
    normalized = normalized.split("；", maxsplit=1)[0].strip()
    normalized = normalized.split(";", maxsplit=1)[0].strip()
    normalized = normalized.split("（", maxsplit=1)[0].strip()
    normalized = normalized.replace(" ", "")
    return normalized


def _validate_formula_support(normalized_formula: str) -> None:
    if "且" in normalized_formula:
        raise ValueError("Multi-stage formulas containing '且' are not yet supported.")
    if "Re[" in normalized_formula or "Im[" in normalized_formula:
        raise ValueError("Complex-valued expressions such as Re[...] are not yet supported.")
    if VARIABLE_CALL_PATTERN.search(normalized_formula):
        raise ValueError("Function-valued variables like q1(x2) are not yet supported.")


def _split_equation(normalized_formula: str) -> tuple[str, str]:
    parts = normalized_formula.split("=")
    if len(parts) != 2:
        raise ValueError(
            "Only single-equation formulas are currently supported. "
            f"Received: {normalized_formula!r}"
        )
    lhs_expression = parts[0].strip()
    rhs_expression = parts[1].strip()
    return lhs_expression, rhs_expression


def _validate_rhs_expression(rhs_expression: str) -> None:
    tree = ast.parse(rhs_expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.FloorDiv,
        ast.UAdd,
        ast.USub,
        ast.Num,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(
                f"The formula contains an unsupported expression construct: {type(node).__name__}."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct scalar function calls are supported.")
            if node.func.id not in SUPPORTED_FUNCTIONS:
                raise ValueError(f"Unsupported function in formula: {node.func.id!r}")


def _parse_variable_mapping(raw_mapping: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in raw_mapping.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = MAP_PATTERN.match(stripped)
        if not match:
            continue
        mapping[match.group(1)] = match.group(2).strip()
    return mapping


def _parse_variable_ranges(raw_ranges: str) -> dict[str, tuple[float, float]]:
    parsed_ranges: dict[str, tuple[float, float]] = {}
    for line in raw_ranges.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = RANGE_PATTERN.match(stripped)
        if not match:
            continue
        lower_bound = float(match.group(2))
        upper_bound = float(match.group(3))
        if lower_bound > upper_bound:
            raise ValueError(f"Invalid variable range with lower bound > upper bound: {stripped!r}")
        parsed_ranges[match.group(1)] = (lower_bound, upper_bound)
    return parsed_ranges


def _sample_uniform(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    lower_bound, upper_bound = bounds
    if lower_bound == upper_bound:
        return float(lower_bound)
    return float(rng.uniform(lower_bound, upper_bound))


def _sort_symbol_key(symbol: str) -> tuple[str, int]:
    return symbol[0], int(symbol[1:])


__all__ = [
    "ExpressionRecord",
    "ExpressionTask",
    "GeneratedExpressionDataset",
    "build_expression_task",
    "dataset_metadata",
    "describe_expression_support",
    "evaluate_scalar_expression",
    "load_expression_library",
    "sample_expression_dataset",
    "save_generated_expression_dataset",
    "select_expression_task",
]
