#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate latent variable expression library in place.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "latent_variable_expressions.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backup", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output) if args.output else input_path
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    frame = apply_overrides(frame)
    frame = drop_redundant_records(frame)
    frame = normalize_x_ranges(frame)
    frame = remove_duplicate_math_forms(frame)
    frame = append_real_world_expressions(frame)
    frame = normalize_x_ranges(frame)
    frame = remove_duplicate_math_forms(frame)
    frame = frame.sort_values("ID").reset_index(drop=True)

    if args.backup and output_path == input_path:
        backup_path = input_path.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.copy2(input_path, backup_path)
        print(f"backup: {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"wrote: {output_path}")
    print(f"rows: {len(frame)}")


def apply_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    overrides = {
        15: {
            "x/q公式": "y = q1·exp(-q2/(8.31446·(x1 + 273.15)))",
            "变量映射": "x1=T_C\nq1=A\nq2=Ea",
            "变量范围": "x1: [0, 10]\nq1: [0, 1.00e+8]\nq2: [1000, 1.00e+5]",
            "公式名字": "Arrhenius 反应速率常数（摄氏输入）",
        },
        6: {
            "x/q公式": "y = q1·x1",
            "变量映射": "x1=ε_strain\nq1=E_eff",
            "变量范围": "x1: [-10, 10]\nq1: [1, 1.00e+5]",
            "公式名字": "广义胡克定律（等效单轴形式）",
        },
        10: {
            "x/q公式": "y = q1 + q2/(x1 + 1)^2",
            "变量映射": "x1=λ_scaled\nq1=A\nq2=B",
            "变量范围": "x1: [0, 10]\nq1: [0.5, 5]\nq2: [-10, 10]",
            "公式名字": "Cauchy 折射率色散（简化式）",
        },
        19: {
            "x/q公式": "y = x1·x2·q1/x3",
            "变量映射": "x1=E\nx2=ε\nx3=η\nq1=ζ",
            "变量范围": "x1: [-10, 10]\nx2: [0, 10]\nx3: [0, 10]\nq1: [-1, 1]",
            "公式名字": "电泳迁移率（ζ 电位，单式化）",
        },
        20: {
            "x/q公式": "y = 6.28319·x1^3·x2·((q1 - x5)/(q1 + 2·x5))·x4",
            "变量映射": "x1=a\nx2=ε_m\nx4=gradE2\nx5=ε_m_ref\nq1=ε_p",
            "变量范围": "x1: [0, 10]\nx2: [0, 10]\nx4: [0, 10]\nx5: [0, 10]\nq1: [0.1, 100]",
            "公式名字": "介电泳力（DEP，Clausius-Mossotti 单式化）",
        },
        32: {
            "x/q公式": "y = 12.56637·q1^2/(1 + (q1·x1)^2)",
            "变量映射": "x1=k\nq1=a_s",
            "变量范围": "x1: [0, 10]\nq1: [-10, 10]",
            "公式名字": "s 波散射截面（含波数修正）",
        },
        33: {
            "x/q公式": "y = 3.87405e-5·q1·x1/(x1 + q2)",
            "变量映射": "x1=B_scaled\nq1=C\nq2=Γ",
            "变量范围": "x1: [0, 10]\nq1: [-10, 10]\nq2: [0.1, 10]",
            "公式名字": "量子霍尔电导（平台展宽简化式）",
        },
        46: {
            "x/q公式": "y = q1·q3·x1/((q2 - x1)·(1 + (q3 - 1)·x1/q2))",
            "变量映射": "x1=P_rel_scaled\nq1=q_m\nq2=P0_scaled\nq3=C_BET",
            "变量范围": "x1: [0, 10]\nq1: [0, 100]\nq2: [11, 100]\nq3: [1, 100]",
            "公式名字": "BET 多层吸附等温式",
        },
    }
    frame = frame.copy()
    for expression_id, updates in overrides.items():
        mask = frame["ID"].astype(int).eq(expression_id)
        for column, value in updates.items():
            frame.loc[mask, column] = value
    return frame


def normalize_x_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    range_col = "变量范围"
    frame = frame.copy()
    denominator_by_formula = {
        str(row["x/q公式"]): denominator_x_variables(str(row["x/q公式"]))
        for _, row in frame.iterrows()
    }
    frame[range_col] = [
        normalize_range_text(raw, denominator_by_formula.get(str(formula), set()))
        for raw, formula in zip(frame[range_col], frame["x/q公式"])
    ]
    return frame


def normalize_range_text(raw: str, denominator_xs: set[str] | None = None) -> str:
    denominator_xs = denominator_xs or set()
    lines: list[str] = []
    for line in str(raw).splitlines():
        if ":" not in line or "[" not in line or "]" not in line:
            lines.append(line)
            continue
        symbol = line.split(":", 1)[0].strip()
        if not symbol.lower().startswith("x"):
            lines.append(line)
            continue
        bounds = line.split("[", 1)[1].split("]", 1)[0]
        try:
            low_s, high_s = bounds.split(",", 1)
            low = float(low_s)
            high = float(high_s)
        except Exception:
            lines.append(f"{symbol}: [-10, 10]")
            continue
        if symbol in denominator_xs:
            new_low, new_high = denominator_safe_range(low, high)
        elif high <= 0:
            new_low, new_high = -10, 0
        elif low < 0 < high:
            new_low, new_high = -10, 10
        else:
            new_low, new_high = 0, 10
        lines.append(f"{symbol}: [{new_low}, {new_high}]")
    return "\n".join(lines)


def denominator_safe_range(low: float, high: float) -> tuple[int, int]:
    if high <= 0:
        return -10, -1
    if low < 0 < high:
        # The current sampler supports one continuous interval, not a union like [-10, -1] U [1, 10].
        # Use the positive branch by default so denominator magnitudes stay away from zero.
        return 1, 10
    return 1, 10


def denominator_x_variables(formula: str) -> set[str]:
    normalized = normalize_formula_for_ast(formula)
    if "=" in normalized:
        normalized = normalized.split("=", 1)[1]
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return set()
    collector = DenominatorVariableCollector()
    collector.visit(tree)
    return collector.variables


class DenominatorVariableCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.variables: set[str] = set()

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            self.variables.update(x_variables_in_ast(node.right))
        if isinstance(node.op, ast.Pow) and is_negative_numeric_node(node.right):
            self.variables.update(x_variables_in_ast(node.left))
        self.generic_visit(node)


def x_variables_in_ast(node: ast.AST) -> set[str]:
    variables: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id.startswith("x"):
            variables.add(child.id)
    return variables


def is_negative_numeric_node(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) < 0
    return False


def normalize_formula_for_ast(formula: str) -> str:
    return (
        str(formula)
        .replace(" ", "")
        .replace("·", "*")
        .replace("^", "**")
        .replace("（", "(")
        .replace("）", ")")
    )


def remove_duplicate_math_forms(frame: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    keep_rows = []
    for row in frame.sort_values("ID").to_dict("records"):
        key = math_key(str(row["x/q公式"]))
        if key in seen:
            continue
        seen.add(key)
        keep_rows.append(row)
    return pd.DataFrame(keep_rows, columns=frame.columns)


def drop_redundant_records(frame: pd.DataFrame) -> pd.DataFrame:
    redundant_math_keys = {
        math_key("y = q1·q2·x1/(1 + q2·x1)"),
    }
    mask = frame["x/q公式"].map(lambda value: math_key(str(value))).isin(redundant_math_keys)
    mask &= frame["公式名字"].astype(str).eq("Langmuir 吸附等温式")
    return frame.loc[~mask].copy()


def math_key(formula: str) -> str:
    return (
        formula.replace(" ", "")
        .replace("·", "*")
        .replace("^", "**")
        .replace("（", "(")
        .replace("）", ")")
        .lower()
    )


def append_real_world_expressions(frame: pd.DataFrame) -> pd.DataFrame:
    next_id = int(frame["ID"].max()) + 1
    additions = [
        (
            "y = q1·exp(-q2/(8.31446·(x1 + 273.15)))",
            "x1=T_C\nq1=A\nq2=Ea",
            "x1: [0, 10]\nq1: [0, 1.00e+8]\nq2: [1000, 1.00e+5]",
            "Arrhenius 反应速率常数",
        ),
        (
            "y = q1·x1/(q2 + x1)",
            "x1=S\nq1=Vmax\nq2=Km",
            "x1: [0, 10]\nq1: [0, 100]\nq2: [0.1, 10]",
            "Michaelis-Menten 酶动力学",
        ),
        (
            "y = q1·x1^q3/(q2^q3 + x1^q3)",
            "x1=S\nq1=Vmax\nq2=K_half\nq3=n",
            "x1: [0, 10]\nq1: [0, 100]\nq2: [0.1, 10]\nq3: [0.5, 4]",
            "Hill 饱和响应模型",
        ),
        (
            "y = q1·x1^(1/q2)",
            "x1=C\nq1=K_F\nq2=n",
            "x1: [0, 10]\nq1: [0, 100]\nq2: [1, 10]",
            "Freundlich 吸附等温式",
        ),
        (
            "y = x2 + (q1 - x2)·exp(-q2·x1)",
            "x1=t\nx2=T_env\nq1=T0\nq2=k",
            "x1: [0, 10]\nx2: [0, 10]\nq1: [-10, 100]\nq2: [0.001, 10]",
            "牛顿冷却定律",
        ),
        (
            "y = q1/(1 + q2·exp(-q3·x1))",
            "x1=t\nq1=K\nq2=A\nq3=r",
            "x1: [0, 10]\nq1: [0, 100]\nq2: [0.001, 100]\nq3: [0.001, 10]",
            "Logistic 生长曲线",
        ),
        (
            "y = q1·exp(-q2·exp(-q3·x1))",
            "x1=t\nq1=A\nq2=B\nq3=C",
            "x1: [0, 10]\nq1: [0, 100]\nq2: [0.001, 100]\nq3: [0.001, 10]",
            "Gompertz 生长曲线",
        ),
        (
            "y = q1 - q2·sqrt(x1) - q3·x1",
            "x1=N_cycle\nq1=Q0\nq2=k_sqrt\nq3=k_lin",
            "x1: [0, 10]\nq1: [0, 10]\nq2: [0, 10]\nq3: [0, 10]",
            "电池容量衰减（平方根+线性项）",
        ),
        (
            "y = q1·x1^q2",
            "x1=ΔK\nq1=C\nq2=m",
            "x1: [0, 10]\nq1: [1.00e-12, 1]\nq2: [1, 6]",
            "Paris 裂纹扩展定律",
        ),
        (
            "y = q1·x1^q2",
            "x1=shear_rate\nq1=K\nq2=n",
            "x1: [0, 10]\nq1: [0.001, 100]\nq2: [0.1, 2]",
            "幂律流体本构关系",
        ),
        (
            "y = q1 + q2·log(x1 + 1)",
            "x1=current_density\nq1=a\nq2=b",
            "x1: [0, 10]\nq1: [-10, 10]\nq2: [-10, 10]",
            "Tafel 极化关系（对数形式）",
        ),
        (
            "y = q1 + 0.025693·x2·log(x1 + 1)",
            "x1=activity_ratio\nx2=charge_factor\nq1=E0",
            "x1: [0, 10]\nx2: [0, 10]\nq1: [-10, 10]",
            "Nernst 电极电势（简化式）",
        ),
        (
            "y = q1·(x1 + 273.15)^1.5/(x1 + q2)",
            "x1=T_C\nq1=C\nq2=S",
            "x1: [0, 10]\nq1: [1.00e-8, 1]\nq2: [20, 1000]",
            "Sutherland 气体黏度公式",
        ),
        (
            "y = q1/(1 + (x1·q2)^2)",
            "x1=frequency\nq1=sigma0\nq2=tau",
            "x1: [0, 10]\nq1: [0, 1.00e+8]\nq2: [1.00e-6, 10]",
            "Drude 交流电导实部（归一化频率）",
        ),
    ]
    rows = []
    for offset, (formula, mapping, ranges, name) in enumerate(additions):
        rows.append(
            {
                "ID": next_id + offset,
                "x/q公式": formula,
                "变量映射": mapping,
                "变量范围": ranges,
                "公式名字": name,
            }
        )
    return pd.concat([frame, pd.DataFrame(rows, columns=frame.columns)], ignore_index=True)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
