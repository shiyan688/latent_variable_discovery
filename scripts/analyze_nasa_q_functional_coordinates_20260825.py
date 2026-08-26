#!/usr/bin/env python3
"""Convert a trained NASA latent q into named decoder-response coordinates."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

from lvs.backends.torch_mlp import build_torch_model_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _slope(frame: pd.DataFrame, low: float, high: float) -> float:
    selected = frame.loc[frame.discharge_index.between(low, high)]
    return float(np.polyfit(selected.discharge_index, selected.target, 1)[0])


def main() -> None:
    args = parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        result["artifacts"]["training_checkpoint"], map_location="cpu", weights_only=False
    )
    feature_columns = checkpoint["feature_columns"]
    expected_features = [
        "discharge_index",
        "ambient_temperature",
        "load_current_amp",
        "cutoff_voltage",
    ]
    if feature_columns != expected_features:
        raise ValueError(f"Expected NASA clean features {expected_features}, got {feature_columns}")

    q_dim = int(result["job"]["q_dim"])
    model = build_torch_model_factory(tuple(checkpoint["hidden_sizes"]))(
        len(feature_columns) + q_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = checkpoint["normalizer"]
    feature_mean = np.asarray(normalizer["feature_mean"], dtype=float)
    feature_std = np.asarray(normalizer["feature_std"], dtype=float)
    target_mean = float(normalizer["target_mean"])
    target_std = float(normalizer["target_std"])

    q_frames = []
    for split, key in (("train", "train_label_q"), ("outer_test", "test_label_q")):
        q_frame = pd.read_csv(result["artifacts"][key])
        q_columns = [column for column in q_frame if re.fullmatch(r"q\d+", column)]
        q_frames.append(q_frame.loc[:, ["label", *q_columns]].assign(split=split))
    q_frame = pd.concat(q_frames, ignore_index=True)
    q_columns = [column for column in q_frame if re.fullmatch(r"q\d+", column)]

    cycles = np.asarray((1.0, 10.0, 20.0, 28.0))
    reference_conditions = np.column_stack(
        [
            cycles,
            np.full_like(cycles, 24.0),
            np.full_like(cycles, 2.0),
            np.full_like(cycles, 2.5),
        ]
    )
    normalized_conditions = (reference_conditions - feature_mean) / feature_std
    probe_rows = []
    coordinate_rows = []
    with torch.no_grad():
        for row in q_frame.itertuples(index=False):
            q = np.asarray([getattr(row, column) for column in q_columns], dtype=float)
            inputs = np.column_stack(
                [normalized_conditions, np.repeat(q[None, :], len(cycles), axis=0)]
            )
            prediction = model(torch.tensor(inputs, dtype=torch.float32)).squeeze(1).numpy()
            prediction = target_mean + target_std * prediction
            for cycle, value in zip(cycles, prediction):
                probe_rows.append(
                    {
                        "label": row.label,
                        "split": row.split,
                        "discharge_index": cycle,
                        "decoder_capacity": float(value),
                    }
                )
            early_fade = float((prediction[0] - prediction[1]) / 9.0)
            mid_fade = float((prediction[1] - prediction[3]) / 18.0)
            coordinate_rows.append(
                {
                    "label": row.label,
                    "split": row.split,
                    "capacity_cycle1": float(prediction[0]),
                    "capacity_cycle28": float(prediction[3]),
                    "early_fade_rate": early_fade,
                    "mid_fade_rate": mid_fade,
                    "fade_acceleration": mid_fade - early_fade,
                }
            )

    prepared_summary = Path(result["job"]["prepared_summary"])
    if not prepared_summary.is_absolute():
        prepared_summary = Path.cwd() / prepared_summary
    record = [
        item
        for item in json.loads(prepared_summary.read_text(encoding="utf-8"))
        if item["name"] == result["job"]["dataset"]
    ][0]
    train = pd.read_csv(record["train_csv"])
    descriptor_rows = []
    for label, frame in train.groupby("label", sort=False):
        frame = frame.sort_values("discharge_index")
        early_slope = _slope(frame, 1.0, 10.0)
        mid_slope = _slope(frame, 10.0, 28.0)
        early_fit = np.polyfit(
            frame.loc[frame.discharge_index <= 10, "discharge_index"],
            frame.loc[frame.discharge_index <= 10, "target"],
            1,
        )
        descriptor_rows.append(
            {
                "label": label,
                "empirical_capacity_cycle1": float(np.polyval(early_fit, 1.0)),
                "empirical_early_fade_rate": -early_slope,
                "empirical_mid_fade_rate": -mid_slope,
                "empirical_fade_acceleration": -mid_slope + early_slope,
            }
        )
    descriptors = pd.DataFrame(descriptor_rows)
    coordinates = pd.DataFrame(coordinate_rows)
    train_analysis = coordinates.loc[coordinates.split == "train"].merge(
        descriptors, on="label", validate="one_to_one"
    )
    functional_columns = [
        "capacity_cycle1",
        "capacity_cycle28",
        "early_fade_rate",
        "mid_fade_rate",
        "fade_acceleration",
    ]
    empirical_columns = [column for column in descriptors if column != "label"]
    correlations = []
    for functional in functional_columns:
        for empirical in empirical_columns:
            correlation, p_value = spearmanr(train_analysis[functional], train_analysis[empirical])
            correlations.append(
                {
                    "functional_coordinate": functional,
                    "empirical_descriptor": empirical,
                    "spearman": float(correlation),
                    "p_value": float(p_value),
                    "train_entities": int(len(train_analysis)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probe_rows).to_csv(args.output_dir / "decoder_probes.csv", index=False)
    coordinates.to_csv(args.output_dir / "functional_coordinates.csv", index=False)
    train_analysis.to_csv(args.output_dir / "train_coordinate_descriptors.csv", index=False)
    pd.DataFrame(correlations).to_csv(args.output_dir / "train_descriptor_correlations.csv", index=False)
    metadata = {
        "source_result": str(args.result_json.resolve()),
        "coordinate_definition": "frozen decoder at common clean-condition grid",
        "reference_conditions": {
            "cycles": cycles.tolist(),
            "ambient_temperature": 24.0,
            "load_current_amp": 2.0,
            "cutoff_voltage": 2.5,
        },
        "outer_targets_used": False,
        "functional_columns": functional_columns,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
