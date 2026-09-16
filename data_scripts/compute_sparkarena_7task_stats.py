#!/usr/bin/env python3
"""Compute LeRobot stats.json for the existing SparkArena 7-task v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    os.environ.get("RLDX_DATASET_PATH", REPO_ROOT / "data" / "spark0_bench_7task_0908")
)
FEATURES = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    dataset = parse_args().dataset.resolve()
    paths = sorted((dataset / "data" / "chunk-000").glob("episode_*.parquet"))
    if len(paths) != 700:
        raise RuntimeError(f"expected 700 chunk-000 parquets, got {len(paths)}")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in FEATURES}
    frames = 0
    for path in paths:
        table = pq.read_table(path, columns=list(FEATURES))
        frames += table.num_rows
        for key in FEATURES:
            parts[key].append(np.vstack([np.asarray(x, dtype=np.float64) for x in table[key].to_pylist()]))
    if frames != 154251:
        raise RuntimeError(f"expected 154251 frames, got {frames}")
    stats = {}
    for key, arrays in parts.items():
        value = np.concatenate(arrays, axis=0)
        if value.ndim == 1:
            value = value[:, None]
        stats[key] = {
            "mean": value.mean(axis=0).tolist(),
            "std": value.std(axis=0).tolist(),
            "min": value.min(axis=0).tolist(),
            "max": value.max(axis=0).tolist(),
            "q01": np.quantile(value, 0.01, axis=0).tolist(),
            "q99": np.quantile(value, 0.99, axis=0).tolist(),
        }
        print(key, "shape", value.shape, flush=True)
    out = dataset / "meta" / "stats.json"
    out.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
