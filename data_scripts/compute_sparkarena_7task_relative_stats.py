#!/usr/bin/env python3
"""Compute and validate Spark joint54 relative-arm action statistics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = Path(
    "/personal/xiangpc/training_0908-real/rldx_0908-real/"
    "code_0908-real/data/spark0_bench_7task_0908"
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "policy/RLDX_1/spark_joint54_relative_config.py"
)
GROUPS = ("left_arm", "left_hand", "right_arm", "right_hand")
ARM_GROUPS = ("left_arm", "right_arm")
EXPECTED_SLICES = {
    "left_arm": (0, 7),
    "left_hand": (7, 27),
    "right_arm": (27, 34),
    "right_hand": (34, 54),
}
STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")
EXPECTED_REPRESENTATIONS = ("relative", "absolute", "relative", "absolute")


def _load_relative_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("spark_joint54_relative_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.SPARK_JOINT54_RELATIVE_MODALITY_CONFIG
    action = config["action"]
    representations = tuple(item.rep.value for item in action.action_configs)
    if tuple(action.modality_keys) != GROUPS:
        raise RuntimeError(f"unexpected action groups: {action.modality_keys}")
    if representations != EXPECTED_REPRESENTATIONS:
        raise RuntimeError(f"unexpected action representations: {representations}")
    if list(action.delta_indices) != list(range(16)):
        raise RuntimeError(f"unexpected action horizon: {action.delta_indices}")


def _validate_dataset(dataset: Path) -> dict[str, Any]:
    info_path = dataset / "meta/info.json"
    modality_path = dataset / "meta/modality.json"
    if not info_path.is_file() or not modality_path.is_file():
        raise RuntimeError(f"dataset metadata is incomplete under {dataset}")
    info = json.loads(info_path.read_text())
    modality = json.loads(modality_path.read_text())
    for feature in ("observation.state", "action"):
        shape = info.get("features", {}).get(feature, {}).get("shape")
        if shape != [54]:
            raise RuntimeError(f"{feature} must have shape [54], got {shape}")
    for kind in ("state", "action"):
        if tuple(modality.get(kind, {})) != GROUPS:
            raise RuntimeError(f"unexpected {kind} group order: {tuple(modality.get(kind, {}))}")
        for key, (start, end) in EXPECTED_SLICES.items():
            entry = modality[kind][key]
            if (entry.get("start"), entry.get("end")) != (start, end):
                raise RuntimeError(f"unexpected {kind}.{key} slice: {entry}")
            if entry.get("absolute") is not True:
                raise RuntimeError(
                    f"stored {kind}.{key} must remain absolute; conversion happens in RLDX"
                )
    return info


def _jsonable_stats(stats: dict[str, Any]) -> dict[str, list]:
    return {field: np.asarray(stats[field], dtype=np.float64).tolist() for field in STAT_FIELDS}


def _validate_stats(stats: Any) -> None:
    if not isinstance(stats, dict) or set(stats) != set(ARM_GROUPS):
        actual = stats.keys() if isinstance(stats, dict) else type(stats)
        raise RuntimeError(
            f"relative stats must contain exactly {ARM_GROUPS}, got {actual}"
        )
    for arm in ARM_GROUPS:
        values = stats[arm]
        if set(values) != set(STAT_FIELDS):
            raise RuntimeError(f"{arm} stats have unexpected fields: {values.keys()}")
        arrays = {field: np.asarray(values[field], dtype=np.float64) for field in STAT_FIELDS}
        for field, array in arrays.items():
            if array.shape != (16, 7):
                raise RuntimeError(f"{arm}.{field} must have shape (16, 7), got {array.shape}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"{arm}.{field} contains non-finite values")
        if np.any(arrays["std"] < 0):
            raise RuntimeError(f"{arm}.std contains negative values")
        if np.any(arrays["min"] > arrays["max"]):
            raise RuntimeError(f"{arm} min exceeds max")
        if np.any(arrays["q01"] > arrays["q99"]):
            raise RuntimeError(f"{arm} q01 exceeds q99")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("RLDX_DATASET", DEFAULT_DATASET)),
    )
    parser.add_argument("--modality-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    config_path = args.modality_config.resolve()
    _load_relative_config(config_path)
    info = _validate_dataset(dataset)
    stats_path = dataset / "meta/relative_stats.json"

    if stats_path.is_file() and not args.force:
        try:
            existing = json.loads(stats_path.read_text())
            _validate_stats(existing)
        except (json.JSONDecodeError, RuntimeError, TypeError, ValueError):
            pass
        else:
            print(
                f"READY {stats_path} episodes={info.get('total_episodes')} "
                f"frames={info.get('total_frames')} tasks={info.get('total_tasks')}"
            )
            return

    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.data.stats import calculate_stats_for_key

    stats = {
        arm: _jsonable_stats(
            calculate_stats_for_key(dataset, EmbodimentTag.GENERAL_EMBODIMENT, arm)
        )
        for arm in ARM_GROUPS
    }
    _validate_stats(stats)

    temporary = stats_path.with_name(f".{stats_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, stats_path)
    print(
        f"WROTE {stats_path} shapes=left_arm:(16,7),right_arm:(16,7) "
        f"episodes={info.get('total_episodes')} frames={info.get('total_frames')}"
    )


if __name__ == "__main__":
    main()
