#!/usr/bin/env python3
"""Compute exact absolute and relative RLDX statistics for EgoVLA joint38."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _assert_finite_shape(value: object, shape: tuple[int, ...], label: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label}: shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("modality_config", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    modality_config = args.modality_config.resolve()
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"missing dataset info: {dataset}")
    if not modality_config.is_file():
        raise FileNotFoundError(modality_config)

    # Registration must happen before importing rldx.data.stats, which binds
    # the global modality registry used for relative-action statistics.
    from rldx.experiment.utils import load_modality_config

    load_modality_config(str(modality_config))

    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.data.stats import generate_rel_stats, generate_stats

    generate_stats(dataset)
    generate_rel_stats(dataset, EmbodimentTag.GENERAL_EMBODIMENT)

    stats = json.loads((dataset / "meta" / "stats.json").read_text())
    for feature in ("observation.state", "action"):
        if feature not in stats:
            raise ValueError(f"stats.json is missing {feature}")
        for field in ("mean", "std", "min", "max", "q01", "q99"):
            _assert_finite_shape(stats[feature][field], (38,), f"{feature}.{field}")

    relative = json.loads((dataset / "meta" / "relative_stats.json").read_text())
    if set(relative) != {"left_arm", "right_arm"}:
        raise ValueError(f"unexpected relative stats keys: {sorted(relative)}")
    for group in ("left_arm", "right_arm"):
        for field in ("mean", "std", "min", "max", "q01", "q99"):
            _assert_finite_shape(relative[group][field], (16, 7), f"{group}.{field}")

    print(
        f"DONE stats dataset={dataset} action_dim=38 horizon=16 "
        "relative=left_arm,right_arm absolute=left_hand,right_hand"
    )


if __name__ == "__main__":
    main()
