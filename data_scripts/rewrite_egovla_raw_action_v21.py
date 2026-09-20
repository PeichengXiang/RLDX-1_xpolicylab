#!/usr/bin/env python3
"""Rewrite an EgoVLA LeRobot v2.1 clone to use the recorded HDF5 action.

The legacy RLDX dataset stored ``state[t + 1]`` in ``action`` while retaining
the true commanded action in ``provenance.raw_commanded_action``.  This script
atomically replaces only the parquet action column and updates prompt and
provenance metadata.  Video files are intentionally left untouched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egovla_authority import load_schema


EXPECTED_EPISODES = 1903
EXPECTED_FRAMES = 510_546
RAW_ACTION_KEY = "provenance.raw_commanded_action"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _column_numpy(table: pa.Table, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values).reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return np.stack(array.to_numpy(zero_copy_only=False))
    return array.to_numpy(zero_copy_only=False)


def _rewrite_parquet(path: Path) -> tuple[int, int, float]:
    table = pq.read_table(path)
    for key in ("observation.state", "action", RAW_ACTION_KEY):
        if key not in table.column_names:
            raise ValueError(f"{path}: missing {key}")

    old_action = _column_numpy(table, "action").astype(np.float32, copy=False)
    raw_action = _column_numpy(table, RAW_ACTION_KEY).astype(np.float32, copy=False)
    if old_action.shape != raw_action.shape or raw_action.ndim != 2 or raw_action.shape[1] != 38:
        raise ValueError(
            f"{path}: action/raw-action shapes are {old_action.shape}/{raw_action.shape}"
        )
    if not np.isfinite(raw_action).all():
        raise ValueError(f"{path}: raw commanded action contains NaN/Inf")

    differing_rows = int(np.count_nonzero(np.any(old_action != raw_action, axis=1)))
    mae = float(np.abs(old_action.astype(np.float64) - raw_action).mean())
    action_index = table.schema.get_field_index("action")
    action_field = table.schema.field(action_index)
    raw_column = table[RAW_ACTION_KEY]
    if raw_column.type != action_field.type:
        raw_column = raw_column.cast(action_field.type)
    rewritten = table.set_column(action_index, action_field, raw_column)

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    pq.write_table(rewritten, temporary, compression="zstd", use_dictionary=True)
    verified = pq.read_table(temporary, columns=["action", RAW_ACTION_KEY])
    if not np.array_equal(_column_numpy(verified, "action"), _column_numpy(verified, RAW_ACTION_KEY)):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{path}: atomic rewrite verification failed")
    os.replace(temporary, path)
    return table.num_rows, differing_rows, mae


def _rewrite_prompts(meta: Path, manifest: dict, task_prompts: dict[str, str]) -> None:
    tasks_path = meta / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    if len(tasks) != len(task_prompts):
        raise ValueError(f"expected {len(task_prompts)} task rows, found {len(tasks)}")
    old_prompt_to_task = {str(prompt): str(task) for task, prompt in manifest["instructions"].items()}
    task_index_to_name: dict[int, str] = {}
    for row in tasks:
        old_prompt = str(row["task"])
        task_name = old_prompt_to_task.get(old_prompt)
        if task_name is None:
            raise ValueError(f"cannot map existing prompt to task: {old_prompt!r}")
        row["task"] = task_prompts[task_name]
        task_index = int(row["task_index"])
        if task_index in task_index_to_name:
            raise ValueError(f"duplicate task_index: {task_index}")
        task_index_to_name[task_index] = task_name
    if set(task_index_to_name) != set(range(len(task_prompts))):
        raise ValueError("task indices must be unique and contiguous from 0 through 11")
    if set(task_index_to_name.values()) != set(task_prompts):
        raise ValueError("task metadata does not cover the canonical 12-task registry")
    _write_jsonl_atomic(tasks_path, tasks)

    episodes_path = meta / "episodes.jsonl"
    episodes = _read_jsonl(episodes_path)
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"unexpected episode count: {len(episodes)}")
    for row in episodes:
        task_indices = row.get("tasks")
        if not isinstance(task_indices, list) or len(task_indices) != 1:
            raise ValueError(f"episode has invalid task list: {row}")
        old_prompt = str(task_indices[0])
        task_name = old_prompt_to_task.get(old_prompt)
        if task_name is None:
            raise ValueError(f"cannot map episode prompt: {old_prompt!r}")
        row["tasks"] = [task_prompts[task_name]]
    _write_jsonl_atomic(episodes_path, episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    published_root = dataset
    if dataset.name.startswith(".") and ".incomplete-" in dataset.name:
        published_name = dataset.name[1:].split(".incomplete-", 1)[0]
        published_root = dataset.parent / published_name
    raw_root = args.raw_root.resolve()
    schema_path = args.schema.resolve()
    task_prompts, policy_joint_indices, schema_sha256 = load_schema(schema_path)
    meta = dataset / "meta"
    info = _read_json(meta / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("input clone is not LeRobot v2.1")
    if info.get("total_episodes") != EXPECTED_EPISODES or info.get("total_frames") != EXPECTED_FRAMES:
        raise ValueError("input clone does not match the complete EgoVLA benchmark")
    if info["features"]["action"]["shape"] != [38] or info["features"][RAW_ACTION_KEY]["shape"] != [38]:
        raise ValueError("input clone does not contain joint38 action provenance")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    source_manifest_path = meta / "xpolicylab_source_conversion.json"
    manifest = _read_json(source_manifest_path)
    legacy_manifest_hash = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    if len(manifest.get("episodes", [])) != EXPECTED_EPISODES:
        raise ValueError("source conversion manifest is incomplete")
    instructions = manifest.get("instructions")
    if not isinstance(instructions, dict) or set(instructions) != set(task_prompts):
        raise ValueError("legacy manifest tasks do not match the benchmark schema")
    if len(set(map(str, instructions.values()))) != len(task_prompts):
        raise ValueError("legacy manifest instructions are not unique")

    parquet_paths = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_paths) != EXPECTED_EPISODES:
        raise ValueError(f"found {len(parquet_paths)} parquet files, expected {EXPECTED_EPISODES}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_rewrite_parquet, parquet_paths))
    total_rows = sum(item[0] for item in results)
    differing_rows = sum(item[1] for item in results)
    weighted_mae = sum(rows * mae for (rows, _different, mae) in results) / total_rows
    if total_rows != EXPECTED_FRAMES:
        raise ValueError(f"rewritten frame count {total_rows}, expected {EXPECTED_FRAMES}")
    if differing_rows == 0:
        raise ValueError("legacy action already equals raw action everywhere; refusing ambiguous rewrite")

    _rewrite_prompts(meta, manifest, task_prompts)
    manifest.update(
        {
            "converter": Path(__file__).name,
            "converter_version": "1.0.0",
            "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_root": str(raw_root),
            "output_root": str(published_root),
            "training_action_source": (
                "raw HDF5 /action indexed from native 50D to policy joint38; "
                "never state[t+1]"
            ),
            "raw_commanded_action_feature": RAW_ACTION_KEY,
            "prompt_source": "EgoVLA benchmark task registry (exact strings)",
            "benchmark_schema_path": str(schema_path),
            "benchmark_schema_sha256": schema_sha256,
            "color_order": "RGB; no channel swap",
            "image_resolution": [384, 384],
            "legacy_next_state_manifest_sha256": legacy_manifest_hash,
            "legacy_action_rows_replaced": differing_rows,
            "legacy_to_raw_action_mae": weighted_mae,
        }
    )
    manifest["instructions"] = dict(task_prompts)
    _write_json_atomic(source_manifest_path, manifest)

    contract = {
        "contract_version": "1.0.0",
        "robot_key": "ego_h1_inspire",
        "action_type": "joint",
        "action_dim": 38,
        "action_source": "raw_hdf5/action",
        "action_order": ["left_arm:7", "left_hand:12", "right_arm:7", "right_hand:12"],
        "stored_action_semantics": "absolute commanded joint target",
        "model_action_semantics": {
            "left_arm": "relative_to_current_state",
            "left_hand": "absolute",
            "right_arm": "relative_to_current_state",
            "right_hand": "absolute",
        },
        "action_horizon": 16,
        "fps": 30,
        "camera_order": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        "camera_policy": (
            "retain real wrist RGB where present; use 384x384 RGB black frames "
            "where the benchmark camera mask marks wrists absent"
        ),
        "image_resolution": [384, 384],
        "processor_resolution": [256, 256],
        "color_order": "RGB",
        "channel_swap": False,
        "prompt_source": "EgoVLA benchmark task registry",
        "prompts": task_prompts,
        "policy_joint_indices": list(policy_joint_indices),
        "benchmark_schema_path": str(schema_path),
        "benchmark_schema_sha256": schema_sha256,
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "real_wrist_episodes": 900,
        "black_wrist_episodes": 1003,
        "raw_root": str(raw_root),
    }
    _write_json_atomic(meta / "egovla_contract.json", contract)
    print(
        f"DONE raw-action rewrite dataset={dataset} frames={total_rows} "
        f"changed_rows={differing_rows} legacy_to_raw_mae={weighted_mae:.9f}"
    )


if __name__ == "__main__":
    main()
