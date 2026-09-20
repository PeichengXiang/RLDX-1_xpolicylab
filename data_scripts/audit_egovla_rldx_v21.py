#!/usr/bin/env python3
"""Fail-closed audit for the EgoVLA LeRobot v2.1 dataset consumed by RLDX."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egovla_authority import load_schema


EXPECTED_EPISODES = 1903
EXPECTED_FRAMES = 510_546
EXPECTED_TASKS = 12
EXPECTED_REAL_WRIST = 900
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _column_numpy(table: pa.Table, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values).reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return np.stack(array.to_numpy(zero_copy_only=False))
    return array.to_numpy(zero_copy_only=False)


def _file_sha256(path: Path) -> tuple[Path, str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return path, digest.hexdigest(), path.stat().st_size


def _tree_sha256(root: Path, paths: list[Path], workers: int) -> tuple[str, int]:
    """Bind file paths, sizes, and contents into one deterministic digest."""

    ordered = sorted(path.resolve() for path in paths)
    digest = hashlib.sha256()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as pool:
        results = pool.map(_file_sha256, ordered)
        for path, file_digest, size in results:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(str(size).encode())
            digest.update(b"\0")
            digest.update(file_digest.encode())
            digest.update(b"\n")
            total_bytes += size
    return digest.hexdigest(), total_bytes


def _video_frame_count(item: tuple[Path, int]) -> tuple[Path, int, int]:
    path, expected = item
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    value = result.stdout.strip().splitlines()
    if not value or not value[-1].isdigit():
        raise ValueError(f"ffprobe returned no frame count for {path}: {result.stdout!r}")
    return path, expected, int(value[-1])


def _verify_black_first_frame(path: Path) -> Path:
    container = av.open(str(path))
    try:
        frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
    finally:
        container.close()
    if frame.shape != (384, 384, 3) or frame.dtype != np.uint8 or np.any(frame):
        raise ValueError(f"missing-wrist video does not start with a 384x384 RGB black frame: {path}")
    return path


def _load_v3_episode_records(root: Path) -> dict[int, dict]:
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no v3 episode metadata under {root}")
    records: dict[int, dict] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in records:
                raise ValueError(f"duplicate v3 episode metadata: {index}")
            records[index] = row
    return records


def _packet_records(path: Path):
    """Yield non-empty compressed video packets and their normalized metadata."""

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            yield (
                bytes(packet),
                packet.pts,
                packet.dts,
                packet.duration,
                bool(packet.is_keyframe),
                packet.time_base,
            )
    finally:
        container.close()


def _relative_timestamp(value: int | None, origin: int | None) -> int | None:
    if value is None or origin is None:
        return None
    return value - origin


def _verify_packet_group(
    item: tuple[Path, list[tuple[int, Path, int]]],
) -> tuple[Path, int, int]:
    """Verify every v2 packet is the exact intended slice of one v3 MP4."""

    source_path, segments = item
    segments = sorted(segments, key=lambda segment: segment[0])
    source_packets = iter(_packet_records(source_path))
    source_index = 0
    verified_frames = 0
    previous_stop = 0

    for start, output_path, expected in segments:
        if start < previous_stop:
            raise ValueError(f"overlapping source video slices in {source_path}")
        while source_index < start:
            try:
                next(source_packets)
            except StopIteration as exc:
                raise ValueError(f"source video ended before frame {start}: {source_path}") from exc
            source_index += 1

        output_packets = iter(_packet_records(output_path))
        source_origin_pts = source_origin_dts = None
        output_origin_pts = output_origin_dts = None
        for frame_index in range(expected):
            try:
                source_packet = next(source_packets)
            except StopIteration as exc:
                raise ValueError(
                    f"source video ended inside segment: {source_path} frame={source_index}"
                ) from exc
            try:
                output_packet = next(output_packets)
            except StopIteration as exc:
                raise ValueError(
                    f"output video ended early: {output_path} frame={frame_index}"
                ) from exc

            source_payload, source_pts, source_dts, source_duration, source_key, source_tb = (
                source_packet
            )
            output_payload, output_pts, output_dts, output_duration, output_key, output_tb = (
                output_packet
            )
            if frame_index == 0:
                source_origin_pts, source_origin_dts = source_pts, source_dts
                output_origin_pts, output_origin_dts = output_pts, output_dts
            if source_payload != output_payload:
                raise ValueError(
                    f"compressed video payload mismatch: {output_path} frame={frame_index}"
                )
            if source_tb != output_tb or source_duration != output_duration:
                raise ValueError(f"video packet timing basis mismatch: {output_path}")
            if source_key != output_key:
                raise ValueError(f"video keyframe flag mismatch: {output_path}")
            if _relative_timestamp(source_pts, source_origin_pts) != _relative_timestamp(
                output_pts, output_origin_pts
            ) or _relative_timestamp(source_dts, source_origin_dts) != _relative_timestamp(
                output_dts, output_origin_dts
            ):
                raise ValueError(
                    f"normalized video packet timeline mismatch: {output_path} frame={frame_index}"
                )
            source_index += 1
            verified_frames += 1

        try:
            next(output_packets)
        except StopIteration:
            pass
        else:
            raise ValueError(f"output video has packets beyond episode boundary: {output_path}")
        previous_stop = start + expected

    return source_path, len(segments), verified_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source-v3", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    source_v3 = args.source_v3.resolve()
    raw_root = args.raw_root.resolve()
    schema_path = args.schema.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    task_prompts, joint_indices, schema_sha256 = load_schema(schema_path)
    policy_joint_indices = np.asarray(joint_indices, dtype=np.int64)
    raw_paths = sorted({*raw_root.rglob("*.hdf5"), *raw_root.rglob("*.h5")})
    if len(raw_paths) != EXPECTED_EPISODES:
        raise ValueError(
            f"raw root contains {len(raw_paths)} HDF5 episodes, expected {EXPECTED_EPISODES}"
        )
    raw_relative_paths = {path.relative_to(raw_root).as_posix() for path in raw_paths}
    if len(raw_relative_paths) != EXPECTED_EPISODES:
        raise ValueError("raw HDF5 episode paths are not unique")

    info = json.loads((dataset / "meta" / "info.json").read_text())
    checks = {
        "codebase_version": "v2.1",
        "robot_type": "ego_h1_inspire",
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "total_tasks": EXPECTED_TASKS,
        "fps": 30,
    }
    for key, expected in checks.items():
        if info.get(key) != expected:
            raise ValueError(f"info.{key}={info.get(key)!r}, expected {expected!r}")
    for key in ("observation.state", "action", "provenance.raw_commanded_action"):
        if info["features"][key]["shape"] != [38]:
            raise ValueError(f"{key} is not joint38")
    if info["features"]["observation.camera_mask"]["shape"] != [3]:
        raise ValueError("camera mask must have three entries")
    for camera in CAMERAS:
        feature = info["features"][camera]
        if feature["dtype"] != "video" or feature["shape"] != [384, 384, 3]:
            raise ValueError(f"invalid video feature {camera}: {feature}")

    expected_modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "left_hand": {"start": 7, "end": 19},
            "right_arm": {"start": 19, "end": 26},
            "right_hand": {"start": 26, "end": 38},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "left_hand": {"start": 7, "end": 19},
            "right_arm": {"start": 19, "end": 26},
            "right_hand": {"start": 26, "end": 38},
        },
        "video": {
            "cam_head": {"original_key": CAMERAS[0]},
            "cam_left_wrist": {"original_key": CAMERAS[1]},
            "cam_right_wrist": {"original_key": CAMERAS[2]},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }
    modality = json.loads((dataset / "meta" / "modality.json").read_text())
    if modality != expected_modality:
        raise ValueError("meta/modality.json does not match EgoVLA joint38 contract")

    episodes = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(dataset / "meta" / "tasks.jsonl")
    manifest = json.loads(
        (dataset / "meta" / "xpolicylab_source_conversion.json").read_text()
    )
    contract = json.loads((dataset / "meta" / "egovla_contract.json").read_text())
    if manifest.get("training_action_source") != (
        "raw HDF5 /action indexed from native 50D to policy joint38; never state[t+1]"
    ):
        raise ValueError("conversion manifest is not bound to raw HDF5 /action")
    if manifest.get("instructions") != task_prompts:
        raise ValueError("conversion prompts do not exactly match the benchmark registry")
    if manifest.get("benchmark_schema_sha256") != schema_sha256:
        raise ValueError("conversion manifest is not bound to the current benchmark schema")
    if contract.get("action_source") != "raw_hdf5/action":
        raise ValueError("dataset contract is not bound to raw HDF5 /action")
    if contract.get("benchmark_schema_sha256") != schema_sha256:
        raise ValueError("dataset contract is not bound to the current benchmark schema")
    if contract.get("policy_joint_indices") != list(joint_indices):
        raise ValueError("dataset joint mapping differs from the benchmark schema")
    if contract.get("color_order") != "RGB" or contract.get("channel_swap") is not False:
        raise ValueError("dataset color contract must be RGB with no channel swap")
    if contract.get("image_resolution") != [384, 384] or contract.get(
        "processor_resolution"
    ) != [256, 256]:
        raise ValueError("dataset/inference resolution contract changed")
    source_info = json.loads((source_v3 / "meta" / "info.json").read_text())
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError("packet audit source is not LeRobot v3.0")
    for key in ("total_episodes", "total_frames", "fps"):
        if source_info.get(key) != checks[key]:
            raise ValueError(f"source v3 info mismatch for {key}")
    source_records = _load_v3_episode_records(source_v3)
    if len(episodes) != EXPECTED_EPISODES or len(manifest["episodes"]) != EXPECTED_EPISODES:
        raise ValueError("episode metadata count mismatch")
    manifest_relative_paths = [
        str(row.get("source_relative_path", "")) for row in manifest["episodes"]
    ]
    if len(set(manifest_relative_paths)) != EXPECTED_EPISODES:
        raise ValueError("source conversion manifest contains duplicate HDF5 paths")
    if set(manifest_relative_paths) != raw_relative_paths:
        missing = sorted(raw_relative_paths - set(manifest_relative_paths))[:5]
        extra = sorted(set(manifest_relative_paths) - raw_relative_paths)[:5]
        raise ValueError(f"manifest/raw HDF5 path set mismatch: missing={missing} extra={extra}")
    if len(tasks) != EXPECTED_TASKS or len({row["task"] for row in tasks}) != EXPECTED_TASKS:
        raise ValueError("task metadata count mismatch")
    if any(not str(row["task"]).strip() or row["task"] == "Do your job." for row in tasks):
        raise ValueError("placeholder/empty language task found")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    if set(task_by_index) != set(range(EXPECTED_TASKS)):
        raise ValueError("task indices must be unique and contiguous from 0 through 11")
    task_indices = set(task_by_index)

    offset = 0
    real_wrist = 0
    next_state_absolute_error = 0.0
    next_state_values = 0
    parquet_jobs: list[Path] = []
    video_jobs: list[tuple[Path, int]] = []
    black_video_jobs: list[Path] = []
    packet_groups: dict[Path, list[tuple[int, Path, int]]] = {}
    for episode_index, (episode, source_episode) in enumerate(
        zip(episodes, manifest["episodes"], strict=True)
    ):
        expected_length = int(source_episode["frames"])
        source_record = source_records.get(episode_index)
        if source_record is None or int(source_record["length"]) != expected_length:
            raise ValueError(f"v3 episode record mismatch at {episode_index}")
        if int(source_episode["dataset_episode_index"]) != episode_index:
            raise ValueError(f"source manifest episode index mismatch at {episode_index}")
        if int(episode["episode_index"]) != episode_index:
            raise ValueError(f"episode index mismatch at {episode_index}")
        if int(episode["length"]) != expected_length:
            raise ValueError(f"episode length mismatch at {episode_index}")
        source_task = str(source_episode["task"])
        if source_task not in task_prompts:
            raise ValueError(f"episode {episode_index}: unknown source task {source_task!r}")
        source_relative_path = Path(str(source_episode["source_relative_path"]))
        source_episode_index = int(source_episode["source_episode_index"])
        if (
            source_relative_path.parts != (source_task, f"episode_{source_episode_index}.hdf5")
            or source_relative_path.as_posix() not in raw_relative_paths
        ):
            raise ValueError(
                f"episode {episode_index}: source path/task/index disagree: "
                f"{source_relative_path} task={source_task} index={source_episode_index}"
            )
        expected_prompt = task_prompts[source_task]
        if episode.get("tasks") != [expected_prompt]:
            raise ValueError(
                f"episode {episode_index}: language metadata does not match source task"
            )

        chunk = episode_index // int(info["chunks_size"])
        parquet_path = dataset / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index
        )
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "provenance.raw_commanded_action",
                "observation.camera_mask",
                "index",
                "episode_index",
                "frame_index",
                "task_index",
            ],
        )
        parquet_jobs.append(parquet_path)
        if table.num_rows != expected_length:
            raise ValueError(f"parquet row count mismatch: {parquet_path}")
        state = _column_numpy(table, "observation.state").astype(np.float32, copy=False)
        action = _column_numpy(table, "action").astype(np.float32, copy=False)
        raw_action = _column_numpy(table, "provenance.raw_commanded_action").astype(
            np.float32, copy=False
        )
        camera_mask = _column_numpy(table, "observation.camera_mask").astype(
            np.float32, copy=False
        )
        for key, array, shape in (
            ("state", state, (expected_length, 38)),
            ("action", action, (expected_length, 38)),
            ("raw_action", raw_action, (expected_length, 38)),
            ("camera_mask", camera_mask, (expected_length, 3)),
        ):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"episode {episode_index}: invalid {key} {array.shape}")
        if not np.array_equal(action, raw_action):
            raise ValueError(f"episode {episode_index}: action differs from raw provenance")
        next_state_absolute_error += float(
            np.abs(action[:-1].astype(np.float64) - state[1:]).sum()
        )
        next_state_values += int(action[:-1].size)

        raw_path = raw_root / source_relative_path
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        with h5py.File(raw_path, "r") as handle:
            if "action" not in handle:
                raise KeyError(f"{raw_path}: missing /action")
            if "observations/qpos" not in handle:
                raise KeyError(f"{raw_path}: missing /observations/qpos")
            raw_native = np.asarray(handle["action"], dtype=np.float32)
            raw_qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
            image_keys = set(handle["observations/images"].keys())
        if raw_native.shape != (expected_length, 50):
            raise ValueError(
                f"episode {episode_index}: raw /action shape {raw_native.shape}, "
                f"expected ({expected_length}, 50)"
            )
        if raw_qpos.shape != (expected_length, 50):
            raise ValueError(
                f"episode {episode_index}: raw qpos shape {raw_qpos.shape}, "
                f"expected ({expected_length}, 50)"
            )
        if not np.isfinite(raw_native).all() or not np.isfinite(raw_qpos).all():
            raise ValueError(f"episode {episode_index}: raw HDF5 action/qpos contains NaN/Inf")
        if not np.array_equal(raw_action, raw_native[:, policy_joint_indices]):
            raise ValueError(
                f"episode {episode_index}: provenance is not the source HDF5 /action"
            )
        if not np.array_equal(state, raw_qpos[:, policy_joint_indices]):
            raise ValueError(
                f"episode {episode_index}: state is not the source HDF5 /observations/qpos"
            )
        expected_image_keys = {"main"}
        if bool(source_episode["has_real_wrist_cameras"]):
            expected_image_keys.update({"left_hand", "right_hand"})
        if image_keys != expected_image_keys:
            raise ValueError(
                f"episode {episode_index}: raw camera keys {sorted(image_keys)} "
                "disagree with the conversion manifest"
            )
        if not np.array_equal(
            _column_numpy(table, "index"), np.arange(offset, offset + expected_length)
        ):
            raise ValueError(f"episode {episode_index}: global index mismatch")
        if not np.all(_column_numpy(table, "episode_index") == episode_index):
            raise ValueError(f"episode {episode_index}: episode_index column mismatch")
        if not np.array_equal(
            _column_numpy(table, "frame_index"), np.arange(expected_length)
        ):
            raise ValueError(f"episode {episode_index}: frame_index mismatch")
        episode_task_indices = np.unique(_column_numpy(table, "task_index"))
        if len(episode_task_indices) != 1 or int(episode_task_indices[0]) not in task_indices:
            raise ValueError(f"episode {episode_index}: invalid task_index")
        if task_by_index[int(episode_task_indices[0])] != expected_prompt:
            raise ValueError(f"episode {episode_index}: task_index prompt mismatch")

        has_wrist = bool(source_episode["has_real_wrist_cameras"])
        expected_mask = np.array([1.0, float(has_wrist), float(has_wrist)], dtype=np.float32)
        if not np.all(camera_mask == expected_mask):
            raise ValueError(f"episode {episode_index}: camera mask mismatch")
        real_wrist += int(has_wrist)

        for camera in CAMERAS:
            video_path = dataset / info["video_path"].format(
                episode_chunk=chunk,
                episode_index=episode_index,
                video_key=camera,
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            video_jobs.append((video_path, expected_length))
            if not has_wrist and camera != CAMERAS[0]:
                black_video_jobs.append(video_path)
            source_video_path = source_v3 / source_info["video_path"].format(
                video_key=camera,
                chunk_index=int(source_record[f"videos/{camera}/chunk_index"]),
                file_index=int(source_record[f"videos/{camera}/file_index"]),
            )
            if not source_video_path.is_file():
                raise FileNotFoundError(source_video_path)
            source_start = round(
                float(source_record[f"videos/{camera}/from_timestamp"])
                * int(source_info["fps"])
            )
            packet_groups.setdefault(source_video_path, []).append(
                (source_start, video_path, expected_length)
            )
        offset += expected_length

    if offset != EXPECTED_FRAMES or real_wrist != EXPECTED_REAL_WRIST:
        raise ValueError(f"totals mismatch: frames={offset}, real_wrist={real_wrist}")
    if next_state_values == 0 or not np.isfinite(next_state_absolute_error):
        raise ValueError("cannot compute raw-action versus next-state diagnostic")
    raw_vs_next_state_mae = next_state_absolute_error / next_state_values

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, (path, expected, actual) in enumerate(
            pool.map(_video_frame_count, video_jobs), start=1
        ):
            if actual != expected:
                raise ValueError(f"video frame mismatch: {path}: {actual} != {expected}")
            if completed % 500 == 0:
                print(f"verified video frames: {completed}/{len(video_jobs)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, _path in enumerate(
            pool.map(_verify_black_first_frame, black_video_jobs), start=1
        ):
            if completed % 500 == 0:
                print(
                    f"verified black missing-wrist first frames: "
                    f"{completed}/{len(black_video_jobs)}"
                )
    if len(black_video_jobs) != (EXPECTED_EPISODES - EXPECTED_REAL_WRIST) * 2:
        raise ValueError("missing-wrist black video count changed")

    packet_segments = packet_frames = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(packet_groups))) as pool:
        for source_path, segments, frames in pool.map(
            _verify_packet_group, packet_groups.items()
        ):
            packet_segments += segments
            packet_frames += frames
            print(
                f"verified lossless packet slices: {source_path.name} "
                f"segments={segments} frames={frames}"
            )
    if packet_segments != len(video_jobs) or packet_frames != EXPECTED_FRAMES * len(CAMERAS):
        raise ValueError(
            f"packet audit totals mismatch: segments={packet_segments} frames={packet_frames}"
        )

    parquet_tree_sha256, parquet_total_bytes = _tree_sha256(
        dataset, parquet_jobs, args.workers
    )
    video_paths = [path for path, _expected in video_jobs]
    video_tree_sha256, video_total_bytes = _tree_sha256(
        dataset, video_paths, args.workers
    )

    bound_files = (
        "info.json",
        "modality.json",
        "stats.json",
        "relative_stats.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "xpolicylab_source_conversion.json",
        "egovla_contract.json",
    )
    optional_bound_files = ("rldx_v21_video_trim.json",)
    marker = {
        "audit_version": "3.0.0",
        "codebase_version": "v2.1",
        "action_type": "joint",
        "action_dim": 38,
        "action_source": "raw_hdf5/action",
        "action_is_next_state": False,
        "raw_vs_next_state_mae": raw_vs_next_state_mae,
        "action_horizon": 16,
        "fps": 30,
        "color_order": "RGB",
        "channel_swap": False,
        "source_image_resolution": [384, 384],
        "processor_image_resolution": [256, 256],
        "camera_policy": "real wrists when present; black wrists when absent",
        "prompt_source": "EgoVLA benchmark task registry exact strings",
        "benchmark_schema_sha256": schema_sha256,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": len(video_jobs),
        "lossless_packet_slices_verified": packet_segments,
        "lossless_video_packets_verified": packet_frames,
        "real_wrist_episodes": real_wrist,
        "black_wrist_episodes": len(episodes) - real_wrist,
        "black_wrist_videos_first_frame_verified": len(black_video_jobs),
        "parquet_files": len(parquet_jobs),
        "parquet_total_bytes": parquet_total_bytes,
        "parquet_tree_sha256": parquet_tree_sha256,
        "video_files": len(video_paths),
        "video_total_bytes": video_total_bytes,
        "video_tree_sha256": video_tree_sha256,
        "sha256": {
            name: hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest()
            for name in (*bound_files, *optional_bound_files)
            if (dataset / "meta" / name).is_file()
        },
    }
    (dataset / "meta" / "rldx_egovla_audit.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )

    print(
        f"DONE audit dataset={dataset} episodes={len(episodes)} frames={offset} "
        f"tasks={len(tasks)} videos={len(video_jobs)} real_wrist={real_wrist} "
        f"raw_vs_next_state_mae={raw_vs_next_state_mae:.9f} "
        "action_type=joint action_dim=38 action_source=raw_hdf5/action fps=30"
    )


if __name__ == "__main__":
    main()
