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
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    source_v3 = args.source_v3.resolve()

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
    source_info = json.loads((source_v3 / "meta" / "info.json").read_text())
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError("packet audit source is not LeRobot v3.0")
    for key in ("total_episodes", "total_frames", "fps"):
        if source_info.get(key) != checks[key]:
            raise ValueError(f"source v3 info mismatch for {key}")
    source_records = _load_v3_episode_records(source_v3)
    if len(episodes) != EXPECTED_EPISODES or len(manifest["episodes"]) != EXPECTED_EPISODES:
        raise ValueError("episode metadata count mismatch")
    if len(tasks) != EXPECTED_TASKS or len({row["task"] for row in tasks}) != EXPECTED_TASKS:
        raise ValueError("task metadata count mismatch")
    if any(not str(row["task"]).strip() or row["task"] == "Do your job." for row in tasks):
        raise ValueError("placeholder/empty language task found")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    task_indices = set(task_by_index)

    offset = 0
    real_wrist = 0
    video_jobs: list[tuple[Path, int]] = []
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
        expected_prompt = str(manifest["instructions"][source_task])
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
        if not np.array_equal(action[:-1], state[1:]) or not np.array_equal(
            action[-1], state[-1]
        ):
            raise ValueError(f"episode {episode_index}: action is not next observed qpos")
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

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, (path, expected, actual) in enumerate(
            pool.map(_video_frame_count, video_jobs), start=1
        ):
            if actual != expected:
                raise ValueError(f"video frame mismatch: {path}: {actual} != {expected}")
            if completed % 500 == 0:
                print(f"verified video frames: {completed}/{len(video_jobs)}")

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

    bound_files = (
        "info.json",
        "modality.json",
        "stats.json",
        "relative_stats.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "xpolicylab_source_conversion.json",
    )
    optional_bound_files = ("rldx_v21_video_trim.json",)
    marker = {
        "audit_version": "2.0.0",
        "codebase_version": "v2.1",
        "action_type": "joint",
        "action_dim": 38,
        "action_horizon": 16,
        "fps": 30,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": len(video_jobs),
        "lossless_packet_slices_verified": packet_segments,
        "lossless_video_packets_verified": packet_frames,
        "real_wrist_episodes": real_wrist,
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
        "action_type=joint action_dim=38 fps=30"
    )


if __name__ == "__main__":
    main()
