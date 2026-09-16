#!/usr/bin/env python3
"""Convert Spark0 benchmark HDF5 episodes to LeRobot v2.1 joint54.

The source is never modified. Every selected HDF5 frame is retained. State and
action are the direct concatenations

    left_arm(7), left_hand(20), right_arm(7), right_hand(20)

and the three HDF5 RGB streams are encoded as H.264 at the recorded 25 Hz. The
dataset is built in a unique sibling directory and atomically renamed only
after all parquet, metadata, statistics, and videos have been validated.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# XPolicyLab is the sole decoder for trajectory image bit streams. It returns
# RGB; deliberately do not apply an OpenCV BGR/RGB swap afterwards.
from utils.process_data import decode_image_bit  # noqa: E402


DEFAULT_SOURCE = Path("/personal/xspark_shared/hand_data/hdf5/spark0_bench_7tasks")
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spark0_bench_7task_0908"
TASK_NAMES = (
    "click_mouse",
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "put_food_in_microwave",
    "retrieve_gap",
    "stack_bowls",
)
ROBOT_DIR = "tianji_marvin_wuji"
FPS = 25
HEIGHT = 480
WIDTH = 640
JOINT_DIM = 54
CHUNK_SIZE = 1000
EPISODE_RE = re.compile(r"episode_(\d{7})\.hdf5$")

VIDEO_MAP = {
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}
VIDEO_MODALITY_MAP = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}

JOINT_NAMES = [
    *(f"left_arm_joint_{i}" for i in range(7)),
    *(f"left_hand_joint_{i}" for i in range(20)),
    *(f"right_arm_joint_{i}" for i in range(7)),
    *(f"right_hand_joint_{i}" for i in range(20)),
]


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    task_name: str
    task_index: int
    instruction: str
    source_episode_index: int
    episode_index: int
    path: Path


def decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="strict")
    text = str(value).strip()
    if not text:
        raise ValueError("empty HDF5 text value")
    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_tasks(raw: list[str] | None) -> tuple[str, ...]:
    if not raw:
        return TASK_NAMES
    requested: list[str] = []
    for item in raw:
        requested.extend(part.strip() for part in item.split(",") if part.strip())
    unknown = sorted(set(requested) - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; expected a subset of {TASK_NAMES}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate task in --tasks: {requested}")
    requested_set = set(requested)
    return tuple(task for task in TASK_NAMES if task in requested_set)


def discover_episodes(
    source_root: Path, tasks: tuple[str, ...], max_per_task: int | None
) -> tuple[list[EpisodeSource], list[dict[str, Any]]]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if max_per_task is not None and max_per_task < 1:
        raise ValueError("--max-episodes-per-task must be >= 1")

    episodes: list[EpisodeSource] = []
    task_rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(tasks):
        data_dir = source_root / task_name / ROBOT_DIR / "data"
        files = sorted(data_dir.glob("episode_*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no HDF5 episodes in {data_dir}")
        actual_indices: list[int] = []
        for path in files:
            match = EPISODE_RE.fullmatch(path.name)
            if not match:
                raise ValueError(f"unexpected episode filename: {path}")
            actual_indices.append(int(match.group(1)))
        if actual_indices != list(range(len(files))):
            raise ValueError(f"non-contiguous episode numbering in {data_dir}")
        selected = files[:max_per_task] if max_per_task is not None else files

        with h5py.File(selected[0], "r") as handle:
            instruction = decode_text(handle["instruction"][()])
            frequency = int(np.asarray(handle["additional_info/frequency"]).item())
            version = decode_text(handle["data_format_version"][()])
        if frequency != FPS or version != "v1.0":
            raise ValueError(
                f"{selected[0]}: expected source v1.0/{FPS}Hz, got {version}/{frequency}Hz"
            )
        task_rows.append({"task_index": task_index, "task": instruction})

        for path in selected:
            source_episode_index = int(EPISODE_RE.fullmatch(path.name).group(1))  # type: ignore[union-attr]
            episodes.append(
                EpisodeSource(
                    task_name=task_name,
                    task_index=task_index,
                    instruction=instruction,
                    source_episode_index=source_episode_index,
                    episode_index=len(episodes),
                    path=path,
                )
            )
    return episodes, task_rows


def read_matrix(
    handle: h5py.File, key: str, width: int, expected_length: int | None = None
) -> np.ndarray:
    if key not in handle:
        raise KeyError(f"{handle.filename}: missing {key}")
    value = np.asarray(handle[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{handle.filename}:{key}: expected (T,{width}), got {value.shape}")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(
            f"{handle.filename}:{key}: expected T={expected_length}, got {len(value)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{handle.filename}:{key}: contains NaN/Inf")
    return value


def load_joint_arrays(handle: h5py.File, episode: EpisodeSource) -> tuple[np.ndarray, np.ndarray]:
    state_left_arm = read_matrix(handle, "state/left_arm_joint_states", 7)
    length = len(state_left_arm)
    if length < 1:
        raise ValueError(f"{episode.path}: empty episode")
    state_left_hand = read_matrix(handle, "state/left_ee_joint_states", 20, length)
    state_right_arm = read_matrix(handle, "state/right_arm_joint_states", 7, length)
    state_right_hand = read_matrix(handle, "state/right_ee_joint_states", 20, length)
    action_left_arm = read_matrix(handle, "action/left_arm_joint_states", 7, length)
    action_left_hand = read_matrix(handle, "action/left_ee_joint_states", 20, length)
    action_right_arm = read_matrix(handle, "action/right_arm_joint_states", 7, length)
    action_right_hand = read_matrix(handle, "action/right_ee_joint_states", 20, length)

    state = np.concatenate(
        [state_left_arm, state_left_hand, state_right_arm, state_right_hand], axis=1
    ).astype(np.float32)
    action = np.concatenate(
        [action_left_arm, action_left_hand, action_right_arm, action_right_hand], axis=1
    ).astype(np.float32)
    if state.shape != (length, JOINT_DIM) or action.shape != state.shape:
        raise AssertionError(f"{episode.path}: invalid state/action shapes {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{episode.path}: converted state/action contains NaN/Inf")

    if decode_text(handle["instruction"][()]) != episode.instruction:
        raise ValueError(f"{episode.path}: instruction differs within task")
    frequency = int(np.asarray(handle["additional_info/frequency"]).item())
    if frequency != FPS:
        raise ValueError(f"{episode.path}: expected {FPS}Hz, got {frequency}Hz")
    for source_camera in VIDEO_MAP.values():
        color_key = f"vision/{source_camera}/colors"
        shape_key = f"vision/{source_camera}/shape"
        if color_key not in handle or shape_key not in handle:
            raise KeyError(f"{episode.path}: missing {source_camera} colors/shape")
        if len(handle[color_key]) != length:
            raise ValueError(
                f"{episode.path}:{color_key}: T={len(handle[color_key])}, expected {length}"
            )
        shape = np.asarray(handle[shape_key]).tolist()
        if shape != [HEIGHT, WIDTH, 3]:
            raise ValueError(f"{episode.path}:{shape_key}: expected {[HEIGHT, WIDTH, 3]}, got {shape}")
    return state, action


class RawH264Writer:
    def __init__(self, path: Path, width: int, height: int, preset: str) -> None:
        self.path = path
        self.tmp_path = path.with_name(path.stem + ".tmp.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            "23",
            "-g",
            str(FPS * 2),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.tmp_path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if self.process.stdin is None or self.process.stderr is None:
            raise RuntimeError("failed to open ffmpeg pipes")

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        try:
            self.process.stdin.write(frame.tobytes())  # type: ignore[union-attr]
        except BrokenPipeError as exc:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace")  # type: ignore[union-attr]
            raise RuntimeError(f"ffmpeg terminated for {self.path}: {stderr[-2000:]}") from exc

    def close(self) -> None:
        self.process.stdin.close()  # type: ignore[union-attr]
        return_code = self.process.wait()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace")  # type: ignore[union-attr]
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr[-2000:]}")
        os.replace(self.tmp_path, self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def decode_rgb(value: Any, expected_shape: tuple[int, int, int], label: str) -> np.ndarray:
    image = np.asarray(decode_image_bit(value))
    if image.shape != expected_shape:
        raise ValueError(f"{label}: decoded shape {image.shape}, expected {expected_shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{label}: decoded dtype {image.dtype}, expected uint8")
    return image


def probe_video(path: Path, expected_frames: int, width: int, height: int) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream, got {streams}")
    stream = streams[0]
    rate = float(Fraction(stream["avg_frame_rate"]))
    actual_frames = int(stream["nb_read_frames"])
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise ValueError(f"{path}: {key}={stream.get(key)!r}, expected {value!r}")
    if abs(rate - FPS) > 1e-6 or actual_frames != expected_frames:
        raise ValueError(
            f"{path}: fps/frames={rate}/{actual_frames}, expected {FPS}/{expected_frames}"
        )


def encode_episode_videos(
    handle: h5py.File,
    root: Path,
    episode_index: int,
    length: int,
    preset: str,
) -> None:
    writers = {
        video_key: RawH264Writer(
            root
            / "videos"
            / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4",
            WIDTH,
            HEIGHT,
            preset,
        )
        for video_key in VIDEO_MAP
    }
    try:
        for frame_index in range(length):
            for video_key, source_camera in VIDEO_MAP.items():
                image = decode_rgb(
                    handle[f"vision/{source_camera}/colors"][frame_index],
                    (HEIGHT, WIDTH, 3),
                    f"{handle.filename}:{source_camera}[{frame_index}]",
                )
                writers[video_key].write(image)
        for writer in writers.values():
            writer.close()
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise
    for writer in writers.values():
        probe_video(writer.path, length, WIDTH, HEIGHT)


def fixed_list(array: np.ndarray) -> pa.FixedSizeListArray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"fixed_list expects 2D input, got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=pa.from_numpy_dtype(array.dtype)), array.shape[1]
    )


def episode_feature_stats(array: np.ndarray) -> dict[str, list[int | float]]:
    value = np.asarray(array)
    if value.ndim == 1:
        value = value[:, None]
    numeric = value.astype(np.float64)
    return {
        "min": numeric.min(axis=0).tolist(),
        "max": numeric.max(axis=0).tolist(),
        "mean": numeric.mean(axis=0).tolist(),
        "std": numeric.std(axis=0).tolist(),
        "count": [int(len(numeric))],
    }


def dataset_feature_stats(parts: list[np.ndarray]) -> dict[str, list[float]]:
    value = np.concatenate(
        [np.asarray(part)[:, None] if np.asarray(part).ndim == 1 else np.asarray(part) for part in parts],
        axis=0,
    ).astype(np.float64)
    return {
        "mean": value.mean(axis=0).tolist(),
        "std": value.std(axis=0).tolist(),
        "min": value.min(axis=0).tolist(),
        "max": value.max(axis=0).tolist(),
        "q01": np.quantile(value, 0.01, axis=0).tolist(),
        "q99": np.quantile(value, 0.99, axis=0).tolist(),
    }


def build_info(total_episodes: int, total_frames: int, total_tasks: int) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [JOINT_DIM],
            "names": [JOINT_NAMES],
        },
        "action": {"dtype": "float32", "shape": [JOINT_DIM], "names": [JOINT_NAMES]},
    }
    for video_key in VIDEO_MAP:
        features[video_key] = {
            "dtype": "video",
            "shape": [3, HEIGHT, WIDTH],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": HEIGHT,
                "video.width": WIDTH,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "tianji_marvin_wuji",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_MAP),
        "total_chunks": math.ceil(total_episodes / CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_modality() -> dict[str, Any]:
    groups = {
        "left_arm": (0, 7),
        "left_hand": (7, 27),
        "right_arm": (27, 34),
        "right_hand": (34, 54),
    }
    modality: dict[str, Any] = {"state": {}, "action": {}}
    for kind, original_key in (("state", "observation.state"), ("action", "action")):
        for name, (start, end) in groups.items():
            modality[kind][name] = {
                "start": start,
                "end": end,
                "absolute": True,
                "dtype": "float32",
                "original_key": original_key,
            }
    # RLDX's policy config consumes the logical key ``cam_head`` while the
    # physical LeRobot feature keeps XPolicyLab's canonical ``cam_high`` name.
    modality["video"] = {
        logical_key: {"original_key": original_key}
        for logical_key, original_key in VIDEO_MODALITY_MAP.items()
    }
    modality["annotation"] = {
        "human.action.task_description": {"original_key": "task_index"}
    }
    return modality


def verify_dataset(root: Path, expected_episodes: int, expected_frames: int) -> None:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info["codebase_version"] != "v2.1":
        raise ValueError("info.json is not LeRobot v2.1")
    modality = json.loads((root / "meta/modality.json").read_text(encoding="utf-8"))
    if modality.get("video") != {
        key: {"original_key": value} for key, value in VIDEO_MODALITY_MAP.items()
    }:
        raise ValueError("modality video mapping is not cam_head + two wrist cameras")
    if info["total_episodes"] != expected_episodes or info["total_frames"] != expected_frames:
        raise ValueError("info.json episode/frame totals are inconsistent")
    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    videos = sorted(root.glob("videos/chunk-*/*/episode_*.mp4"))
    if len(parquets) != expected_episodes:
        raise ValueError(f"parquet count {len(parquets)} != {expected_episodes}")
    if len(videos) != expected_episodes * len(VIDEO_MAP):
        raise ValueError(
            f"video count {len(videos)} != {expected_episodes * len(VIDEO_MAP)}"
        )
    seen_frames = 0
    for episode_index, path in enumerate(parquets):
        table = pq.read_table(path)
        seen_frames += table.num_rows
        state_type = table.schema.field("observation.state").type
        action_type = table.schema.field("action").type
        if not pa.types.is_fixed_size_list(state_type) or state_type.list_size != JOINT_DIM:
            raise ValueError(f"{path}: invalid state type {state_type}")
        if not pa.types.is_fixed_size_list(action_type) or action_type.list_size != JOINT_DIM:
            raise ValueError(f"{path}: invalid action type {action_type}")
        ep_values = table["episode_index"].to_numpy()
        if not np.all(ep_values == episode_index):
            raise ValueError(f"{path}: episode_index mismatch")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(table.num_rows)):
            raise ValueError(f"{path}: frame_index mismatch")
    if seen_frames != expected_frames:
        raise ValueError(f"parquet frames {seen_frames} != {expected_frames}")
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    for key, width in (("observation.state", JOINT_DIM), ("action", JOINT_DIM)):
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            value = np.asarray(stats[key][stat_name], dtype=np.float64)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"stats {key}.{stat_name} is invalid")


def convert(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    tasks = normalize_tasks(args.tasks)
    episodes, task_rows = discover_episodes(
        source_root, tasks, args.max_episodes_per_task
    )
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_root}; choose a new --output-root"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    work = output_root.parent / f".{output_root.name}.incomplete-{stamp}-{os.getpid()}"
    if work.exists():
        raise FileExistsError(f"unexpected staging path already exists: {work}")
    work.mkdir()

    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    stats_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )
    }
    total_frames = 0
    try:
        for ordinal, episode in enumerate(episodes, start=1):
            with h5py.File(episode.path, "r") as handle:
                state, action = load_joint_arrays(handle, episode)
                length = len(state)
                frame_index = np.arange(length, dtype=np.int64)
                arrays = {
                    "observation.state": state,
                    "action": action,
                    "timestamp": (frame_index / FPS).astype(np.float32),
                    "frame_index": frame_index,
                    "episode_index": np.full(length, episode.episode_index, dtype=np.int64),
                    "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                    "task_index": np.full(length, episode.task_index, dtype=np.int64),
                }
                table = pa.table(
                    {
                        "observation.state": fixed_list(state),
                        "action": fixed_list(action),
                        "timestamp": pa.array(arrays["timestamp"], type=pa.float32()),
                        "frame_index": pa.array(frame_index, type=pa.int64()),
                        "episode_index": pa.array(arrays["episode_index"], type=pa.int64()),
                        "index": pa.array(arrays["index"], type=pa.int64()),
                        "task_index": pa.array(arrays["task_index"], type=pa.int64()),
                    }
                )
                parquet_path = (
                    work
                    / "data"
                    / f"chunk-{episode.episode_index // CHUNK_SIZE:03d}"
                    / f"episode_{episode.episode_index:06d}.parquet"
                )
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, parquet_path, compression="snappy")
                encode_episode_videos(
                    handle, work, episode.episode_index, length, args.ffmpeg_preset
                )

            episode_rows.append(
                {
                    "episode_index": episode.episode_index,
                    "tasks": [episode.instruction],
                    "length": length,
                }
            )
            episode_stats_rows.append(
                {
                    "episode_index": episode.episode_index,
                    "stats": {
                        key: episode_feature_stats(value) for key, value in arrays.items()
                    },
                }
            )
            for key, value in arrays.items():
                stats_parts[key].append(value)
            total_frames += length
            print(
                f"[{ordinal}/{len(episodes)}] {episode.task_name}/"
                f"episode_{episode.source_episode_index:07d}: {length} frames",
                flush=True,
            )

        write_json(work / "meta/info.json", build_info(len(episodes), total_frames, len(tasks)))
        write_jsonl(work / "meta/tasks.jsonl", task_rows)
        write_jsonl(work / "meta/episodes.jsonl", episode_rows)
        write_jsonl(work / "meta/episodes_stats.jsonl", episode_stats_rows)
        write_json(work / "meta/stats.json", {key: dataset_feature_stats(parts) for key, parts in stats_parts.items()})
        write_json(work / "meta/modality.json", build_modality())
        write_json(
            work / "meta/xpolicylab_conversion.json",
            {
                "format": "LeRobot v2.1",
                "source_root": str(source_root),
                "source_format": "XPolicyLab HDF5 v1.0",
                "tasks": list(tasks),
                "episodes_per_task_limit": args.max_episodes_per_task,
                "total_episodes": len(episodes),
                "total_frames": total_frames,
                "fps": FPS,
                "state_order": ["left_arm:7", "left_hand:20", "right_arm:7", "right_hand:20"],
                "action_order": ["left_arm:7", "left_hand:20", "right_arm:7", "right_hand:20"],
                "action_semantics": "direct absolute HDF5 action; no temporal shift",
                "terminal_frame_policy": "keep every source frame, including the final frame",
                "image_decoder": "utils.process_data.decode_image_bit",
                "color_order": "RGB; no channel swap",
                "videos": {key: f"vision/{value}/colors" for key, value in VIDEO_MAP.items()},
                "video_encoding": "H.264/yuv420p/25fps",
            },
        )
        verify_dataset(work, len(episodes), total_frames)
        os.rename(work, output_root)
        print(f"published atomically: {output_root}", flush=True)
    except Exception:
        if args.keep_incomplete:
            print(f"conversion failed; preserved staging directory: {work}", file=sys.stderr)
        elif work.exists():
            shutil.rmtree(work)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", "--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-episodes-per-task",
        "--episodes-per-task",
        dest="max_episodes_per_task",
        type=int,
        default=None,
        help="limit each selected task (use 1 for a smoke conversion)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="task names, space- or comma-separated"
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="veryfast",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
    )
    parser.add_argument("--keep-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
