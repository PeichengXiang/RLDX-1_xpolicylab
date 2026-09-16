#!/usr/bin/env python3
"""Atomically remove a single leaked boundary frame from legacy v2.1 videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
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
        raise ValueError(f"ffprobe returned no packet count for {path}")
    return int(value[-1])


def _trim(item: tuple[Path, int]) -> tuple[str, Path]:
    path, expected = item
    actual = _frame_count(path)
    if actual == expected:
        return "already_exact", path
    if actual != expected + 1:
        raise ValueError(f"unexpected source frame count: {path}: {actual} != {expected}+1")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.trim-incomplete-", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                str(expected),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "1",
                "-y",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if _frame_count(temporary) != expected:
            raise ValueError(f"trimmed output is not exactly {expected} frames: {path}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "trimmed", path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = _read_jsonl(dataset / "meta/episodes.jsonl")
    if info.get("codebase_version") != "v2.1" or len(episodes) != 1903:
        raise ValueError("input is not the expected complete EgoVLA v2.1 staging dataset")

    jobs: list[tuple[Path, int]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        expected = int(episode["length"])
        chunk = episode_index // int(info["chunks_size"])
        for camera in CAMERAS:
            path = dataset / info["video_path"].format(
                episode_chunk=chunk,
                episode_index=episode_index,
                video_key=camera,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            jobs.append((path, expected))

    counts = {"trimmed": 0, "already_exact": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, (status, _) in enumerate(pool.map(_trim, jobs), start=1):
            counts[status] += 1
            if completed % 250 == 0:
                print(f"processed videos: {completed}/{len(jobs)}", flush=True)

    marker = {
        "repair": "remove_cross_episode_tail_packet",
        "repair_version": "1.0.0",
        "method": "ffmpeg_stream_copy_frames_v_expected_episode_length",
        "atomic_per_file": True,
        "total_videos": len(jobs),
        **counts,
    }
    (dataset / "meta/rldx_v21_video_trim.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    print(f"DONE trim videos={len(jobs)} {counts}")


if __name__ == "__main__":
    main()
