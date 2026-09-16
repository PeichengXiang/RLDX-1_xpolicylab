# Spark0 -> RLDX-1 LeRobot v2.1 joint54

`convert_spark0_bench_to_lerobot_v21_joint54.py` builds a complete local
LeRobot v2.1 dataset (parquet, H.264 video, `info`, tasks, episodes,
episode/global statistics, `modality.json`, and a conversion manifest).

Default output: `data/Spark0_bench_lerobotV21_joint54`.

State/action order is left arm 7, left hand 20, right arm 7, right hand 20.
Every HDF5 frame is retained, and action is copied directly without a temporal
shift. Head and both wrist RGB streams are H.264/yuv420p at 25 FPS. Existing
output is never overwritten; completed output is published atomically.

```bash
/personal/miniconda3/envs/dexora_1b/bin/python \
  scripts/convert_spark0_bench_to_lerobot_v21_joint54.py
```

Smoke conversion:

```bash
/personal/miniconda3/envs/dexora_1b/bin/python \
  scripts/convert_spark0_bench_to_lerobot_v21_joint54.py \
  --tasks collect_objects --max-episodes-per-task 1 \
  --output-root /tmp/spark0_rldx_smoke
```
