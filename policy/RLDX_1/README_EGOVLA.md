# RLDX-1 × EgoVLA joint38

This recipe fine-tunes RLDX-1 on the 12-task EgoVLA H1 + Inspire dataset. It
is isolated from the existing Spark joint54 recipe and does not modify the
vendored `RLDX-1/` upstream checkout.

## Contract

- robot / `env_cfg_type`: `ego_h1_inspire`
- `action_type`: `joint`
- state/action layout: left arm 7, left hand 12, right arm 7, right hand 12
- stored action: next observed absolute qpos, 38D
- RLDX transform: arms `RELATIVE`, hands `ABSOLUTE`
- cameras: head, left wrist, right wrist; missing raw wrists remain black with
  `observation.camera_mask=0`
- source frequency: 30 FPS
- action horizon: 16; video context: 4 frames at stride 2

The launcher performs strict full-parameter fine-tuning: the full LLM, vision
tower, cognition tokens, projector, and diffusion action model are trainable;
neither backbone nor action LoRA is enabled. It uses DeepSpeed ZeRO-3 and
gradient checkpointing so this surface can be tested on eight 80 GB GPUs.

## Data

Run on a machine where the verified Pi0.5 LeRobot v3 dataset and its Python
environment are available:

```bash
cd /personal/xiangpc/0811_Xpolicylab_bench/RLDX_1
bash data_scripts/prepare_egovla_rldx_v21.sh
```

The script hard-links the immutable v3 source into private staging, converts
only that staging copy to LeRobot v2.1, computes exact absolute and arm-relative
statistics, audits every parquet episode and language label, verifies every
video frame count, and compares all 1,531,638 compressed packets against the
exact v3 source slices before atomically publishing:

```text
data/EgoVLA_benchmark_rldx_v21
```

Training refuses data whose audit-bound metadata hashes changed.

## A800_13 runtime

The rented node exposes the shared filesystem as `/mnt/xspark-data`. The
launcher uses only assets below `/mnt/xspark-data/xiangpc`, including the clean
RLDX base checkpoint and the existing Python 3.10 package set. System FFmpeg is
required by upstream TorchCodec.

One-step, eight-GPU smoke test:

```bash
cd /mnt/xspark-data/xiangpc/0811_Xpolicylab_bench/RLDX_1/policy/RLDX_1
bash prepare_egovla_full_base_overlay.sh
bash train_egovla_joint38.sh smoke
```

Formal run target (8 GPUs, global batch 64, per-GPU batch 8, 80,000 optimizer
steps, saving every 10,000 steps, LR 2e-5):

```bash
export WANDB_API_KEY='...'
bash train_egovla_joint38.sh start
```

The key must be supplied through the environment and is never stored in the
repository or launcher. The default W&B project is
`XPolicyLab-RLDX1-EgoVLA`. Checkpoints land in:

```text
policy/RLDX_1/checkpoints/
  <EgoVLA-run-directory>/
```

An already-started staged job may retain a historical run-directory suffix for
continuity; that naming detail is not the stopping criterion.  The effective
target remains 80,000 steps with a 10,000-step save cadence.  If an older
launcher state publishes intermediate checkpoints before continuation, those
artifacts are not the final run; continuation must still finish at 80,000 and
publish the requested cadence thereafter.

To continue an interrupted run, supply the same W&B environment and run:

```bash
bash train_egovla_joint38.sh resume
```

## Evaluation

After a complete checkpoint exists, use the standard adapter entry point:

```bash
RLDX_POLICY_PYTHON_BIN=/mnt/xspark-data/xiangpc/.uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 \
RLDX_POLICY_SITE_PACKAGES=/mnt/xspark-data/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/RLDX-1/.venv/lib/python3.10/site-packages \
RLDX_EVAL_PYTHON_BIN="/mnt/xspark-data/xiangpc/EgoVLA benchmark/.runtime/conda/egovla-isaaclab-1.2.0/bin/python" \
bash eval.sh EgoVLA <task_name> \
  /absolute/path/to/checkpoint-80000 \
  ego_h1_inspire joint 0 0 0 \
  /mnt/xspark-data/xiangpc/.uv-python/cpython-3.10.20-linux-x86_64-gnu \
  "/mnt/xspark-data/xiangpc/EgoVLA benchmark/.runtime/conda/egovla-isaaclab-1.2.0"
```

Real numerical inference remains unverified until a trained joint38 checkpoint
is produced. Debug transport alone can be checked with the adapter's reserved
`ckpt_name=debug` flow.
