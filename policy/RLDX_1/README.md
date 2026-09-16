# RLDX_1

**Contributor:** xiangpc | **Paper:** [RLDX-1 Technical Report](https://arxiv.org/abs/2605.03269) | **arXiv:** [2605.03269](https://arxiv.org/abs/2605.03269) | **Original code:** [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1)

`RLDX_1` integrates the upstream RLDX-1 policy through a nested, vendored `RLDX-1/` checkout pinned to commit `cf67c31f7f168ebbca6d06e8c3c23914cb94435c`. The private evaluation target is EgoVLA: `bench_name=EgoVLA`, `env_cfg_type=ego_h1_inspire`, and `action_type=joint`; end-effector pose actions are not supported.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

The EgoVLA state/action order is left arm, left hand, right arm, right hand. The private registry at `utils/robot/_robot_info.json` records `ego_h1_inspire` as 7, 12, 7, and 12 (38 total). The RLDX recipe uses four RGB frames at stride two, three cameras, a 16-step action horizon, and `GENERAL_EMBODIMENT`. The dataset stores the next observed absolute qpos. During training, `egovla_joint38_config.py` converts both arm groups to `RELATIVE` deltas while keeping both hand groups `ABSOLUTE`; inference converts arm deltas back to joint-position targets. `action_type=joint` names the control space, not the absolute/relative representation. The audited external EgoVLA bridge scatters these 38 arm/hand targets into the simulator's 50-joint action and preserves the current qpos for the 12 leg joints.

## Installation

The policy side uses the upstream Python 3.10 `uv` environment. The install entry point has no positional arguments:

```bash
bash install.sh
```

Fresh-clone example:

```bash
git clone --recurse-submodules https://github.com/PeichengXiang/RLDX-1_xpolicylab.git
cd RLDX-1_xpolicylab/policy/RLDX_1
bash install.sh
```

This runs the official `uv sync --python 3.10` recipe, installs RLDX-1 and XPolicyLab editable, and creates the policy environment at `RLDX-1/.venv`.

## Data Processing

`process_data.sh` forwards optional arguments unchanged to the repository converter:

```bash
bash process_data.sh [converter_args...]
```

From the repository root, convert all seven SparkArena tasks into the directory
used by `train.sh`:

```bash
cd policy/RLDX_1
bash process_data.sh \
  --source-root /path/to/spark0_bench_7tasks \
  --output-root ../../data/spark0_bench_7task_0908
```

The converted EgoVLA data used by this checkout is `data/EgoVLA_benchmark_rldx_v21`. The output must be LeRobot v2.1 and contain `meta/modality.json`; state/action slices must expose `left_arm`, `left_hand`, `right_arm`, and `right_hand`, while video exposes `cam_head`, `cam_left_wrist`, and `cam_right_wrist`. Offline image decoding is owned by the converter and must use `XPolicyLab.utils.process_data.decode_image_bit`.

## Training

Authenticate W&B in the RLDX environment before launch. Do not store an API key in this repository.

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_ids>
```

Runnable eight-GPU example:

```bash
cd policy/RLDX_1
bash train.sh Spark0_bench rldx1_ft tianji_marvin_wuji joint 0 0,1,2,3,4,5,6,7
```

The wrapper calls the official `rldx/experiment/launch_train.py` entry point with eight GPUs, global batch size 64, gradient accumulation 1, learning rate `1e-4`, 100,000 optimizer steps, and a checkpoint every 5,000 steps with 20 retained. W&B is enabled and `WANDB_PROJECT` defaults to `spark0-bench-rldx1`.

Only paths and run metadata are configurable through the wrapper:

```bash
RLDX_BASE_MODEL_PATH=/absolute/checkpoint \
RLDX_DATASET_PATH=/absolute/Spark0_bench_lerobotV21_joint54 \
WANDB_PROJECT=my-project \
bash train.sh Spark0_bench rldx1_ft tianji_marvin_wuji joint 0 0,1,2,3,4,5,6,7
```

The run directory follows the shared convention:

```text
policy/RLDX_1/checkpoints/Spark0_bench-rldx1_ft-tianji_marvin_wuji-joint-0/
```

RLDX-1 at the pinned commit has no training-seed CLI field. The `seed` argument remains part of the XPolicyLab run name, but the adapter does not claim that it controls every upstream RNG.

## Evaluation

The standard ten-argument entry point is:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Runnable EgoVLA trained-checkpoint example (the final checkpoint path may be a
`checkpoint-80000` child):

```bash
XPL_ROOT="$(git rev-parse --show-toplevel)"
POLICY_DIR="${XPL_ROOT}/policy/RLDX_1"
EVAL_ENV_PATH=/path/to/egovla-isaaclab-1.2.0
export EVAL_MAIN_ROOT=/path/to/EgoVLA
export RLDX_POLICY_PYTHON_BIN="${POLICY_DIR}/RLDX-1/.venv/bin/python"
export RLDX_POLICY_SITE_PACKAGES="${POLICY_DIR}/RLDX-1/.venv/lib/python3.10/site-packages"
export RLDX_EVAL_PYTHON_BIN="${EVAL_ENV_PATH}/bin/python"
cd "${POLICY_DIR}"
bash eval.sh EgoVLA close_drawer \
  /absolute/path/to/checkpoint-80000 \
  ego_h1_inspire joint 0 0 0 \
  "${POLICY_DIR}/RLDX-1/.venv" "${EVAL_ENV_PATH}"
```

Checkpoint roots follow `XPolicyLab.utils.checkpoint_resolver` precedence: an explicit `model_path`, `ckpt_name` supplied as a path, the standard concatenated run name, then `checkpoints/<ckpt_name>`. Historical bare-path and `pretrain_model/<ckpt_name>` locations remain lower-priority compatibility fallbacks. If a resolved run directory contains `checkpoint-*` children, the largest numeric step is loaded.

The reserved `ckpt_name=debug` skips RLDX weight loading and returns zero actions with registry-derived dimensions. The private wrapper resolves direct Python executables and does not require conda on A800_13. Use both debug commands to exercise plain and encoded observations:

```bash
EVAL_ENV_TYPE=debug \
RLDX_DEBUG_EPISODES=1 bash eval.sh EgoVLA close_drawer debug \
  ego_h1_inspire joint 0 0 0 "${POLICY_DIR}/RLDX-1/.venv" "${EVAL_ENV_PATH}"

DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug \
RLDX_DEBUG_EPISODES=1 bash eval.sh EgoVLA close_drawer debug \
  ego_h1_inspire joint 0 0 0 "${POLICY_DIR}/RLDX-1/.venv" "${EVAL_ENV_PATH}"
```

Debug mode validates adapter transport, action keys, and shapes; it does not validate numerical RLDX inference.

## Model Assets

Expected layout:

```text
policy/RLDX_1/
├── RLDX-1/                       # pinned upstream checkout
├── egovla_joint38_config.py      # RLDX modality registration
├── model.py                      # XPolicyLab ↔ RLDXPolicy adapter
└── checkpoints/                  # fine-tuning outputs

pretrain_model/
├── RLDX-1-PT-1592013a/          # official pretrained checkpoint
└── hf_home/                     # metadata-only RLDX-1-VLM Hub cache

data/
└── EgoVLA_benchmark_rldx_v21/
```

Both asset directories are ignored by Git and must be prepared separately. The reserved `ckpt_name=pretrain` uses `pretrained_model_path`, which defaults to the checkpoint directory shown above. A base PT checkpoint normally lacks EgoVLA modality statistics, so real evaluation should use an EgoVLA fine-tuned checkpoint.

## Configuration

Model-specific `deploy.yml` keys:

| Key | Purpose |
| --- | --- |
| `embodiment_tag` | RLDX embodiment head; defaults to `GENERAL_EMBODIMENT`. |
| `device`, `strict`, `verbose` | Upstream `RLDXPolicy` runtime options. |
| `video_length`, `video_stride`, `execution_horizon` | Temporal input and returned action-chunk geometry. |
| `language_key` | RLDX language modality key. |
| `camera_map` | RLDX video keys mapped to XPolicyLab camera keys. |
| `model_path` | Optional explicit checkpoint root; highest resolver priority. |
| `pretrained_model_path` | Asset used by the reserved `pretrain`/`base` aliases. |
| `debug_no_model` | Explicitly enable dimension-only zero-action debug mode. |
| `request_timeout_s` | WebSocket request timeout for slow model startup/inference. |
| `max_connect_attempts`, `connect_retry_delay_s` | Policy-client connection retry policy. |
| `ws_ping_interval_s`, `ws_ping_timeout_s` | WebSocket keepalive settings. |

## Notes

- Only the dual-arm EgoVLA `joint` representation is supported; `ee` is rejected.
- `model.py` receives server-decoded RGB arrays. SparkArena checkpoints that were trained from OpenCV BGR frames enable a compatibility RGB-to-BGR swap; EgoVLA inputs remain RGB.
- A fine-tuned checkpoint must carry EgoVLA joint38 modality keys and normalization statistics; incompatible checkpoints fail fast.
- `install.sh` applies the tracked inference-precision patch to the pinned upstream checkout; pretrained weights remain unchanged.
- Full training and real checkpoint inference require the converted dataset, installed `uv` environment, and actual weights.
