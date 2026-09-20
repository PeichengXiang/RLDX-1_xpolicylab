"""Load the canonical task and joint contracts from the EgoVLA benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType


JOINT_GROUPS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def load_schema(path: Path) -> tuple[dict[str, str], tuple[int, ...], str]:
    """Return prompts, policy joint indices, and SHA256 from benchmark schema.py."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("egovla_benchmark_schema", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load EgoVLA schema: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _validate_module(module, path)
    prompts = {str(key): str(value) for key, value in module.TASK_INSTRUCTIONS.items()}
    indices = tuple(
        int(index)
        for group in JOINT_GROUPS
        for index in module.JOINT_INDICES[group]
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return prompts, indices, digest


def _validate_module(module: ModuleType, path: Path) -> None:
    prompts = getattr(module, "TASK_INSTRUCTIONS", None)
    indices = getattr(module, "JOINT_INDICES", None)
    if not isinstance(prompts, dict) or len(prompts) != 12:
        raise ValueError(f"{path}: expected 12 canonical task instructions")
    if len(set(prompts)) != 12 or len(set(prompts.values())) != 12:
        raise ValueError(f"{path}: task names and instructions must be unique")
    if not isinstance(indices, dict) or any(group not in indices for group in JOINT_GROUPS):
        raise ValueError(f"{path}: missing canonical joint groups")
    flattened = [int(index) for group in JOINT_GROUPS for index in indices[group]]
    if [len(indices[group]) for group in JOINT_GROUPS] != [7, 12, 7, 12]:
        raise ValueError(f"{path}: expected joint groups 7+12+7+12")
    if len(flattened) != 38 or len(set(flattened)) != 38:
        raise ValueError(f"{path}: canonical policy indices are not 38 unique joints")
    if min(flattened) < 0 or max(flattened) >= 50:
        raise ValueError(f"{path}: canonical policy indices fall outside the native 50D action")
