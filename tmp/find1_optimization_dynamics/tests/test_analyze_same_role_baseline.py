"""验证同角色 natural/replay 校准器的身份边界。"""

from __future__ import annotations

from copy import deepcopy

from tmp.find1_optimization_dynamics.analyze_same_role_baseline import (
    validate_trace_pair,
)


def _trace_pair() -> tuple[dict[str, object], dict[str, object]]:
    natural = {
        "schema_version": 4,
        "track": "natural",
        "role": "trunk",
        "experiment": "CPC1/Find_1_trunk",
        "source_identity": "source",
        "project_root": "/project",
        "project_source_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
        "checkpoint": "/checkpoint",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_global_step": 10,
        "checkpoint_epoch": 0,
        "batch_size": 6,
        "accumulate_steps": 8,
        "optimizer_steps": 2,
        "seed": 3407,
        "input_channels": 107,
        "trainable_parameter_tensors": 353,
        "voxel_parameter_tensors": 353,
        "other_parameter_tensors": 0,
        "voxel_parameter_names_sha256": "d" * 64,
        "shared_state_count": 353,
        "process_nonce": "natural-process",
        "process_pid": 101,
        "recycle_sequence": [1, 2, 3],
        "recycle_sequence_source_sha256": None,
    }
    replay = deepcopy(natural)
    replay.update(
        {
            "track": "replay",
            "process_nonce": "replay-process",
            "process_pid": 202,
            "recycle_sequence_source_sha256": "e" * 64,
        }
    )
    return natural, replay


def test_same_role_pair_accepts_independent_replay_of_natural_sequence() -> None:
    natural, replay = _trace_pair()
    assert validate_trace_pair(natural, replay, natural_sha256="e" * 64) == []


def test_same_role_pair_rejects_wrong_recycle_source() -> None:
    natural, replay = _trace_pair()
    mismatches = validate_trace_pair(natural, replay, natural_sha256="f" * 64)
    assert mismatches == ["replay 没有引用当前 natural JSON 的真实 SHA-256"]
