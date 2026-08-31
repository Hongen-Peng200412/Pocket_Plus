from __future__ import annotations

import math

from tmp.find1_optimization_dynamics.compare_optimization_traces import (
    AUTO_TRUNK_SOURCE_IDENTITY,
    EXPECTED_FULL_PARAMETER_TENSORS,
    EXPECTED_OTHER_PARAMETER_TENSORS,
    EXPECTED_VOXEL_PARAMETER_NAMES_SHA256,
    EXPECTED_VOXEL_PARAMETER_TENSORS,
    compare_traces,
)


FULL_SOURCE_IDENTITY = "f" * 40


def _signature(value: float) -> dict[str, object]:
    return {
        "shape": [1],
        "dtype": "torch.float32",
        "numel": 1,
        "sha256": f"sha256-{value!r}",
        "all_finite": math.isfinite(value),
        "l2": abs(value),
        "mean": value,
        "max_abs": abs(value),
        "samples": [value],
    }


def _signatures(prefix: str, count: int, value: float) -> dict[str, object]:
    return {
        f"{prefix}.{index:04d}": _signature(value)
        for index in range(count)
    }


def _trace(*, role: str, track: str = "controlled") -> dict[str, object]:
    voxel_trace = _signatures(
        "backbone.voxel_backbone.parameter",
        EXPECTED_VOXEL_PARAMETER_TENSORS,
        0.25,
    )
    if role == "full":
        other_trace = _signatures(
            "backbone.point_parameter",
            EXPECTED_OTHER_PARAMETER_TENSORS,
            0.125,
        )
        point_losses = {"atom": 0.4, "pseudo": 0.5}
        source_identity = FULL_SOURCE_IDENTITY
        experiment = "CPC1/Find_1"
        project_root = "/release/full"
        source_sha256 = "full-source"
        process_nonce = "full-process"
        trainable_count = EXPECTED_FULL_PARAMETER_TENSORS
        other_count = EXPECTED_OTHER_PARAMETER_TENSORS
    else:
        other_trace = {}
        point_losses = {}
        source_identity = AUTO_TRUNK_SOURCE_IDENTITY
        experiment = "CPC1/Find_1_trunk"
        project_root = "/release/trunk"
        source_sha256 = "trunk-source"
        process_nonce = "trunk-process"
        trainable_count = EXPECTED_VOXEL_PARAMETER_TENSORS
        other_count = 0

    microbatch_template = {
        "optimizer_step": 0,
        "loss_terms": {"voxel_ligand": 0.2, **point_losses},
        "voxel_outputs": {"voxel_logits_ligand": _signature(0.2)},
        "accumulated_voxel_gradient_norm": 0.1,
        "accumulated_voxel_gradients": voxel_trace,
    }
    return {
        "schema_version": 2,
        "role": role,
        "track": track,
        "source_identity": source_identity,
        "process_nonce": process_nonce,
        "process_pid": 100 if role == "trunk" else 200,
        "project_root": project_root,
        "project_source_sha256": source_sha256,
        "experiment": experiment,
        "resolved_config_sha256": f"{role}-config",
        "checkpoint": "/checkpoint/BEST.ckpt",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_global_step": 2018,
        "checkpoint_epoch": 0,
        "shared_state_count": 2000,
        "batch_size": 6,
        "accumulate_steps": 2,
        "optimizer_steps": 1,
        "seed": 3407,
        "input_channels": 56,
        "trainable_parameter_tensors": trainable_count,
        "voxel_parameter_tensors": EXPECTED_VOXEL_PARAMETER_TENSORS,
        "other_parameter_tensors": other_count,
        "voxel_parameter_names_sha256": EXPECTED_VOXEL_PARAMETER_NAMES_SHA256,
        "point_to_voxel_gradient_probe": {
            "loss_term_names": ["atom", "pseudo"],
            "per_loss_voxel_gradient_norm": {"atom": 0.0, "pseudo": 0.0},
        },
        "recycle_sequence_source_sha256": None,
        "recycle_sequence": [1, 2],
        "microbatches": [
            {
                **microbatch_template,
                "microbatch": 0,
                "request_positions": [1],
                "requests": [{"pdb_id": "1abc", "box_start_zyx": [0, 0, 0]}],
                "recycle_passes": 1,
                "total_loss": 0.2 + sum(point_losses.values()),
            },
            {
                **microbatch_template,
                "microbatch": 1,
                "request_positions": [2],
                "requests": [{"pdb_id": "2def", "box_start_zyx": [1, 1, 1]}],
                "recycle_passes": 2,
                "total_loss": 0.3 + sum(point_losses.values()),
                "loss_terms": {"voxel_ligand": 0.3, **point_losses},
                "voxel_outputs": {"voxel_logits_ligand": _signature(0.3)},
                "accumulated_voxel_gradient_norm": 0.4,
            },
        ],
        "optimizer_step_records": [
            {
                "last_microbatch": 1,
                "lr_before_step": 1.0e-5,
                "pre_clip_group_norm": 0.6,
                "other_pre_clip_group_norm": 0.7 if role == "full" else 0.0,
                "post_clip_group_norm": 0.5,
                "other_post_clip_group_norm": 0.5 if role == "full" else 0.0,
                "voxel_clip_coefficient": 5 / 6,
                "lr_after_step": 2.0e-5,
                "optimizer_state_step": {
                    name: 1 for name in voxel_trace
                },
                "other_optimizer_state_step": {
                    name: 1 for name in other_trace
                },
                "parameters_before_step": voxel_trace,
                "pre_clip_gradients": voxel_trace,
                "post_clip_gradients": voxel_trace,
                "parameters_after_step": voxel_trace,
                "parameter_updates": voxel_trace,
                "exp_avg_after_step": voxel_trace,
                "exp_avg_sq_after_step": voxel_trace,
                "other_parameters_before_step": other_trace,
                "other_pre_clip_gradients": other_trace,
                "other_post_clip_gradients": other_trace,
                "other_parameters_after_step": other_trace,
                "other_parameter_updates": other_trace,
                "other_exp_avg_after_step": other_trace,
                "other_exp_avg_sq_after_step": other_trace,
            }
        ],
    }


def _compare(
    trunk: dict[str, object],
    full: dict[str, object],
    *,
    counterfactual_replays: tuple[dict[str, object], ...] = (),
    natural_trace_sha256s: tuple[str, ...] = (),
) -> dict[str, object]:
    return compare_traces(
        trunk,
        full,
        atol=1.0e-6,
        rtol=1.0e-5,
        expected_full_source_identity=FULL_SOURCE_IDENTITY,
        expected_trunk_project_source_sha256="trunk-source",
        expected_full_project_source_sha256="full-source",
        counterfactual_replays=counterfactual_replays,
        natural_trace_sha256s=natural_trace_sha256s,
    )


def test_controlled_trace_requires_full_optimizer_equivalence() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    assert _compare(trunk, full)["passed"]

    full["optimizer_step_records"][0]["parameters_after_step"][
        "backbone.voxel_backbone.parameter.0000"
    ]["sha256"] = "changed"
    result = _compare(trunk, full)
    assert result["passed"] is False
    assert result["mismatch_count"] >= 1


def test_optimizer_state_step_preserves_matching_inactive_parameters() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = next(
        iter(trunk["optimizer_step_records"][0]["optimizer_state_step"])
    )
    for trace in (trunk, full):
        step = trace["optimizer_step_records"][0]
        step["optimizer_state_step"][parameter_name] = None
        step["exp_avg_after_step"][parameter_name] = None
        step["exp_avg_sq_after_step"][parameter_name] = None

    assert _compare(trunk, full)["passed"]

    full["optimizer_step_records"][0]["optimizer_state_step"][
        parameter_name
    ] = 1
    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("optimizer_state_step" in item for item in result["mismatches"])


def test_identity_cannot_reuse_the_same_process_product() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["process_nonce"] = trunk["process_nonce"]

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("process_nonce" in mismatch for mismatch in result["mismatches"])


def test_identity_cannot_reuse_the_same_process_pid() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["process_pid"] = trunk["process_pid"]

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("process_pid" in mismatch for mismatch in result["mismatches"])


def test_identity_requires_the_reviewed_project_source_digest() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["project_source_sha256"] = "wrong-full-source"

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any(
        "full.project_source_sha256" in mismatch
        for mismatch in result["mismatches"]
    )


def test_each_point_loss_gradient_leak_is_a_hard_failure() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["point_to_voxel_gradient_probe"]["per_loss_voxel_gradient_norm"][
        "atom"
    ] = 1.0e-12

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any(
        "point_to_voxel_gradient_probe.atom" in mismatch
        for mismatch in result["mismatches"]
    )


def test_missing_point_loss_and_nonfinite_other_group_are_hard_failures() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    del full["microbatches"][0]["loss_terms"]["pseudo"]
    full["optimizer_step_records"][0]["other_post_clip_group_norm"] = float("nan")

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("缺少 atom 或 pseudo" in item for item in result["mismatches"])
    assert any("非有限值" in item for item in result["mismatches"])


def test_voxel_output_difference_is_a_hard_failure() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["microbatches"][0]["voxel_outputs"]["voxel_logits_ligand"][
        "sha256"
    ] = "changed"

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("voxel_outputs" in mismatch for mismatch in result["mismatches"])


def test_natural_divergence_requires_both_counterfactual_replays() -> None:
    trunk = _trace(role="trunk", track="natural")
    full = _trace(role="full", track="natural")
    full["microbatches"][0]["recycle_passes"] = 3
    full["recycle_sequence"][0] = 3

    without_replay = _compare(
        trunk,
        full,
        natural_trace_sha256s=("trunk-natural", "full-natural"),
    )
    assert without_replay["passed"] is False

    replay_results = (
        {
            "passed": True,
            "track": "replay",
            "recycle_sequence_source_sha256": "trunk-natural",
        },
        {
            "passed": True,
            "track": "replay",
            "recycle_sequence_source_sha256": "full-natural",
        },
    )
    with_replay = _compare(
        trunk,
        full,
        counterfactual_replays=replay_results,
        natural_trace_sha256s=("trunk-natural", "full-natural"),
    )
    assert with_replay["passed"]
    assert with_replay["first_recycle_divergence"] == 0
    assert with_replay["eligible_microbatches"] == 0
