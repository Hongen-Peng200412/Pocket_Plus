from __future__ import annotations

import copy
from pathlib import Path

import torch

from tmp.find1_optimization_dynamics.compare_optimization_traces import (
    AUTO_TRUNK_SOURCE_IDENTITY,
    EXPECTED_FULL_PARAMETER_TENSORS,
    EXPECTED_OTHER_PARAMETER_TENSORS,
    EXPECTED_VOXEL_LOSS_NAMES,
    EXPECTED_VOXEL_OUTPUT_NAMES,
    EXPECTED_VOXEL_PARAMETER_NAMES_SHA256,
    EXPECTED_VOXEL_PARAMETER_TENSORS,
    compare_traces,
)
from tmp.find1_optimization_dynamics.optimization_trace import (
    DenseTensorStore,
    tensor_signature,
)


FULL_SOURCE_IDENTITY = "f" * 40


def _signature(value: float) -> dict[str, object]:
    return tensor_signature(
        torch.tensor([value], dtype=torch.float32),
        include_dense=True,
    )


def _signatures(prefix: str, count: int, value: float) -> dict[str, object]:
    return {f"{prefix}.{index:04d}": _signature(value) for index in range(count)}


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

    voxel_losses = {name: 0.2 for name in EXPECTED_VOXEL_LOSS_NAMES}
    voxel_outputs = {name: _signature(0.2) for name in EXPECTED_VOXEL_OUTPUT_NAMES}
    microbatch_template = {
        "optimizer_step": 0,
        "loss_terms": {**voxel_losses, **point_losses},
        "voxel_outputs": voxel_outputs,
        "accumulated_voxel_gradient_norm": 0.1,
        "accumulated_voxel_gradients": copy.deepcopy(voxel_trace),
    }
    return {
        "schema_version": 4,
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
        "batch_size": 1,
        "accumulate_steps": 2,
        "optimizer_steps": 1,
        "seed": 3407,
        "input_channels": 56,
        "trainable_parameter_tensors": trainable_count,
        "voxel_parameter_tensors": EXPECTED_VOXEL_PARAMETER_TENSORS,
        "other_parameter_tensors": other_count,
        "voxel_parameter_names_sha256": EXPECTED_VOXEL_PARAMETER_NAMES_SHA256,
        "dense_tensor_store": {
            "version": 1,
            "file_name": None,
            "dtype": "float32-le",
            "total_bytes": 0,
            "captured_signature_count": 0,
            "unique_sidecar_tensor_count": 0,
        },
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
                "voxel_outputs": copy.deepcopy(voxel_outputs),
                "accumulated_voxel_gradients": copy.deepcopy(voxel_trace),
            },
            {
                **microbatch_template,
                "microbatch": 1,
                "request_positions": [2],
                "requests": [{"pdb_id": "2def", "box_start_zyx": [1, 1, 1]}],
                "recycle_passes": 2,
                "total_loss": 0.3 + sum(point_losses.values()),
                "loss_terms": {
                    **{name: 0.3 for name in EXPECTED_VOXEL_LOSS_NAMES},
                    **point_losses,
                },
                "voxel_outputs": {
                    name: _signature(0.3) for name in EXPECTED_VOXEL_OUTPUT_NAMES
                },
                "accumulated_voxel_gradient_norm": 0.4,
                "accumulated_voxel_gradients": copy.deepcopy(voxel_trace),
            },
        ],
        "optimizer_step_records": [
            {
                "optimizer_step": 0,
                "last_microbatch": 1,
                "lr_before_step": 1.0e-5,
                "pre_clip_group_norm": 0.6,
                "other_pre_clip_group_norm": 0.7 if role == "full" else 0.0,
                "post_clip_group_norm": 0.5,
                "other_post_clip_group_norm": 0.5 if role == "full" else 0.0,
                "voxel_clip_coefficient": 5 / 6,
                "lr_after_step": 2.0e-5,
                "optimizer_state_step": {name: 1 for name in voxel_trace},
                "other_optimizer_state_step": {name: 1 for name in other_trace},
                "parameters_before_step": copy.deepcopy(voxel_trace),
                "pre_clip_gradients": copy.deepcopy(voxel_trace),
                "post_clip_gradients": copy.deepcopy(voxel_trace),
                "parameters_after_step": copy.deepcopy(voxel_trace),
                "parameter_updates": copy.deepcopy(voxel_trace),
                "exp_avg_after_step": copy.deepcopy(voxel_trace),
                "exp_avg_sq_after_step": copy.deepcopy(voxel_trace),
                "other_parameters_before_step": copy.deepcopy(other_trace),
                "other_pre_clip_gradients": copy.deepcopy(other_trace),
                "other_post_clip_gradients": copy.deepcopy(other_trace),
                "other_parameters_after_step": copy.deepcopy(other_trace),
                "other_parameter_updates": copy.deepcopy(other_trace),
                "other_exp_avg_after_step": copy.deepcopy(other_trace),
                "other_exp_avg_sq_after_step": copy.deepcopy(other_trace),
            }
        ],
    }


def _compare(
    trunk: dict[str, object],
    full: dict[str, object],
    *,
    counterfactual_replays: tuple[dict[str, object], ...] = (),
    natural_trace_sha256s: tuple[str, ...] = (),
    trunk_dense_file: Path | None = None,
    full_dense_file: Path | None = None,
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
        trunk_dense_file=trunk_dense_file,
        full_dense_file=full_dense_file,
    )


def _replay_comparison(source_sha256: str, suffix: str) -> dict[str, object]:
    trunk = _trace(role="trunk", track="replay")
    full = _trace(role="full", track="replay")
    trunk["recycle_sequence_source_sha256"] = source_sha256
    full["recycle_sequence_source_sha256"] = source_sha256
    result = _compare(trunk, full)
    assert result["passed"]
    result["trunk_trace_sha256"] = suffix * 64
    result["full_trace_sha256"] = chr(ord(suffix) + 1) * 64
    return result


def _append_second_optimizer_step(
    trace: dict[str, object],
    *,
    inactive_voxel_name: str,
) -> None:
    """把合成轨迹扩为两步，并让一个参数在第二步合法保持旧 state。"""

    trace["optimizer_steps"] = 2
    for microbatch_index in (2, 3):
        microbatch = copy.deepcopy(trace["microbatches"][microbatch_index - 2])
        microbatch["microbatch"] = microbatch_index
        microbatch["optimizer_step"] = 1
        microbatch["request_positions"] = [microbatch_index + 1]
        microbatch["requests"] = [
            {
                "pdb_id": f"{microbatch_index}xyz",
                "box_start_zyx": [microbatch_index] * 3,
            }
        ]
        microbatch["accumulated_voxel_gradients"][inactive_voxel_name] = None
        trace["microbatches"].append(microbatch)
        trace["recycle_sequence"].append(microbatch["recycle_passes"])

    step = copy.deepcopy(trace["optimizer_step_records"][0])
    step["optimizer_step"] = 1
    step["last_microbatch"] = 3
    step["lr_before_step"] = 2.0e-5
    step["lr_after_step"] = 3.0e-5
    step["pre_clip_gradients"][inactive_voxel_name] = None
    step["post_clip_gradients"][inactive_voxel_name] = None
    for name in step["optimizer_state_step"]:
        step["optimizer_state_step"][name] = 1 if name == inactive_voxel_name else 2
    for name in step["other_optimizer_state_step"]:
        step["other_optimizer_state_step"][name] = 2
    trace["optimizer_step_records"].append(step)


def test_common_initial_parameters_require_exact_bytes() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    assert _compare(trunk, full)["passed"]

    full["optimizer_step_records"][0]["parameters_before_step"][
        "backbone.voxel_backbone.parameter.0000"
    ]["sha256"] = "changed"
    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("parameters_before_step" in item for item in result["mismatches"])


def test_common_initial_parameters_cannot_both_be_empty() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = "backbone.voxel_backbone.parameter.0000"
    for trace in (trunk, full):
        trace["optimizer_step_records"][0]["parameters_before_step"][
            parameter_name
        ] = None

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("两侧均为空" in item for item in result["mismatches"])


def test_noninitial_sha_difference_inside_numerical_envelope_is_reported() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = "backbone.voxel_backbone.parameter.0000"
    step = full["optimizer_step_records"][0]
    parameters_after_step = dict(step["parameters_after_step"])
    signature = parameters_after_step[parameter_name]
    parameters_after_step[parameter_name] = {
        **signature,
        "sha256": "changed",
    }
    step["parameters_after_step"] = parameters_after_step

    result = _compare(trunk, full)

    assert result["passed"]
    diagnostics = result["numerical_diagnostics"]["optimizer_signatures"]
    assert diagnostics["parameters_after_step"]["sha256_mismatch_count"] == 1


def test_optimizer_state_step_preserves_matching_inactive_parameters() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = next(
        iter(trunk["optimizer_step_records"][0]["optimizer_state_step"])
    )
    for trace in (trunk, full):
        step = trace["optimizer_step_records"][0]
        step["optimizer_state_step"][parameter_name] = None
        step["pre_clip_gradients"][parameter_name] = None
        step["post_clip_gradients"][parameter_name] = None
        step["exp_avg_after_step"][parameter_name] = None
        step["exp_avg_sq_after_step"][parameter_name] = None
        for microbatch in trace["microbatches"]:
            microbatch["accumulated_voxel_gradients"][parameter_name] = None

    assert _compare(trunk, full)["passed"]

    full["optimizer_step_records"][0]["optimizer_state_step"][parameter_name] = 1
    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("optimizer_state_step" in item for item in result["mismatches"])


def test_accumulated_gradient_null_requires_matching_pre_clip_gradient() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = next(
        iter(trunk["optimizer_step_records"][0]["optimizer_state_step"])
    )
    for trace in (trunk, full):
        trace["microbatches"][-1]["accumulated_voxel_gradients"][parameter_name] = None

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("空值集合不同" in item for item in result["mismatches"])


def test_adamw_state_step_persists_when_second_step_has_no_gradient() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    parameter_name = next(
        iter(trunk["optimizer_step_records"][0]["optimizer_state_step"])
    )
    _append_second_optimizer_step(trunk, inactive_voxel_name=parameter_name)
    _append_second_optimizer_step(full, inactive_voxel_name=parameter_name)

    result = _compare(trunk, full)

    assert result["passed"], result["mismatches"]


def test_trace_cannot_omit_expected_microbatches() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    for trace in (trunk, full):
        trace["microbatches"].pop()
        trace["recycle_sequence"].pop()

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("microbatches: 1 != 2" in item for item in result["mismatches"])


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
        "full.project_source_sha256" in mismatch for mismatch in result["mismatches"]
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
    signature = full["microbatches"][0]["voxel_outputs"]["voxel_logits_ligand"]
    signature.update(
        {
            "sha256": "changed",
            "l2": 2.0,
            "mean": 2.0,
            "max_abs": 2.0,
            "samples": [2.0],
        }
    )

    result = _compare(trunk, full)
    assert result["passed"] is False
    assert any("numerical.voxel_outputs" in item for item in result["mismatches"])


def test_bf16_output_difference_inside_envelope_is_reported_and_accepted() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    voxel_outputs = full["microbatches"][0]["voxel_outputs"]
    full["microbatches"][0]["voxel_outputs"] = {
        **voxel_outputs,
        "voxel_logits_ligand": _signature(0.20005),
    }

    result = _compare(trunk, full)

    assert result["passed"]
    diagnostics = result["numerical_diagnostics"]["voxel_outputs"]
    assert diagnostics["sha256_mismatch_count"] == 1


def test_gradient_distribution_outside_envelope_is_a_hard_failure() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    gradients = full["microbatches"][0]["accumulated_voxel_gradients"]
    for name in sorted(gradients)[:25]:
        gradients[name] = _signature(1.0)

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any(
        "microbatches[0].numerical.accumulated_voxel_gradients" in item
        for item in result["mismatches"]
    )


def test_unsampled_permutation_is_rejected_by_dense_element_comparison(
    tmp_path: Path,
) -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    left_tensor = torch.zeros(1_000_000, dtype=torch.float32)
    left_tensor[9004] = 50.0
    left_tensor[22749] = -50.0
    right_tensor = left_tensor.clone()
    right_tensor[9004] = -50.0
    right_tensor[22749] = 50.0
    left_store = DenseTensorStore(tmp_path / "left.json")
    right_store = DenseTensorStore(tmp_path / "right.json")
    left_signature = tensor_signature(
        left_tensor,
        dense_store=left_store,
        include_dense=True,
    )
    right_signature = tensor_signature(
        right_tensor,
        dense_store=right_store,
        include_dense=True,
    )
    trunk["dense_tensor_store"] = left_store.metadata()
    full["dense_tensor_store"] = right_store.metadata()
    left_store.publish()
    right_store.publish()
    assert left_signature["samples"] == right_signature["samples"]
    assert left_signature["l2"] == right_signature["l2"]
    assert left_signature["mean"] == right_signature["mean"]
    assert left_signature["max_abs"] == right_signature["max_abs"]
    trunk["microbatches"][0]["voxel_outputs"]["voxel_logits_ligand"] = left_signature
    full["microbatches"][0]["voxel_outputs"]["voxel_logits_ligand"] = right_signature

    result = _compare(
        trunk,
        full,
        trunk_dense_file=left_store.final_path,
        full_dense_file=right_store.final_path,
    )

    assert result["passed"] is False
    assert any("element_absolute_max" in item for item in result["mismatches"])


def test_common_signature_cannot_omit_dense_values() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    signature = full["microbatches"][0]["voxel_outputs"]["voxel_logits_ligand"]
    del signature["dense_float32"]

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("缺少 dense_float32" in item for item in result["mismatches"])


def test_extreme_finite_loss_is_a_hard_failure() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    for trace in (trunk, full):
        trace["microbatches"][0]["loss_terms"]["voxel_ligand"] = 1.0e20
        trace["microbatches"][0]["total_loss"] = 1.0e20

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("不在 [0, 10.0]" in item for item in result["mismatches"])


def test_unexpected_full_loss_term_is_a_hard_failure() -> None:
    trunk = _trace(role="trunk")
    full = _trace(role="full")
    full["microbatches"][0]["loss_terms"]["unexpected"] = 0.0

    result = _compare(trunk, full)

    assert result["passed"] is False
    assert any("full.loss_terms" in item for item in result["mismatches"])


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

    trunk_sha256 = "1" * 64
    full_sha256 = "2" * 64
    replay_results = (
        _replay_comparison(trunk_sha256, "3"),
        _replay_comparison(full_sha256, "5"),
    )
    with_replay = _compare(
        trunk,
        full,
        counterfactual_replays=replay_results,
        natural_trace_sha256s=(trunk_sha256, full_sha256),
    )
    assert with_replay["passed"]
    assert with_replay["first_recycle_divergence"] == 0
    assert with_replay["eligible_microbatches"] == 0


def test_natural_replay_identity_must_match_current_traces() -> None:
    trunk = _trace(role="trunk", track="natural")
    full = _trace(role="full", track="natural")
    full["microbatches"][0]["recycle_passes"] = 3
    full["recycle_sequence"][0] = 3
    trunk_sha256 = "1" * 64
    full_sha256 = "2" * 64
    replay_results = [
        _replay_comparison(trunk_sha256, "3"),
        _replay_comparison(full_sha256, "5"),
    ]
    replay_results[0]["trace_identity"]["trunk"]["checkpoint_sha256"] = "wrong"

    result = _compare(
        trunk,
        full,
        counterfactual_replays=tuple(replay_results),
        natural_trace_sha256s=(trunk_sha256, full_sha256),
    )

    assert result["passed"] is False
    assert any("trace_identity" in item for item in result["mismatches"])


def test_natural_replay_must_record_both_trace_hashes() -> None:
    trunk = _trace(role="trunk", track="natural")
    full = _trace(role="full", track="natural")
    full["microbatches"][0]["recycle_passes"] = 3
    full["recycle_sequence"][0] = 3
    trunk_sha256 = "1" * 64
    full_sha256 = "2" * 64
    replay_results = [
        _replay_comparison(trunk_sha256, "3"),
        _replay_comparison(full_sha256, "5"),
    ]
    del replay_results[0]["trunk_trace_sha256"]

    result = _compare(
        trunk,
        full,
        counterfactual_replays=tuple(replay_results),
        natural_trace_sha256s=(trunk_sha256, full_sha256),
    )

    assert result["passed"] is False
    assert any("trace_sha256" in item for item in result["mismatches"])


def test_natural_divergence_cannot_hide_extreme_loss() -> None:
    trunk = _trace(role="trunk", track="natural")
    full = _trace(role="full", track="natural")
    full["microbatches"][0]["recycle_passes"] = 3
    full["recycle_sequence"][0] = 3
    for trace in (trunk, full):
        trace["microbatches"][1]["loss_terms"]["voxel_ligand"] = 1.0e20
        trace["microbatches"][1]["total_loss"] = 1.0e20
    trunk_sha256 = "1" * 64
    full_sha256 = "2" * 64
    replay_results = (
        _replay_comparison(trunk_sha256, "3"),
        _replay_comparison(full_sha256, "5"),
    )

    result = _compare(
        trunk,
        full,
        counterfactual_replays=replay_results,
        natural_trace_sha256s=(trunk_sha256, full_sha256),
    )

    assert result["passed"] is False
    assert any("不在 [0, 10.0]" in item for item in result["mismatches"])
