"""比较 AUTO B_trunk 与完整 Find_1 的体素优化轨迹 JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


AUTO_TRUNK_SOURCE_IDENTITY = "7aae1f126ee0c1636a3f14455f8239a072ac28c9"
EXPECTED_VOXEL_PARAMETER_TENSORS = 353
EXPECTED_FULL_PARAMETER_TENSORS = 1594
EXPECTED_OTHER_PARAMETER_TENSORS = 1241
EXPECTED_VOXEL_PARAMETER_NAMES_SHA256 = (
    "57cb249c0410369a60c17b09950432a23069c541ac4ddf56e9ef4cf9e9467e2e"
)
EXPECTED_POINT_LOSS_NAMES = frozenset(("atom", "pseudo"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-full-source-identity", required=True)
    parser.add_argument("--expected-trunk-project-source-sha256", required=True)
    parser.add_argument("--expected-full-project-source-sha256", required=True)
    parser.add_argument(
        "--counterfactual-replay-comparison",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite(
    value: Any,
    *,
    path: str,
    mismatches: list[str],
) -> float:
    number = float(value)
    if not math.isfinite(number):
        mismatches.append(f"{path}: 非有限值 {value}")
    return number


def _compare_signature(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    path: str,
    atol: float,
    rtol: float,
    mismatches: list[str],
) -> None:
    if left is None or right is None:
        if left is not right:
            mismatches.append(f"{path}: 一侧为空")
        return
    for side, signature in (("trunk", left), ("full", right)):
        if not bool(signature["all_finite"]):
            mismatches.append(f"{path}.{side}: 张量包含 NaN 或 Inf")
    for key in ("shape", "dtype", "numel", "sha256"):
        if left[key] != right[key]:
            mismatches.append(f"{path}.{key}: {left[key]} != {right[key]}")
    for key in ("l2", "mean", "max_abs"):
        if not _close(float(left[key]), float(right[key]), atol=atol, rtol=rtol):
            mismatches.append(f"{path}.{key}: {left[key]} != {right[key]}")
    left_samples = list(left["samples"])
    right_samples = list(right["samples"])
    if len(left_samples) != len(right_samples):
        mismatches.append(f"{path}.samples: 长度不同")
        return
    for index, (left_value, right_value) in enumerate(
        zip(left_samples, right_samples, strict=True)
    ):
        if not _close(
            float(left_value),
            float(right_value),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(
                f"{path}.samples[{index}]: {left_value} != {right_value}"
            )


def _validate_signatures_finite(
    signatures: Mapping[str, Any],
    *,
    path: str,
    expected_count: int,
    mismatches: list[str],
) -> None:
    if len(signatures) != expected_count:
        mismatches.append(
            f"{path}: 参数数量 {len(signatures)} != {expected_count}"
        )
    for name, signature in signatures.items():
        if signature is None:
            continue
        if not bool(signature["all_finite"]):
            mismatches.append(f"{path}.{name}: 张量包含 NaN 或 Inf")
        for key in ("l2", "mean", "max_abs"):
            _require_finite(
                signature[key],
                path=f"{path}.{name}.{key}",
                mismatches=mismatches,
            )


def _compare_parameter_signatures(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    path: str,
    atol: float,
    rtol: float,
    mismatches: list[str],
) -> None:
    if set(left) != set(right):
        mismatches.append(f"{path}: 参数名称集合不同")
        return
    for name in sorted(left):
        _compare_signature(
            left[name],
            right[name],
            path=f"{path}.{name}",
            atol=atol,
            rtol=rtol,
            mismatches=mismatches,
        )


def _first_recycle_divergence(
    trunk_microbatches: Sequence[Mapping[str, Any]],
    full_microbatches: Sequence[Mapping[str, Any]],
) -> int | None:
    for trunk, full in zip(trunk_microbatches, full_microbatches, strict=True):
        if trunk["recycle_passes"] != full["recycle_passes"]:
            return int(trunk["microbatch"])
    return None


def _validate_trace_identity(
    trunk: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    expected_full_source_identity: str,
    expected_trunk_project_source_sha256: str,
    expected_full_project_source_sha256: str,
    mismatches: list[str],
) -> None:
    """核对两个独立进程、两个项目根和冻结提交身份."""

    if trunk.get("schema_version") != 2 or full.get("schema_version") != 2:
        mismatches.append("metadata.schema_version: 两条轨迹都必须为 2")
    expected_values = (
        ("trunk.role", trunk.get("role"), "trunk"),
        ("full.role", full.get("role"), "full"),
        ("trunk.experiment", trunk.get("experiment"), "CPC1/Find_1_trunk"),
        ("full.experiment", full.get("experiment"), "CPC1/Find_1"),
        (
            "trunk.source_identity",
            trunk.get("source_identity"),
            AUTO_TRUNK_SOURCE_IDENTITY,
        ),
        (
            "full.source_identity",
            full.get("source_identity"),
            expected_full_source_identity,
        ),
        (
            "trunk.project_source_sha256",
            trunk.get("project_source_sha256"),
            expected_trunk_project_source_sha256,
        ),
        (
            "full.project_source_sha256",
            full.get("project_source_sha256"),
            expected_full_project_source_sha256,
        ),
    )
    for name, actual, expected in expected_values:
        if actual != expected:
            mismatches.append(f"metadata.{name}: {actual} != {expected}")
    if trunk.get("project_root") == full.get("project_root"):
        mismatches.append("metadata.project_root: trunk 与 full 使用了同一目录")
    if trunk.get("project_source_sha256") == full.get("project_source_sha256"):
        mismatches.append("metadata.project_source_sha256: 两个项目源码摘要相同")
    if not trunk.get("project_source_sha256") or not full.get("project_source_sha256"):
        mismatches.append("metadata.project_source_sha256: 缺失")
    if trunk.get("process_nonce") == full.get("process_nonce"):
        mismatches.append("metadata.process_nonce: 两条轨迹不是独立进程产物")
    if not trunk.get("process_nonce") or not full.get("process_nonce"):
        mismatches.append("metadata.process_nonce: 缺失")
    if trunk.get("process_pid") == full.get("process_pid"):
        mismatches.append("metadata.process_pid: 两条轨迹来自同一进程")

    for key in (
        "track",
        "checkpoint_sha256",
        "checkpoint",
        "batch_size",
        "accumulate_steps",
        "optimizer_steps",
        "seed",
        "input_channels",
        "voxel_parameter_tensors",
        "voxel_parameter_names_sha256",
        "shared_state_count",
        "recycle_sequence_source_sha256",
    ):
        if trunk[key] != full[key]:
            mismatches.append(f"metadata.{key}: {trunk[key]} != {full[key]}")


def _validate_parameter_partition_and_probe(
    trunk: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    mismatches: list[str],
) -> None:
    """核对 353/1241 参数边界和每个点损失的严格零回灌."""

    expected_counts = (
        (
            "trunk.trainable_parameter_tensors",
            trunk,
            "trainable_parameter_tensors",
            EXPECTED_VOXEL_PARAMETER_TENSORS,
        ),
        (
            "trunk.voxel_parameter_tensors",
            trunk,
            "voxel_parameter_tensors",
            EXPECTED_VOXEL_PARAMETER_TENSORS,
        ),
        ("trunk.other_parameter_tensors", trunk, "other_parameter_tensors", 0),
        (
            "full.trainable_parameter_tensors",
            full,
            "trainable_parameter_tensors",
            EXPECTED_FULL_PARAMETER_TENSORS,
        ),
        (
            "full.voxel_parameter_tensors",
            full,
            "voxel_parameter_tensors",
            EXPECTED_VOXEL_PARAMETER_TENSORS,
        ),
        (
            "full.other_parameter_tensors",
            full,
            "other_parameter_tensors",
            EXPECTED_OTHER_PARAMETER_TENSORS,
        ),
    )
    for name, trace, key, expected in expected_counts:
        if trace.get(key) != expected:
            mismatches.append(f"metadata.{name}: {trace.get(key)} != {expected}")

    for role, trace in (("trunk", trunk), ("full", full)):
        if (
            trace.get("voxel_parameter_names_sha256")
            != EXPECTED_VOXEL_PARAMETER_NAMES_SHA256
        ):
            mismatches.append(f"metadata.{role}.voxel_parameter_names_sha256: 不匹配")
        probe = trace["point_to_voxel_gradient_probe"]
        names = set(probe["loss_term_names"])
        per_loss = probe["per_loss_voxel_gradient_norm"]
        if role == "full" and names != EXPECTED_POINT_LOSS_NAMES:
            mismatches.append(
                f"full.point_to_voxel_gradient_probe.loss_term_names: {sorted(names)}"
            )
        if set(per_loss) != names:
            mismatches.append(f"{role}.point_to_voxel_gradient_probe: 名称不一致")
        for loss_name, value in per_loss.items():
            gradient_norm = _require_finite(
                value,
                path=f"{role}.point_to_voxel_gradient_probe.{loss_name}",
                mismatches=mismatches,
            )
            if gradient_norm != 0.0:
                mismatches.append(
                    f"{role}.point_to_voxel_gradient_probe.{loss_name}: "
                    f"{gradient_norm} != 0"
                )


def _resolve_microbatch_boundary(
    trunk: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    counterfactual_replays: Sequence[Mapping[str, Any]],
    natural_trace_sha256s: Sequence[str],
    mismatches: list[str],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], int | None, int]:
    """确定自然 recycle 分离点并核对必要的双向反事实重放."""

    trunk_microbatches = list(trunk["microbatches"])
    full_microbatches = list(full["microbatches"])
    if len(trunk_microbatches) != len(full_microbatches):
        mismatches.append("microbatches: 数量不同")
        return trunk_microbatches, full_microbatches, None, 0

    recycle_divergence = _first_recycle_divergence(
        trunk_microbatches,
        full_microbatches,
    )
    if trunk["track"] in {"controlled", "replay"} and recycle_divergence is not None:
        mismatches.append(
            f"{trunk['track']}.recycle_passes: "
            f"第 {recycle_divergence} 个 microbatch 分离"
        )
    eligible_microbatches = (
        len(trunk_microbatches)
        if recycle_divergence is None
        else recycle_divergence
    )
    if recycle_divergence is not None and trunk["track"] == "natural":
        required_sources = set(natural_trace_sha256s)
        observed_sources = {
            str(comparison.get("recycle_sequence_source_sha256"))
            for comparison in counterfactual_replays
            if comparison.get("passed") and comparison.get("track") == "replay"
        }
        if len(required_sources) != 2 or observed_sources != required_sources:
            mismatches.append(
                "natural.counterfactual_replays: 必须由两条自然轨迹各自的 "
                "recycle 序列完成两次通过的 replay 比较"
            )
    return (
        trunk_microbatches,
        full_microbatches,
        recycle_divergence,
        eligible_microbatches,
    )


def _compare_microbatch(
    trunk_microbatch: Mapping[str, Any],
    full_microbatch: Mapping[str, Any],
    *,
    index: int,
    atol: float,
    rtol: float,
    mismatches: list[str],
) -> None:
    """比较一个 recycle 一致的 microbatch 及其完整体素梯度摘要."""

    if trunk_microbatch["request_positions"] != full_microbatch["request_positions"]:
        mismatches.append(f"microbatches[{index}].request_positions: 不同")
    if trunk_microbatch["requests"] != full_microbatch["requests"]:
        mismatches.append(f"microbatches[{index}].requests: 不同")
    for role, microbatch in (
        ("trunk", trunk_microbatch),
        ("full", full_microbatch),
    ):
        _require_finite(
            microbatch["total_loss"],
            path=f"microbatches[{index}].{role}.total_loss",
            mismatches=mismatches,
        )
        for loss_name, value in microbatch["loss_terms"].items():
            _require_finite(
                value,
                path=f"microbatches[{index}].{role}.loss_terms.{loss_name}",
                mismatches=mismatches,
            )
    _compare_parameter_signatures(
        trunk_microbatch["voxel_outputs"],
        full_microbatch["voxel_outputs"],
        path=f"microbatches[{index}].voxel_outputs",
        atol=atol,
        rtol=rtol,
        mismatches=mismatches,
    )
    trunk_terms = trunk_microbatch["loss_terms"]
    full_terms = full_microbatch["loss_terms"]
    if not EXPECTED_POINT_LOSS_NAMES.issubset(full_terms):
        mismatches.append(f"microbatches[{index}].full.loss_terms: 缺少 atom 或 pseudo")
    for loss_name, trunk_value in trunk_terms.items():
        if loss_name not in full_terms:
            mismatches.append(f"microbatches[{index}].loss_terms.{loss_name}: 缺失")
            continue
        if not _close(
            float(trunk_value),
            float(full_terms[loss_name]),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(
                f"microbatches[{index}].loss_terms.{loss_name}: "
                f"{trunk_value} != {full_terms[loss_name]}"
            )
    _compare_parameter_signatures(
        trunk_microbatch["accumulated_voxel_gradients"],
        full_microbatch["accumulated_voxel_gradients"],
        path=f"microbatches[{index}].accumulated_voxel_gradients",
        atol=atol,
        rtol=rtol,
        mismatches=mismatches,
    )
    if not _close(
        float(trunk_microbatch["accumulated_voxel_gradient_norm"]),
        float(full_microbatch["accumulated_voxel_gradient_norm"]),
        atol=atol,
        rtol=rtol,
    ):
        mismatches.append(f"microbatches[{index}].accumulated_voxel_gradient_norm: 不同")


def _validate_other_parameter_step(
    full_step: Mapping[str, Any],
    *,
    index: int,
    mismatches: list[str],
) -> None:
    """核对完整模型其他参数组的有限性、集合和 AdamW step."""

    other_keys = (
        "other_parameters_before_step",
        "other_pre_clip_gradients",
        "other_post_clip_gradients",
        "other_parameters_after_step",
        "other_parameter_updates",
        "other_exp_avg_after_step",
        "other_exp_avg_sq_after_step",
    )
    expected_names = set(full_step[other_keys[0]])
    if len(expected_names) != EXPECTED_OTHER_PARAMETER_TENSORS:
        mismatches.append(
            f"optimizer_step_records[{index}].full.other_parameters: "
            f"{len(expected_names)} != {EXPECTED_OTHER_PARAMETER_TENSORS}"
        )
    for key in other_keys:
        signatures = full_step[key]
        if set(signatures) != expected_names:
            mismatches.append(
                f"optimizer_step_records[{index}].full.{key}: 参数集合不同"
            )
        _validate_signatures_finite(
            signatures,
            path=f"optimizer_step_records[{index}].full.{key}",
            expected_count=EXPECTED_OTHER_PARAMETER_TENSORS,
            mismatches=mismatches,
        )
    if full_step["other_optimizer_state_step"] != [index + 1]:
        mismatches.append(
            f"optimizer_step_records[{index}].full.other_optimizer_state_step: "
            f"{full_step['other_optimizer_state_step']} != {[index + 1]}"
        )


def _compare_optimizer_step(
    trunk_step: Mapping[str, Any],
    full_step: Mapping[str, Any],
    *,
    index: int,
    atol: float,
    rtol: float,
    mismatches: list[str],
) -> None:
    """比较一个 optimizer step 的体素动力学并验收其他参数组."""

    scalar_keys = (
        "lr_before_step",
        "pre_clip_group_norm",
        "post_clip_group_norm",
        "voxel_clip_coefficient",
        "lr_after_step",
    )
    for key in scalar_keys:
        if not _close(
            float(trunk_step[key]),
            float(full_step[key]),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(f"optimizer_step_records[{index}].{key}: 不同")
    for role, step in (("trunk", trunk_step), ("full", full_step)):
        for key in scalar_keys:
            _require_finite(
                step[key],
                path=f"optimizer_step_records[{index}].{role}.{key}",
                mismatches=mismatches,
            )
        if float(step["post_clip_group_norm"]) > 0.50001:
            mismatches.append(
                f"optimizer_step_records[{index}].{role}.post_clip_group_norm: "
                f"{step['post_clip_group_norm']} > 0.5"
            )
    _require_finite(
        full_step["other_pre_clip_group_norm"],
        path=f"optimizer_step_records[{index}].full.other_pre_clip_group_norm",
        mismatches=mismatches,
    )
    other_post_clip_norm = _require_finite(
        full_step["other_post_clip_group_norm"],
        path=f"optimizer_step_records[{index}].full.other_post_clip_group_norm",
        mismatches=mismatches,
    )
    if other_post_clip_norm > 0.50001:
        mismatches.append(
            f"optimizer_step_records[{index}].full.other_post_clip_group_norm: "
            f"{full_step['other_post_clip_group_norm']} > 0.5"
        )
    if trunk_step["optimizer_state_step"] != full_step["optimizer_state_step"]:
        mismatches.append(f"optimizer_step_records[{index}].optimizer_state_step: 不同")
    if trunk_step["optimizer_state_step"] != [index + 1]:
        mismatches.append(
            f"optimizer_step_records[{index}].optimizer_state_step: "
            f"{trunk_step['optimizer_state_step']} != {[index + 1]}"
        )
    for key in (
        "parameters_before_step",
        "pre_clip_gradients",
        "post_clip_gradients",
        "parameters_after_step",
        "parameter_updates",
        "exp_avg_after_step",
        "exp_avg_sq_after_step",
    ):
        _compare_parameter_signatures(
            trunk_step[key],
            full_step[key],
            path=f"optimizer_step_records[{index}].{key}",
            atol=atol,
            rtol=rtol,
            mismatches=mismatches,
        )
    _validate_other_parameter_step(full_step, index=index, mismatches=mismatches)


def compare_traces(
    trunk: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
    expected_full_source_identity: str,
    expected_trunk_project_source_sha256: str,
    expected_full_project_source_sha256: str,
    counterfactual_replays: Sequence[Mapping[str, Any]] = (),
    natural_trace_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """比较两条独立轨迹，并用双向 replay 解释自然 recycle 分离."""

    mismatches: list[str] = []
    _validate_trace_identity(
        trunk,
        full,
        expected_full_source_identity=expected_full_source_identity,
        expected_trunk_project_source_sha256=expected_trunk_project_source_sha256,
        expected_full_project_source_sha256=expected_full_project_source_sha256,
        mismatches=mismatches,
    )
    _validate_parameter_partition_and_probe(trunk, full, mismatches=mismatches)
    (
        trunk_microbatches,
        full_microbatches,
        recycle_divergence,
        eligible_microbatches,
    ) = _resolve_microbatch_boundary(
        trunk,
        full,
        counterfactual_replays=counterfactual_replays,
        natural_trace_sha256s=natural_trace_sha256s,
        mismatches=mismatches,
    )
    for index in range(eligible_microbatches):
        _compare_microbatch(
            trunk_microbatches[index],
            full_microbatches[index],
            index=index,
            atol=atol,
            rtol=rtol,
            mismatches=mismatches,
        )

    trunk_steps = list(trunk["optimizer_step_records"])
    full_steps = list(full["optimizer_step_records"])
    if len(trunk_steps) != len(full_steps):
        mismatches.append("optimizer_step_records: 数量不同")
    else:
        for index, (trunk_step, full_step) in enumerate(
            zip(trunk_steps, full_steps, strict=True)
        ):
            if recycle_divergence is not None and int(trunk_step["last_microbatch"]) >= (
                recycle_divergence
            ):
                break
            _compare_optimizer_step(
                trunk_step,
                full_step,
                index=index,
                atol=atol,
                rtol=rtol,
                mismatches=mismatches,
            )

    return {
        "schema_version": 2,
        "passed": not mismatches,
        "track": trunk["track"],
        "atol": atol,
        "rtol": rtol,
        "first_recycle_divergence": recycle_divergence,
        "eligible_microbatches": eligible_microbatches,
        "recycle_sequence_source_sha256": trunk.get(
            "recycle_sequence_source_sha256"
        ),
        "counterfactual_replay_sources": sorted(
            str(comparison.get("recycle_sequence_source_sha256"))
            for comparison in counterfactual_replays
            if comparison.get("passed") and comparison.get("track") == "replay"
        ),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
    }


def main() -> None:
    arguments = build_parser().parse_args()
    trunk_path = arguments.trunk.resolve()
    full_path = arguments.full.resolve()
    if trunk_path == full_path:
        raise ValueError("trunk 与 full 不得指向同一个 JSON 文件。")
    trunk = json.loads(trunk_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    counterfactual_replays = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in arguments.counterfactual_replay_comparison
    ]
    result = compare_traces(
        trunk,
        full,
        atol=arguments.atol,
        rtol=arguments.rtol,
        expected_full_source_identity=arguments.expected_full_source_identity,
        expected_trunk_project_source_sha256=(
            arguments.expected_trunk_project_source_sha256
        ),
        expected_full_project_source_sha256=(
            arguments.expected_full_project_source_sha256
        ),
        counterfactual_replays=counterfactual_replays,
        natural_trace_sha256s=(
            _sha256_file(trunk_path),
            _sha256_file(full_path),
        ),
    )
    result["trunk_trace_sha256"] = _sha256_file(trunk_path)
    result["full_trace_sha256"] = _sha256_file(full_path)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, arguments.output)
    print(arguments.output)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
