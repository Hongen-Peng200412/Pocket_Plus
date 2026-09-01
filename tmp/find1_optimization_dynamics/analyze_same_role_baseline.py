"""量化同一模型在独立进程中重放相同 recycle 序列的数值差异。"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tmp.find1_optimization_dynamics.compare_optimization_traces import (
        DENSE_TENSOR_STORE_VERSION,
        EXPECTED_VOXEL_OUTPUT_NAMES,
        EXPECTED_VOXEL_PARAMETER_TENSORS,
        OPTIMIZER_SIGNATURE_LIMITS,
        TRACE_SCHEMA_VERSION,
        DenseTensorReader,
        SignatureDivergence,
        _compare_parameter_signatures,
        _sha256_file,
        _validate_dense_tensor_store,
    )
except ModuleNotFoundError:
    from compare_optimization_traces import (
        DENSE_TENSOR_STORE_VERSION,
        EXPECTED_VOXEL_OUTPUT_NAMES,
        EXPECTED_VOXEL_PARAMETER_TENSORS,
        OPTIMIZER_SIGNATURE_LIMITS,
        TRACE_SCHEMA_VERSION,
        DenseTensorReader,
        SignatureDivergence,
        _compare_parameter_signatures,
        _sha256_file,
        _validate_dense_tensor_store,
    )


TRACE_IDENTITY_FIELDS = (
    "role",
    "experiment",
    "source_identity",
    "project_root",
    "project_source_sha256",
    "resolved_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_global_step",
    "checkpoint_epoch",
    "batch_size",
    "accumulate_steps",
    "optimizer_steps",
    "seed",
    "input_channels",
    "trainable_parameter_tensors",
    "voxel_parameter_tensors",
    "other_parameter_tensors",
    "voxel_parameter_names_sha256",
    "shared_state_count",
)
GRADIENT_SIGNATURE_FIELDS = {
    "pre_clip_gradients",
    "post_clip_gradients",
    "exp_avg_after_step",
    "exp_avg_sq_after_step",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve_dense_file(
    trace_path: Path,
    trace: Mapping[str, Any],
) -> Path | None:
    record = trace.get("dense_tensor_store")
    if not isinstance(record, Mapping):
        return None
    if int(record.get("unique_sidecar_tensor_count", 0)) == 0:
        return None
    file_name = record.get("file_name")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        return None
    return trace_path.parent / file_name


def validate_trace_pair(
    natural: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    natural_sha256: str,
) -> list[str]:
    """核对同角色 natural/replay 是否构成可解释的独立进程重复。"""

    mismatches: list[str] = []
    if natural.get("schema_version") != TRACE_SCHEMA_VERSION:
        mismatches.append("natural.schema_version 不是当前轨迹版本")
    if replay.get("schema_version") != TRACE_SCHEMA_VERSION:
        mismatches.append("replay.schema_version 不是当前轨迹版本")
    if natural.get("track") != "natural":
        mismatches.append("natural.track 必须为 natural")
    if natural.get("recycle_sequence_source_sha256") is not None:
        mismatches.append("natural 不得引用 recycle 序列来源")
    if replay.get("track") != "replay":
        mismatches.append("replay.track 必须为 replay")
    if replay.get("recycle_sequence_source_sha256") != natural_sha256:
        mismatches.append("replay 没有引用当前 natural JSON 的真实 SHA-256")
    if natural.get("recycle_sequence") != replay.get("recycle_sequence"):
        mismatches.append("natural 与 replay 的 recycle 序列不同")
    for field_name in TRACE_IDENTITY_FIELDS:
        if natural.get(field_name) != replay.get(field_name):
            mismatches.append(f"运行身份字段 {field_name} 不同")
    if natural.get("process_nonce") == replay.get("process_nonce"):
        mismatches.append("natural 与 replay 的 process_nonce 相同")
    if natural.get("process_pid") == replay.get("process_pid"):
        mismatches.append("natural 与 replay 的 process_pid 相同")
    return mismatches


def _compare_signature_group(
    natural_signatures: Mapping[str, Any],
    replay_signatures: Mapping[str, Any],
    *,
    path: str,
    allow_none: bool,
    expected_count: int,
    natural_reader: DenseTensorReader,
    replay_reader: DenseTensorReader,
    mismatches: list[str],
) -> SignatureDivergence:
    divergence = SignatureDivergence()
    _compare_parameter_signatures(
        natural_signatures,
        replay_signatures,
        path=path,
        divergence=divergence,
        allow_none=allow_none,
        expected_count=expected_count,
        left_dense_reader=natural_reader,
        right_dense_reader=replay_reader,
        mismatches=mismatches,
    )
    return divergence


def analyze_same_role_baseline(
    natural_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    """比较同角色的 natural/replay 稠密张量并返回不设阈值的诊断。"""

    natural_path = natural_path.expanduser().resolve()
    replay_path = replay_path.expanduser().resolve()
    if natural_path == replay_path:
        raise ValueError("natural 与 replay 不得指向同一个 JSON 文件。")
    natural = json.loads(natural_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    natural_sha256 = _sha256_file(natural_path)
    replay_sha256 = _sha256_file(replay_path)
    mismatches = validate_trace_pair(
        natural,
        replay,
        natural_sha256=natural_sha256,
    )

    natural_dense_file = _resolve_dense_file(natural_path, natural)
    replay_dense_file = _resolve_dense_file(replay_path, replay)
    _validate_dense_tensor_store(
        natural,
        role="natural",
        dense_file=natural_dense_file,
        mismatches=mismatches,
    )
    _validate_dense_tensor_store(
        replay,
        role="replay",
        dense_file=replay_dense_file,
        mismatches=mismatches,
    )
    natural_reader = DenseTensorReader(natural_dense_file)
    replay_reader = DenseTensorReader(replay_dense_file)

    global_diagnostics = {
        "voxel_outputs": SignatureDivergence(),
        "accumulated_voxel_gradients": SignatureDivergence(),
        **{
            f"optimizer_signatures.{field_name}": SignatureDivergence()
            for field_name in OPTIMIZER_SIGNATURE_LIMITS
        },
    }
    local_diagnostics: list[dict[str, Any]] = []
    natural_microbatches = list(natural.get("microbatches", ()))
    replay_microbatches = list(replay.get("microbatches", ()))
    if len(natural_microbatches) != len(replay_microbatches):
        mismatches.append("natural 与 replay 的 microbatch 数量不同")
    for index, (natural_microbatch, replay_microbatch) in enumerate(
        zip(natural_microbatches, replay_microbatches)
    ):
        for field_name in (
            "microbatch",
            "optimizer_step",
            "recycle_passes",
            "request_positions",
            "requests",
        ):
            if natural_microbatch.get(field_name) != replay_microbatch.get(field_name):
                mismatches.append(f"microbatches[{index}].{field_name} 不同")
        voxel_divergence = _compare_signature_group(
            natural_microbatch["voxel_outputs"],
            replay_microbatch["voxel_outputs"],
            path=f"microbatches[{index}].voxel_outputs",
            allow_none=False,
            expected_count=len(EXPECTED_VOXEL_OUTPUT_NAMES),
            natural_reader=natural_reader,
            replay_reader=replay_reader,
            mismatches=mismatches,
        )
        gradient_divergence = _compare_signature_group(
            natural_microbatch["accumulated_voxel_gradients"],
            replay_microbatch["accumulated_voxel_gradients"],
            path=f"microbatches[{index}].accumulated_voxel_gradients",
            allow_none=True,
            expected_count=EXPECTED_VOXEL_PARAMETER_TENSORS,
            natural_reader=natural_reader,
            replay_reader=replay_reader,
            mismatches=mismatches,
        )
        global_diagnostics["voxel_outputs"].merge(voxel_divergence)
        global_diagnostics["accumulated_voxel_gradients"].merge(gradient_divergence)
        local_diagnostics.append(
            {
                "kind": "microbatch",
                "index": index,
                "voxel_outputs": voxel_divergence.report(),
                "accumulated_voxel_gradients": gradient_divergence.report(),
            }
        )

    natural_steps = list(natural.get("optimizer_step_records", ()))
    replay_steps = list(replay.get("optimizer_step_records", ()))
    if len(natural_steps) != len(replay_steps):
        mismatches.append("natural 与 replay 的 optimizer step 数量不同")
    for index, (natural_step, replay_step) in enumerate(
        zip(natural_steps, replay_steps)
    ):
        for field_name in (
            "optimizer_step",
            "last_microbatch",
            "lr_before_step",
            "lr_after_step",
            "optimizer_state_step",
        ):
            if natural_step.get(field_name) != replay_step.get(field_name):
                mismatches.append(f"optimizer_step_records[{index}].{field_name} 不同")
        step_diagnostics = {}
        for field_name in OPTIMIZER_SIGNATURE_LIMITS:
            divergence = _compare_signature_group(
                natural_step[field_name],
                replay_step[field_name],
                path=f"optimizer_step_records[{index}].{field_name}",
                allow_none=field_name in GRADIENT_SIGNATURE_FIELDS,
                expected_count=EXPECTED_VOXEL_PARAMETER_TENSORS,
                natural_reader=natural_reader,
                replay_reader=replay_reader,
                mismatches=mismatches,
            )
            step_diagnostics[field_name] = divergence.report()
            global_diagnostics[f"optimizer_signatures.{field_name}"].merge(divergence)
        local_diagnostics.append(
            {
                "kind": "optimizer_step",
                "index": index,
                "optimizer_signatures": step_diagnostics,
            }
        )

    return {
        "analysis_schema_version": 1,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "dense_tensor_store_version": DENSE_TENSOR_STORE_VERSION,
        "valid": not mismatches,
        "role": natural.get("role"),
        "natural_trace_sha256": natural_sha256,
        "replay_trace_sha256": replay_sha256,
        "natural_process_nonce": natural.get("process_nonce"),
        "replay_process_nonce": replay.get("process_nonce"),
        "voxel_parameter_tensor_count": EXPECTED_VOXEL_PARAMETER_TENSORS,
        "global_diagnostics": {
            name: divergence.report() for name, divergence in global_diagnostics.items()
        },
        "local_diagnostics": local_diagnostics,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    arguments = build_parser().parse_args()
    result = analyze_same_role_baseline(arguments.natural, arguments.replay)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, arguments.output)
    print(arguments.output)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
