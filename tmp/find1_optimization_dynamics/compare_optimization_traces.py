"""比较 AUTO B_trunk 与完整 Find_1 的体素优化轨迹 JSON."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


AUTO_TRUNK_SOURCE_IDENTITY = "7aae1f126ee0c1636a3f14455f8239a072ac28c9"
EXPECTED_VOXEL_PARAMETER_TENSORS = 353
EXPECTED_FULL_PARAMETER_TENSORS = 1594
EXPECTED_OTHER_PARAMETER_TENSORS = 1241
EXPECTED_VOXEL_PARAMETER_NAMES_SHA256 = (
    "57cb249c0410369a60c17b09950432a23069c541ac4ddf56e9ef4cf9e9467e2e"
)
EXPECTED_POINT_LOSS_NAMES = frozenset(("atom", "pseudo"))
EXPECTED_VOXEL_LOSS_NAMES = frozenset(
    (
        "receptor",
        "voxel_ligand",
        "protein_mainchain",
        "nucleic_mainchain",
        "ligand_distance",
    )
)
EXPECTED_VOXEL_OUTPUT_NAMES = frozenset(
    (
        "voxel_logits_aux",
        "voxel_logits_ligand",
        "voxel_logits_protein",
        "voxel_logits_nucleic",
        "voxel_logits_distance",
        "voxel_recycle_out",
    )
)
TRACE_IDENTITY_FIELDS = (
    "role",
    "source_identity",
    "project_root",
    "project_source_sha256",
    "experiment",
    "resolved_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_global_step",
    "checkpoint_epoch",
    "shared_state_count",
    "batch_size",
    "accumulate_steps",
    "optimizer_steps",
    "seed",
    "input_channels",
    "trainable_parameter_tensors",
    "voxel_parameter_tensors",
    "other_parameter_tensors",
    "voxel_parameter_names_sha256",
)
MAX_LOSS_VALUE = 10.0
TRACE_SCHEMA_VERSION = 4
DENSE_TENSOR_STORE_VERSION = 1
DENSE_COMPARISON_CHUNK_NUMEL = 1_048_576

# H100 BF16 轨迹中的共同体素路径含 CUDA scatter_add_。同一模型、同一请求、
# 同一 recycle 序列的独立进程重复运行也不会逐字节相等。传统统计字段的包络由
# attempt a6 的同角色重复轨迹和双向 replay 轨迹标定；schema 4 新增的逐元素
# float32 稠密比较及局部相对差由 attempt a7/a8 的独立同角色基线标定。
# 身份、初始参数、参数集合、点损失回灌和 AdamW step 映射仍按严格相等核对。
VOXEL_OUTPUT_LIMITS = {
    "l2_absolute_max": 2.0e1,
    "l2_relative_max": 5.0e-4,
    "mean_absolute_max": 1.0e-2,
    "mean_relative_max": 5.0e-4,
    "max_abs_absolute_max": 2.0,
    "max_abs_relative_max": 5.0e-2,
    "sample_absolute_max": 5.0e-1,
    "element_absolute_max": 3.5,
}
COMMON_LOSS_ABSOLUTE_MAX = 1.0e-3
COMMON_LOSS_RELATIVE_MAX = 5.0e-3
ACCUMULATED_GRADIENT_LIMITS = {
    "l2_absolute_max": 1.0e-2,
    "l2_relative_p95": 1.0e-1,
    "mean_absolute_max": 2.0e-4,
    "max_abs_absolute_max": 2.0e-3,
    "max_abs_relative_p95": 1.6e-1,
    "sample_absolute_max": 5.0e-4,
    "element_absolute_max": 2.0e-3,
}
ACCUMULATED_NORM_ABSOLUTE_MAX = 1.0e-2
ACCUMULATED_NORM_RELATIVE_MAX = 5.0e-2
OPTIMIZER_SCALAR_ABSOLUTE_MAX = 2.0e-2
OPTIMIZER_SCALAR_RELATIVE_MAX = 5.0e-2
OPTIMIZER_SIGNATURE_LIMITS = {
    "parameters_before_step": {
        "l2_absolute_max": 2.0e-4,
        "l2_relative_p95": 5.0e-4,
        "l2_relative_max": 2.0e-3,
        "mean_absolute_max": 1.0e-5,
        "max_abs_absolute_max": 2.0e-4,
        "max_abs_relative_p95": 1.0e-3,
        "sample_absolute_max": 1.0e-4,
        "element_absolute_max": 2.0e-4,
    },
    "pre_clip_gradients": {
        "l2_absolute_max": 1.0e-2,
        "l2_relative_p95": 5.0e-2,
        "l2_relative_max": 1.5e-1,
        "mean_absolute_max": 2.0e-4,
        "max_abs_absolute_max": 2.0e-3,
        "max_abs_relative_p95": 8.0e-2,
        "sample_absolute_max": 5.0e-4,
        "element_absolute_max": 2.0e-3,
    },
    "post_clip_gradients": {
        "l2_absolute_max": 1.0e-2,
        "l2_relative_p95": 5.0e-2,
        "l2_relative_max": 1.5e-1,
        "mean_absolute_max": 2.0e-4,
        "max_abs_absolute_max": 2.0e-3,
        "max_abs_relative_p95": 8.0e-2,
        "sample_absolute_max": 5.0e-4,
        "element_absolute_max": 2.0e-3,
    },
    "parameters_after_step": {
        "l2_absolute_max": 2.0e-4,
        "l2_relative_p95": 5.0e-4,
        "l2_relative_max": 2.0e-3,
        "mean_absolute_max": 1.0e-5,
        "max_abs_absolute_max": 2.0e-4,
        "max_abs_relative_p95": 1.0e-3,
        "sample_absolute_max": 1.0e-4,
        "element_absolute_max": 2.0e-4,
    },
    "parameter_updates": {
        "l2_absolute_max": 2.0e-4,
        "l2_relative_p95": 5.0e-2,
        "l2_relative_max": 2.0e-1,
        "mean_absolute_max": 1.0e-5,
        "max_abs_absolute_max": 1.0e-5,
        "max_abs_relative_p95": 2.0e-2,
        "sample_absolute_max": 1.0e-4,
        "element_absolute_max": 5.0e-5,
    },
    "exp_avg_after_step": {
        "l2_absolute_max": 1.0e-3,
        "l2_relative_p95": 5.0e-2,
        "l2_relative_max": 1.5e-1,
        "mean_absolute_max": 1.0e-5,
        "max_abs_absolute_max": 2.0e-4,
        "max_abs_relative_p95": 8.0e-2,
        "sample_absolute_max": 5.0e-5,
        "element_absolute_max": 2.0e-4,
    },
    "exp_avg_sq_after_step": {
        "l2_absolute_max": 2.0e-7,
        "l2_relative_p95": 1.0e-1,
        "l2_relative_max": 3.0e-1,
        "mean_absolute_max": 1.0e-9,
        "max_abs_absolute_max": 1.0e-7,
        "max_abs_relative_p95": 1.5e-1,
        "sample_absolute_max": 1.0e-8,
        "element_absolute_max": 1.0e-7,
    },
}


def _relative_difference(left: float, right: float) -> float:
    """计算对称相对差；极小量以 1e-12 限制分母。"""

    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def _quantile(values: Sequence[float], fraction: float) -> float:
    """按保守的左连续位置读取小样本分位数。"""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(int((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


class DenseTensorReader:
    """按签名描述读取一条轨迹的 float32 稠密旁车。"""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._mapped: np.memmap | None = None

    def read(
        self,
        signature: Mapping[str, Any],
        *,
        path: str,
        mismatches: list[str],
    ) -> np.ndarray | None:
        """恢复签名对应的扁平 float32 数组并核对内容摘要。"""

        descriptor = signature.get("dense_float32")
        if not isinstance(descriptor, Mapping):
            mismatches.append(f"{path}: 缺少 dense_float32")
            return None
        encoding = descriptor.get("encoding")
        expected_num_bytes = int(signature["numel"]) * np.dtype("<f4").itemsize
        try:
            num_bytes = int(descriptor.get("num_bytes", -1))
        except (TypeError, ValueError):
            mismatches.append(f"{path}.num_bytes: 不是整数")
            return None
        if num_bytes != expected_num_bytes:
            mismatches.append(f"{path}.num_bytes: {num_bytes} != {expected_num_bytes}")
            return None

        if encoding == "base64-float32-le":
            try:
                raw_values: bytes | np.ndarray = base64.b64decode(
                    descriptor.get("data", ""),
                    validate=True,
                )
            except (TypeError, ValueError):
                mismatches.append(f"{path}.data: 不是合法 Base64")
                return None
            if len(raw_values) != num_bytes:
                mismatches.append(f"{path}.data: {len(raw_values)} 字节 != {num_bytes}")
                return None
        elif encoding == "sidecar-float32-le":
            raw_values = self._sidecar_slice(
                descriptor,
                num_bytes=num_bytes,
                path=path,
                mismatches=mismatches,
            )
            if raw_values is None:
                return None
        else:
            mismatches.append(f"{path}.encoding: 不支持 {encoding}")
            return None

        expected_sha256 = descriptor.get("sha256")
        if not _is_sha256(expected_sha256):
            mismatches.append(f"{path}.sha256: 不是 SHA-256")
            return None
        actual_sha256 = hashlib.sha256(raw_values).hexdigest()
        if actual_sha256 != expected_sha256:
            mismatches.append(
                f"{path}.sha256: 旁车内容 {actual_sha256} != {expected_sha256}"
            )
            return None
        values = np.frombuffer(raw_values, dtype="<f4", count=int(signature["numel"]))
        if not bool(np.isfinite(values).all()):
            mismatches.append(f"{path}: 稠密数值包含 NaN 或 Inf")
            return None
        return values

    def _sidecar_slice(
        self,
        descriptor: Mapping[str, Any],
        *,
        num_bytes: int,
        path: str,
        mismatches: list[str],
    ) -> np.ndarray | None:
        """从内存映射旁车读取一个边界已核对的字节区间。"""

        if self.path is None or not self.path.is_file():
            mismatches.append(f"{path}: 稠密旁车不存在：{self.path}")
            return None
        try:
            offset = int(descriptor.get("offset", -1))
        except (TypeError, ValueError):
            mismatches.append(f"{path}.offset: 不是整数")
            return None
        file_size = self.path.stat().st_size
        if offset < 0 or num_bytes < 0 or offset + num_bytes > file_size:
            mismatches.append(
                f"{path}: 旁车区间 [{offset}, {offset + num_bytes}) "
                f"越过文件大小 {file_size}"
            )
            return None
        if self._mapped is None:
            self._mapped = np.memmap(self.path, mode="r", dtype=np.uint8)
        return self._mapped[offset : offset + num_bytes]


@dataclass
class SignatureDivergence:
    """累计同一组张量签名的数值差异与原始字节差异。"""

    signature_count: int = 0
    sha256_mismatch_count: int = 0
    l2_absolute: list[float] = field(default_factory=list)
    l2_relative: list[float] = field(default_factory=list)
    mean_absolute: list[float] = field(default_factory=list)
    mean_relative: list[float] = field(default_factory=list)
    max_abs_absolute: list[float] = field(default_factory=list)
    max_abs_relative: list[float] = field(default_factory=list)
    sample_absolute: list[float] = field(default_factory=list)
    element_absolute: list[float] = field(default_factory=list)

    def merge(self, other: SignatureDivergence) -> None:
        """把一个局部比较的逐张量差异并入全局诊断。"""

        self.signature_count += other.signature_count
        self.sha256_mismatch_count += other.sha256_mismatch_count
        for field_name in (
            "l2_absolute",
            "l2_relative",
            "mean_absolute",
            "mean_relative",
            "max_abs_absolute",
            "max_abs_relative",
            "sample_absolute",
            "element_absolute",
        ):
            getattr(self, field_name).extend(getattr(other, field_name))

    def report(self) -> dict[str, Any]:
        """返回可写入比较 JSON 的固定统计字段。"""

        return {
            "signature_count": self.signature_count,
            "sha256_mismatch_count": self.sha256_mismatch_count,
            "l2_absolute_p95": _quantile(self.l2_absolute, 0.95),
            "l2_absolute_max": max(self.l2_absolute, default=0.0),
            "l2_relative_p95": _quantile(self.l2_relative, 0.95),
            "l2_relative_max": max(self.l2_relative, default=0.0),
            "mean_absolute_p95": _quantile(self.mean_absolute, 0.95),
            "mean_absolute_max": max(self.mean_absolute, default=0.0),
            "mean_relative_p95": _quantile(self.mean_relative, 0.95),
            "mean_relative_max": max(self.mean_relative, default=0.0),
            "max_abs_absolute_p95": _quantile(self.max_abs_absolute, 0.95),
            "max_abs_absolute_max": max(self.max_abs_absolute, default=0.0),
            "max_abs_relative_p95": _quantile(self.max_abs_relative, 0.95),
            "max_abs_relative_max": max(self.max_abs_relative, default=0.0),
            "sample_absolute_p95": _quantile(self.sample_absolute, 0.95),
            "sample_absolute_max": max(self.sample_absolute, default=0.0),
            "element_absolute_p95": _quantile(self.element_absolute, 0.95),
            "element_absolute_max": max(self.element_absolute, default=0.0),
        }


@dataclass
class ScalarDivergence:
    """累计标量的绝对差与对称相对差。"""

    absolute: list[float] = field(default_factory=list)
    relative: list[float] = field(default_factory=list)

    def add(self, left: float, right: float) -> None:
        self.absolute.append(abs(left - right))
        self.relative.append(_relative_difference(left, right))

    def merge(self, other: ScalarDivergence) -> None:
        """把一个局部比较的标量差异并入全局诊断。"""

        self.absolute.extend(other.absolute)
        self.relative.extend(other.relative)

    def report(self) -> dict[str, float | int]:
        """返回标量差异的数量、P95 和最大值。"""

        return {
            "count": len(self.absolute),
            "absolute_p95": _quantile(self.absolute, 0.95),
            "absolute_max": max(self.absolute, default=0.0),
            "relative_p95": _quantile(self.relative, 0.95),
            "relative_max": max(self.relative, default=0.0),
        }


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


def _require_loss_range(
    value: Any,
    *,
    path: str,
    mismatches: list[str],
) -> float:
    """要求损失落在仅用于排除损坏轨迹的宽松有限区间。"""

    number = _require_finite(value, path=path, mismatches=mismatches)
    if number < 0.0 or number > MAX_LOSS_VALUE:
        mismatches.append(f"{path}: {number} 不在 [0, {MAX_LOSS_VALUE}] 内")
    return number


def _max_element_absolute_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """分块计算两个扁平 float32 数组的逐元素最大绝对差。"""

    maximum = 0.0
    for begin in range(0, int(left.size), DENSE_COMPARISON_CHUNK_NUMEL):
        end = min(begin + DENSE_COMPARISON_CHUNK_NUMEL, int(left.size))
        differences = np.abs(
            left[begin:end].astype(np.float64) - right[begin:end].astype(np.float64)
        )
        if differences.size:
            maximum = max(maximum, float(np.max(differences)))
    return maximum


def _compare_signature(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    path: str,
    divergence: SignatureDivergence,
    require_sha256: bool,
    allow_none: bool,
    left_dense_reader: DenseTensorReader,
    right_dense_reader: DenseTensorReader,
    mismatches: list[str],
) -> None:
    """严格核对张量结构，并累计 BF16 数值差异。"""

    if left is None or right is None:
        if left is None and right is None and allow_none:
            return
        if left is None and right is None:
            mismatches.append(f"{path}: 两侧均为空，但该字段必须存在")
        else:
            mismatches.append(f"{path}: 一侧为空")
        return
    for side, signature in (("trunk", left), ("full", right)):
        if not bool(signature["all_finite"]):
            mismatches.append(f"{path}.{side}: 张量包含 NaN 或 Inf")
    structure_matches = True
    for key in ("shape", "dtype", "numel"):
        if left[key] != right[key]:
            mismatches.append(f"{path}.{key}: {left[key]} != {right[key]}")
            structure_matches = False
    divergence.signature_count += 1
    if left["sha256"] != right["sha256"]:
        divergence.sha256_mismatch_count += 1
        if require_sha256:
            mismatches.append(f"{path}.sha256: {left['sha256']} != {right['sha256']}")
    for key, absolute_target, relative_target in (
        ("l2", divergence.l2_absolute, divergence.l2_relative),
        ("mean", divergence.mean_absolute, divergence.mean_relative),
        ("max_abs", divergence.max_abs_absolute, divergence.max_abs_relative),
    ):
        left_value = _require_finite(
            left[key],
            path=f"{path}.trunk.{key}",
            mismatches=mismatches,
        )
        right_value = _require_finite(
            right[key],
            path=f"{path}.full.{key}",
            mismatches=mismatches,
        )
        absolute_target.append(abs(left_value - right_value))
        relative_target.append(_relative_difference(left_value, right_value))
    left_samples = list(left["samples"])
    right_samples = list(right["samples"])
    if len(left_samples) != len(right_samples):
        mismatches.append(f"{path}.samples: 长度不同")
        return
    for index, (left_value, right_value) in enumerate(
        zip(left_samples, right_samples, strict=True)
    ):
        left_number = _require_finite(
            left_value,
            path=f"{path}.trunk.samples[{index}]",
            mismatches=mismatches,
        )
        right_number = _require_finite(
            right_value,
            path=f"{path}.full.samples[{index}]",
            mismatches=mismatches,
        )
        divergence.sample_absolute.append(abs(left_number - right_number))
    if not structure_matches:
        return
    left_values = left_dense_reader.read(
        left,
        path=f"{path}.trunk.dense_float32",
        mismatches=mismatches,
    )
    right_values = right_dense_reader.read(
        right,
        path=f"{path}.full.dense_float32",
        mismatches=mismatches,
    )
    if left_values is None or right_values is None:
        return
    divergence.element_absolute.append(
        _max_element_absolute_difference(left_values, right_values)
    )


def _validate_signatures_finite(
    signatures: Mapping[str, Any],
    *,
    path: str,
    expected_count: int,
    allow_none: bool,
    mismatches: list[str],
) -> None:
    if len(signatures) != expected_count:
        mismatches.append(f"{path}: 参数数量 {len(signatures)} != {expected_count}")
    for name, signature in signatures.items():
        if signature is None:
            if not allow_none:
                mismatches.append(f"{path}.{name}: 该字段不得为空")
            continue
        if not bool(signature["all_finite"]):
            mismatches.append(f"{path}.{name}: 张量包含 NaN 或 Inf")
        for key in ("l2", "mean", "max_abs"):
            _require_finite(
                signature[key],
                path=f"{path}.{name}.{key}",
                mismatches=mismatches,
            )
        for sample_index, sample in enumerate(signature["samples"]):
            _require_finite(
                sample,
                path=f"{path}.{name}.samples[{sample_index}]",
                mismatches=mismatches,
            )


def _compare_parameter_signatures(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    path: str,
    divergence: SignatureDivergence,
    require_sha256: bool = False,
    allow_none: bool = False,
    expected_count: int | None = None,
    left_dense_reader: DenseTensorReader,
    right_dense_reader: DenseTensorReader,
    mismatches: list[str],
) -> None:
    if expected_count is not None:
        for role, signatures in (("trunk", left), ("full", right)):
            if len(signatures) != expected_count:
                mismatches.append(
                    f"{path}.{role}: {len(signatures)} != {expected_count}"
                )
    if set(left) != set(right):
        mismatches.append(f"{path}: 参数名称集合不同")
        return
    for name in sorted(left):
        _compare_signature(
            left[name],
            right[name],
            path=f"{path}.{name}",
            divergence=divergence,
            require_sha256=require_sha256,
            allow_none=allow_none,
            left_dense_reader=left_dense_reader,
            right_dense_reader=right_dense_reader,
            mismatches=mismatches,
        )


def _validate_signature_limits(
    divergence: SignatureDivergence,
    *,
    path: str,
    limits: Mapping[str, float],
    mismatches: list[str],
) -> dict[str, Any]:
    """把一组签名差异与已冻结的数值包络逐字段比较。"""

    report = divergence.report()
    for key, limit in limits.items():
        actual = float(report[key])
        if actual > float(limit):
            mismatches.append(f"{path}.{key}: {actual} > {limit}")
    return report


def _validate_scalar_limits(
    divergence: ScalarDivergence,
    *,
    path: str,
    absolute_max: float,
    relative_max: float,
    mismatches: list[str],
) -> dict[str, float | int]:
    """要求一组标量差同时满足绝对差与相对差上限。"""

    report = divergence.report()
    if float(report["absolute_max"]) > absolute_max:
        mismatches.append(
            f"{path}.absolute_max: {report['absolute_max']} > {absolute_max}"
        )
    if float(report["relative_max"]) > relative_max:
        mismatches.append(
            f"{path}.relative_max: {report['relative_max']} > {relative_max}"
        )
    return report


def _validate_optimizer_state_steps(
    state_steps: Mapping[str, Any],
    exp_avg: Mapping[str, Any],
    exp_avg_sq: Mapping[str, Any],
    *,
    expected_names: set[str],
    maximum_step: int,
    path: str,
    mismatches: list[str],
) -> None:
    """核对逐参数 AdamW step 与一、二阶矩的存在性关系."""

    if set(state_steps) != expected_names:
        mismatches.append(f"{path}: 参数名称集合不同")
        return
    active_count = 0
    for name in sorted(expected_names):
        step = state_steps[name]
        has_moments = exp_avg[name] is not None and exp_avg_sq[name] is not None
        if step is None:
            if exp_avg[name] is not None or exp_avg_sq[name] is not None:
                mismatches.append(f"{path}.{name}: step 为空但 AdamW 矩已存在")
            continue
        active_count += 1
        integer_step = int(step)
        if integer_step <= 0 or integer_step > maximum_step:
            mismatches.append(
                f"{path}.{name}: {integer_step} 不在 [1, {maximum_step}] 内"
            )
        if not has_moments:
            mismatches.append(f"{path}.{name}: step 已存在但 AdamW 矩缺失")
    if active_count == 0:
        mismatches.append(f"{path}: 没有参数参与本次 optimizer step")


def _first_recycle_divergence(
    trunk_microbatches: Sequence[Mapping[str, Any]],
    full_microbatches: Sequence[Mapping[str, Any]],
) -> int | None:
    for trunk, full in zip(trunk_microbatches, full_microbatches, strict=True):
        if trunk["recycle_passes"] != full["recycle_passes"]:
            return int(trunk["microbatch"])
    return None


def _trace_identity(trace: Mapping[str, Any]) -> dict[str, Any]:
    """提取可跨 natural 与 replay 比较的完整运行身份。"""

    return {field_name: trace.get(field_name) for field_name in TRACE_IDENTITY_FIELDS}


def _is_sha256(value: Any) -> bool:
    """判断字段是否为非空的小写或大写 SHA-256 十六进制文本。"""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _validate_dense_tensor_store(
    trace: Mapping[str, Any],
    *,
    role: str,
    dense_file: Path | None,
    mismatches: list[str],
) -> None:
    """核对一条轨迹声明的稠密旁车身份与文件规模。"""

    record = trace.get("dense_tensor_store")
    if not isinstance(record, Mapping):
        mismatches.append(f"metadata.{role}.dense_tensor_store: 缺失")
        return
    if record.get("version") != DENSE_TENSOR_STORE_VERSION:
        mismatches.append(
            f"metadata.{role}.dense_tensor_store.version: "
            f"{record.get('version')} != {DENSE_TENSOR_STORE_VERSION}"
        )
    if record.get("dtype") != "float32-le":
        mismatches.append(
            f"metadata.{role}.dense_tensor_store.dtype: {record.get('dtype')}"
        )
    integer_values = {}
    for key in (
        "total_bytes",
        "captured_signature_count",
        "unique_sidecar_tensor_count",
    ):
        try:
            integer_values[key] = int(record.get(key, -1))
        except (TypeError, ValueError):
            mismatches.append(f"metadata.{role}.dense_tensor_store.{key}: 不是整数")
            continue
        if integer_values[key] < 0:
            mismatches.append(
                f"metadata.{role}.dense_tensor_store.{key}: "
                f"{integer_values[key]} < 0"
            )
    unique_count = integer_values.get("unique_sidecar_tensor_count", -1)
    if unique_count == 0 and dense_file is None:
        return
    file_name = record.get("file_name")
    if dense_file is None:
        mismatches.append(f"metadata.{role}.dense_tensor_store: 未提供旁车文件")
        return
    if file_name != dense_file.name:
        mismatches.append(
            f"metadata.{role}.dense_tensor_store.file_name: "
            f"{file_name} != {dense_file.name}"
        )
    if not dense_file.is_file():
        mismatches.append(
            f"metadata.{role}.dense_tensor_store: 文件不存在 {dense_file}"
        )
        return
    expected_size = integer_values.get("total_bytes", -1)
    if dense_file.stat().st_size != expected_size:
        mismatches.append(
            f"metadata.{role}.dense_tensor_store.total_bytes: "
            f"文件为 {dense_file.stat().st_size}，记录为 {expected_size}"
        )


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

    if (
        trunk.get("schema_version") != TRACE_SCHEMA_VERSION
        or full.get("schema_version") != TRACE_SCHEMA_VERSION
    ):
        mismatches.append(
            f"metadata.schema_version: 两条轨迹都必须为 {TRACE_SCHEMA_VERSION}"
        )
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
    for role, trace in (("trunk", trunk), ("full", full)):
        process_pid = trace.get("process_pid")
        if not isinstance(process_pid, int) or process_pid <= 0:
            mismatches.append(f"metadata.{role}.process_pid: 不是正整数")

    track = trunk.get("track")
    if track not in {"controlled", "natural", "replay"}:
        mismatches.append(f"metadata.track: 不支持 {track}")
    recycle_source = trunk.get("recycle_sequence_source_sha256")
    if track == "replay" and not _is_sha256(recycle_source):
        mismatches.append(
            "metadata.recycle_sequence_source_sha256: replay 必须提供 SHA-256"
        )
    if track != "replay" and recycle_source is not None:
        mismatches.append("metadata.recycle_sequence_source_sha256: 非 replay 必须为空")

    for key in (
        "track",
        "checkpoint_sha256",
        "checkpoint",
        "checkpoint_global_step",
        "checkpoint_epoch",
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


def _validate_counterfactual_replays(
    trunk: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    counterfactual_replays: Sequence[Mapping[str, Any]],
    natural_trace_sha256s: Sequence[str],
    mismatches: list[str],
) -> None:
    """核对自然轨迹所依赖的两条 replay 比较及其完整身份。"""

    required_sources = tuple(str(value) for value in natural_trace_sha256s)
    if (
        len(required_sources) != 2
        or len(set(required_sources)) != 2
        or not all(_is_sha256(value) for value in required_sources)
    ):
        mismatches.append("natural.trace_sha256s: 必须提供两条不同的真实 SHA-256")
        return
    if len(counterfactual_replays) != 2:
        mismatches.append(
            "natural.counterfactual_replays: 必须恰好提供两次 replay 比较"
        )

    expected_identity = {
        "trunk": _trace_identity(trunk),
        "full": _trace_identity(full),
    }
    observed_sources = []
    replay_trace_hashes = []
    for index, comparison in enumerate(counterfactual_replays):
        path = f"natural.counterfactual_replays[{index}]"
        if comparison.get("schema_version") != TRACE_SCHEMA_VERSION:
            mismatches.append(f"{path}.schema_version: 必须为 {TRACE_SCHEMA_VERSION}")
        if comparison.get("passed") is not True:
            mismatches.append(f"{path}.passed: 必须为 true")
        if comparison.get("track") != "replay":
            mismatches.append(f"{path}.track: 必须为 replay")
        if comparison.get("trace_identity") != expected_identity:
            mismatches.append(f"{path}.trace_identity: 与当前自然轨迹身份不同")

        source = comparison.get("recycle_sequence_source_sha256")
        if not _is_sha256(source):
            mismatches.append(f"{path}.recycle_sequence_source_sha256: 不是 SHA-256")
        else:
            observed_sources.append(str(source))
        pair_hashes = (
            comparison.get("trunk_trace_sha256"),
            comparison.get("full_trace_sha256"),
        )
        if not all(_is_sha256(value) for value in pair_hashes):
            mismatches.append(f"{path}.trace_sha256: 两条 replay 轨迹摘要必须存在")
        else:
            replay_trace_hashes.extend(str(value) for value in pair_hashes)
            if pair_hashes[0] == pair_hashes[1]:
                mismatches.append(f"{path}.trace_sha256: trunk 与 full 复用了同一轨迹")

    if set(observed_sources) != set(required_sources) or len(observed_sources) != 2:
        mismatches.append(
            "natural.counterfactual_replays: 必须分别重放两条自然轨迹的 recycle 序列"
        )
    if len(replay_trace_hashes) == 4 and len(set(replay_trace_hashes)) != 4:
        mismatches.append(
            "natural.counterfactual_replays: 两次 replay 必须使用四条独立轨迹"
        )


def _validate_microbatch_loss_ranges(
    trace: Mapping[str, Any],
    *,
    role: str,
    mismatches: list[str],
) -> None:
    """在跨角色比较前独立核对整条轨迹的损失名称与安全区间。"""

    expected_names = (
        EXPECTED_VOXEL_LOSS_NAMES
        if role == "trunk"
        else EXPECTED_VOXEL_LOSS_NAMES | EXPECTED_POINT_LOSS_NAMES
    )
    for index, microbatch in enumerate(trace["microbatches"]):
        terms = microbatch["loss_terms"]
        if set(terms) != expected_names:
            mismatches.append(
                f"microbatches[{index}].{role}.loss_terms: "
                f"{sorted(terms)} != {sorted(expected_names)}"
            )
        _require_loss_range(
            microbatch["total_loss"],
            path=f"microbatches[{index}].{role}.total_loss",
            mismatches=mismatches,
        )
        for loss_name, value in terms.items():
            _require_loss_range(
                value,
                path=f"microbatches[{index}].{role}.loss_terms.{loss_name}",
                mismatches=mismatches,
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
    expected_microbatch_count = int(trunk["accumulate_steps"]) * int(
        trunk["optimizer_steps"]
    )
    for role, trace, microbatches in (
        ("trunk", trunk, trunk_microbatches),
        ("full", full, full_microbatches),
    ):
        if len(microbatches) != expected_microbatch_count:
            mismatches.append(
                f"{role}.microbatches: {len(microbatches)} != "
                f"{expected_microbatch_count}"
            )
        actual_sequence = [
            int(microbatch["recycle_passes"]) for microbatch in microbatches
        ]
        if list(trace["recycle_sequence"]) != actual_sequence:
            mismatches.append(f"{role}.recycle_sequence: 与 microbatches 不一致")
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
        len(trunk_microbatches) if recycle_divergence is None else recycle_divergence
    )
    if recycle_divergence is not None and trunk["track"] == "natural":
        _validate_counterfactual_replays(
            trunk,
            full,
            counterfactual_replays=counterfactual_replays,
            natural_trace_sha256s=natural_trace_sha256s,
            mismatches=mismatches,
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
    voxel_output_divergence: SignatureDivergence,
    common_loss_divergence: ScalarDivergence,
    gradient_divergence: SignatureDivergence,
    accumulated_norm_divergence: ScalarDivergence,
    batch_size: int,
    accumulate_steps: int,
    trunk_dense_reader: DenseTensorReader,
    full_dense_reader: DenseTensorReader,
    mismatches: list[str],
) -> None:
    """核对一个 recycle 一致的 microbatch 并累计数值差异。"""

    local_output_divergence = SignatureDivergence()
    local_loss_divergence = ScalarDivergence()
    local_gradient_divergence = SignatureDivergence()
    local_norm_divergence = ScalarDivergence()
    for key in ("microbatch", "optimizer_step", "recycle_passes"):
        if trunk_microbatch[key] != full_microbatch[key]:
            mismatches.append(f"microbatches[{index}].{key}: 不同")
    if int(trunk_microbatch["microbatch"]) != index:
        mismatches.append(
            f"microbatches[{index}].microbatch: {trunk_microbatch['microbatch']} != {index}"
        )
    expected_optimizer_step = index // accumulate_steps
    if int(trunk_microbatch["optimizer_step"]) != expected_optimizer_step:
        mismatches.append(
            f"microbatches[{index}].optimizer_step: "
            f"{trunk_microbatch['optimizer_step']} != {expected_optimizer_step}"
        )
    for role, microbatch in (
        ("trunk", trunk_microbatch),
        ("full", full_microbatch),
    ):
        if len(microbatch["request_positions"]) != batch_size:
            mismatches.append(
                f"microbatches[{index}].{role}.request_positions: "
                f"{len(microbatch['request_positions'])} != {batch_size}"
            )
        if len(microbatch["requests"]) != batch_size:
            mismatches.append(
                f"microbatches[{index}].{role}.requests: "
                f"{len(microbatch['requests'])} != {batch_size}"
            )
    if trunk_microbatch["request_positions"] != full_microbatch["request_positions"]:
        mismatches.append(f"microbatches[{index}].request_positions: 不同")
    if trunk_microbatch["requests"] != full_microbatch["requests"]:
        mismatches.append(f"microbatches[{index}].requests: 不同")
    trunk_outputs = trunk_microbatch["voxel_outputs"]
    full_outputs = full_microbatch["voxel_outputs"]
    for role, outputs in (("trunk", trunk_outputs), ("full", full_outputs)):
        if set(outputs) != EXPECTED_VOXEL_OUTPUT_NAMES:
            mismatches.append(
                f"microbatches[{index}].{role}.voxel_outputs: "
                f"{sorted(outputs)} != {sorted(EXPECTED_VOXEL_OUTPUT_NAMES)}"
            )
    _compare_parameter_signatures(
        trunk_outputs,
        full_outputs,
        path=f"microbatches[{index}].voxel_outputs",
        divergence=local_output_divergence,
        expected_count=len(EXPECTED_VOXEL_OUTPUT_NAMES),
        left_dense_reader=trunk_dense_reader,
        right_dense_reader=full_dense_reader,
        mismatches=mismatches,
    )
    trunk_terms = trunk_microbatch["loss_terms"]
    full_terms = full_microbatch["loss_terms"]
    if set(trunk_terms) != EXPECTED_VOXEL_LOSS_NAMES:
        mismatches.append(
            f"microbatches[{index}].trunk.loss_terms: "
            f"{sorted(trunk_terms)} != {sorted(EXPECTED_VOXEL_LOSS_NAMES)}"
        )
    if not EXPECTED_POINT_LOSS_NAMES.issubset(full_terms):
        mismatches.append(f"microbatches[{index}].full.loss_terms: 缺少 atom 或 pseudo")
    expected_full_terms = set(trunk_terms) | EXPECTED_POINT_LOSS_NAMES
    if set(full_terms) != expected_full_terms:
        mismatches.append(
            f"microbatches[{index}].full.loss_terms: "
            f"{sorted(full_terms)} != {sorted(expected_full_terms)}"
        )
    for loss_name, trunk_value in trunk_terms.items():
        if loss_name not in full_terms:
            mismatches.append(f"microbatches[{index}].loss_terms.{loss_name}: 缺失")
            continue
        local_loss_divergence.add(
            float(trunk_value),
            float(full_terms[loss_name]),
        )
    _compare_parameter_signatures(
        trunk_microbatch["accumulated_voxel_gradients"],
        full_microbatch["accumulated_voxel_gradients"],
        path=f"microbatches[{index}].accumulated_voxel_gradients",
        divergence=local_gradient_divergence,
        allow_none=True,
        expected_count=EXPECTED_VOXEL_PARAMETER_TENSORS,
        left_dense_reader=trunk_dense_reader,
        right_dense_reader=full_dense_reader,
        mismatches=mismatches,
    )
    trunk_norm = _require_finite(
        trunk_microbatch["accumulated_voxel_gradient_norm"],
        path=f"microbatches[{index}].trunk.accumulated_voxel_gradient_norm",
        mismatches=mismatches,
    )
    full_norm = _require_finite(
        full_microbatch["accumulated_voxel_gradient_norm"],
        path=f"microbatches[{index}].full.accumulated_voxel_gradient_norm",
        mismatches=mismatches,
    )
    local_norm_divergence.add(trunk_norm, full_norm)

    _validate_signature_limits(
        local_output_divergence,
        path=f"microbatches[{index}].numerical.voxel_outputs",
        limits=VOXEL_OUTPUT_LIMITS,
        mismatches=mismatches,
    )
    _validate_scalar_limits(
        local_loss_divergence,
        path=f"microbatches[{index}].numerical.common_losses",
        absolute_max=COMMON_LOSS_ABSOLUTE_MAX,
        relative_max=COMMON_LOSS_RELATIVE_MAX,
        mismatches=mismatches,
    )
    _validate_signature_limits(
        local_gradient_divergence,
        path=f"microbatches[{index}].numerical.accumulated_voxel_gradients",
        limits=ACCUMULATED_GRADIENT_LIMITS,
        mismatches=mismatches,
    )
    _validate_scalar_limits(
        local_norm_divergence,
        path=f"microbatches[{index}].numerical.accumulated_voxel_gradient_norm",
        absolute_max=ACCUMULATED_NORM_ABSOLUTE_MAX,
        relative_max=ACCUMULATED_NORM_RELATIVE_MAX,
        mismatches=mismatches,
    )
    voxel_output_divergence.merge(local_output_divergence)
    common_loss_divergence.merge(local_loss_divergence)
    gradient_divergence.merge(local_gradient_divergence)
    accumulated_norm_divergence.merge(local_norm_divergence)


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
            allow_none=key
            in {
                "other_pre_clip_gradients",
                "other_post_clip_gradients",
                "other_exp_avg_after_step",
                "other_exp_avg_sq_after_step",
            },
            mismatches=mismatches,
        )
    _validate_optimizer_state_steps(
        full_step["other_optimizer_state_step"],
        full_step["other_exp_avg_after_step"],
        full_step["other_exp_avg_sq_after_step"],
        expected_names=expected_names,
        maximum_step=index + 1,
        path=f"optimizer_step_records[{index}].full.other_optimizer_state_step",
        mismatches=mismatches,
    )
    _validate_gradient_state_relation(
        full_step,
        names=expected_names,
        gradient_prefix="other_",
        state_key="other_optimizer_state_step",
        path=f"optimizer_step_records[{index}].full.other_gradients",
        mismatches=mismatches,
    )


def _validate_gradient_state_relation(
    step: Mapping[str, Any],
    *,
    names: set[str],
    gradient_prefix: str,
    state_key: str,
    path: str,
    mismatches: list[str],
) -> None:
    """核对裁剪前后梯度空值与 optimizer state 的存在关系。"""

    pre_clip = step[f"{gradient_prefix}pre_clip_gradients"]
    post_clip = step[f"{gradient_prefix}post_clip_gradients"]
    state_steps = step[state_key]
    for name in sorted(names):
        pre_is_none = pre_clip[name] is None
        post_is_none = post_clip[name] is None
        if pre_is_none != post_is_none:
            mismatches.append(f"{path}.{name}: 裁剪前后仅一侧梯度为空")
        if state_steps[name] is None and not pre_is_none:
            mismatches.append(f"{path}.{name}: 梯度存在但 optimizer state 为空")


def _validate_optimizer_state_transitions(
    optimizer_steps: Sequence[Mapping[str, Any]],
    *,
    names: set[str],
    gradient_prefix: str,
    state_key: str,
    path: str,
    mismatches: list[str],
) -> None:
    """按当前梯度是否存在核对逐参数 AdamW step 的持久迁移。"""

    previous_steps: dict[str, int | None] = {name: None for name in names}
    for step_index, step in enumerate(optimizer_steps):
        gradients = step[f"{gradient_prefix}pre_clip_gradients"]
        state_steps = step[state_key]
        if set(gradients) != names or set(state_steps) != names:
            mismatches.append(f"{path}[{step_index}]: 参数名称集合不同")
            continue
        for name in sorted(names):
            previous = previous_steps[name]
            expected = (
                (0 if previous is None else previous) + 1
                if gradients[name] is not None
                else previous
            )
            actual_value = state_steps[name]
            actual = None if actual_value is None else int(actual_value)
            if actual != expected:
                mismatches.append(
                    f"{path}[{step_index}].{name}: {actual} != {expected}"
                )
            previous_steps[name] = actual


def _validate_accumulated_gradient_step_relation(
    microbatches: Sequence[Mapping[str, Any]],
    optimizer_steps: Sequence[Mapping[str, Any]],
    *,
    role: str,
    mismatches: list[str],
) -> None:
    """核对每个累积区间末尾梯度与裁剪前梯度的空值集合。"""

    for step_index, step in enumerate(optimizer_steps):
        last_microbatch = int(step["last_microbatch"])
        if last_microbatch < 0 or last_microbatch >= len(microbatches):
            mismatches.append(
                f"{role}.optimizer_step_records[{step_index}].last_microbatch: "
                f"{last_microbatch} 越界"
            )
            continue
        gradients = microbatches[last_microbatch]["accumulated_voxel_gradients"]
        pre_clip_gradients = step["pre_clip_gradients"]
        if set(gradients) != set(pre_clip_gradients):
            mismatches.append(
                f"{role}.optimizer_step_records[{step_index}]: "
                "累计梯度与裁剪前梯度参数集合不同"
            )
            continue
        gradient_null_names = {
            name for name, signature in gradients.items() if signature is None
        }
        pre_clip_null_names = {
            name for name, signature in pre_clip_gradients.items() if signature is None
        }
        if gradient_null_names != pre_clip_null_names:
            mismatches.append(
                f"{role}.optimizer_step_records[{step_index}]: "
                "末尾累计梯度与裁剪前梯度空值集合不同"
            )


def _compare_optimizer_step(
    trunk_step: Mapping[str, Any],
    full_step: Mapping[str, Any],
    *,
    index: int,
    expected_last_microbatch: int,
    atol: float,
    rtol: float,
    optimizer_scalar_divergence: ScalarDivergence,
    signature_divergences: Mapping[str, SignatureDivergence],
    trunk_dense_reader: DenseTensorReader,
    full_dense_reader: DenseTensorReader,
    mismatches: list[str],
) -> None:
    """比较一个 optimizer step 的体素动力学并验收其他参数组。"""

    local_scalar_divergence = ScalarDivergence()
    local_signature_divergences = {
        key: SignatureDivergence() for key in OPTIMIZER_SIGNATURE_LIMITS
    }
    for key in ("optimizer_step", "last_microbatch"):
        if trunk_step[key] != full_step[key]:
            mismatches.append(f"optimizer_step_records[{index}].{key}: 不同")
    if int(trunk_step["optimizer_step"]) != index:
        mismatches.append(
            f"optimizer_step_records[{index}].optimizer_step: "
            f"{trunk_step['optimizer_step']} != {index}"
        )
    if int(trunk_step["last_microbatch"]) != expected_last_microbatch:
        mismatches.append(
            f"optimizer_step_records[{index}].last_microbatch: "
            f"{trunk_step['last_microbatch']} != {expected_last_microbatch}"
        )
    schedule_keys = (
        "lr_before_step",
        "lr_after_step",
    )
    envelope_scalar_keys = (
        "pre_clip_group_norm",
        "post_clip_group_norm",
        "voxel_clip_coefficient",
    )
    for key in schedule_keys:
        if not _close(
            float(trunk_step[key]),
            float(full_step[key]),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(f"optimizer_step_records[{index}].{key}: 不同")
    for key in envelope_scalar_keys:
        local_scalar_divergence.add(
            float(trunk_step[key]),
            float(full_step[key]),
        )
    for role, step in (("trunk", trunk_step), ("full", full_step)):
        for key in (*schedule_keys, *envelope_scalar_keys):
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
    voxel_names = set(trunk_step["parameters_before_step"])
    if len(voxel_names) != EXPECTED_VOXEL_PARAMETER_TENSORS:
        mismatches.append(
            f"optimizer_step_records[{index}].voxel_parameters: "
            f"{len(voxel_names)} != {EXPECTED_VOXEL_PARAMETER_TENSORS}"
        )
    for role, step in (("trunk", trunk_step), ("full", full_step)):
        _validate_optimizer_state_steps(
            step["optimizer_state_step"],
            step["exp_avg_after_step"],
            step["exp_avg_sq_after_step"],
            expected_names=voxel_names,
            maximum_step=index + 1,
            path=(f"optimizer_step_records[{index}].{role}.optimizer_state_step"),
            mismatches=mismatches,
        )
        _validate_gradient_state_relation(
            step,
            names=voxel_names,
            gradient_prefix="",
            state_key="optimizer_state_step",
            path=f"optimizer_step_records[{index}].{role}.gradients",
            mismatches=mismatches,
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
            divergence=local_signature_divergences[key],
            require_sha256=(key == "parameters_before_step" and index == 0),
            allow_none=key
            in {
                "pre_clip_gradients",
                "post_clip_gradients",
                "exp_avg_after_step",
                "exp_avg_sq_after_step",
            },
            expected_count=EXPECTED_VOXEL_PARAMETER_TENSORS,
            left_dense_reader=trunk_dense_reader,
            right_dense_reader=full_dense_reader,
            mismatches=mismatches,
        )
    _validate_other_parameter_step(full_step, index=index, mismatches=mismatches)
    _validate_scalar_limits(
        local_scalar_divergence,
        path=f"optimizer_step_records[{index}].numerical.scalars",
        absolute_max=OPTIMIZER_SCALAR_ABSOLUTE_MAX,
        relative_max=OPTIMIZER_SCALAR_RELATIVE_MAX,
        mismatches=mismatches,
    )
    for key, limits in OPTIMIZER_SIGNATURE_LIMITS.items():
        _validate_signature_limits(
            local_signature_divergences[key],
            path=f"optimizer_step_records[{index}].numerical.{key}",
            limits=limits,
            mismatches=mismatches,
        )
        signature_divergences[key].merge(local_signature_divergences[key])
    optimizer_scalar_divergence.merge(local_scalar_divergence)


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
    trunk_dense_file: Path | None = None,
    full_dense_file: Path | None = None,
) -> dict[str, Any]:
    """比较两条独立轨迹，并用双向 replay 解释自然 recycle 分离."""

    mismatches: list[str] = []
    voxel_output_divergence = SignatureDivergence()
    common_loss_divergence = ScalarDivergence()
    accumulated_gradient_divergence = SignatureDivergence()
    accumulated_norm_divergence = ScalarDivergence()
    optimizer_scalar_divergence = ScalarDivergence()
    optimizer_signature_divergences = {
        key: SignatureDivergence() for key in OPTIMIZER_SIGNATURE_LIMITS
    }
    trunk_dense_reader = DenseTensorReader(trunk_dense_file)
    full_dense_reader = DenseTensorReader(full_dense_file)
    _validate_trace_identity(
        trunk,
        full,
        expected_full_source_identity=expected_full_source_identity,
        expected_trunk_project_source_sha256=expected_trunk_project_source_sha256,
        expected_full_project_source_sha256=expected_full_project_source_sha256,
        mismatches=mismatches,
    )
    _validate_dense_tensor_store(
        trunk,
        role="trunk",
        dense_file=trunk_dense_file,
        mismatches=mismatches,
    )
    _validate_dense_tensor_store(
        full,
        role="full",
        dense_file=full_dense_file,
        mismatches=mismatches,
    )
    _validate_parameter_partition_and_probe(trunk, full, mismatches=mismatches)
    _validate_microbatch_loss_ranges(trunk, role="trunk", mismatches=mismatches)
    _validate_microbatch_loss_ranges(full, role="full", mismatches=mismatches)
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
            voxel_output_divergence=voxel_output_divergence,
            common_loss_divergence=common_loss_divergence,
            gradient_divergence=accumulated_gradient_divergence,
            accumulated_norm_divergence=accumulated_norm_divergence,
            batch_size=int(trunk["batch_size"]),
            accumulate_steps=int(trunk["accumulate_steps"]),
            trunk_dense_reader=trunk_dense_reader,
            full_dense_reader=full_dense_reader,
            mismatches=mismatches,
        )

    trunk_steps = list(trunk["optimizer_step_records"])
    full_steps = list(full["optimizer_step_records"])
    expected_step_count = int(trunk["optimizer_steps"])
    for role, steps, microbatches in (
        ("trunk", trunk_steps, trunk_microbatches),
        ("full", full_steps, full_microbatches),
    ):
        if len(steps) != expected_step_count:
            mismatches.append(
                f"{role}.optimizer_step_records: {len(steps)} != {expected_step_count}"
            )
        _validate_accumulated_gradient_step_relation(
            microbatches,
            steps,
            role=role,
            mismatches=mismatches,
        )
        if steps:
            voxel_names = set(steps[0]["parameters_before_step"])
            _validate_optimizer_state_transitions(
                steps,
                names=voxel_names,
                gradient_prefix="",
                state_key="optimizer_state_step",
                path=f"{role}.optimizer_state_transitions",
                mismatches=mismatches,
            )
            if role == "full":
                other_names = set(steps[0]["other_parameters_before_step"])
                _validate_optimizer_state_transitions(
                    steps,
                    names=other_names,
                    gradient_prefix="other_",
                    state_key="other_optimizer_state_step",
                    path="full.other_optimizer_state_transitions",
                    mismatches=mismatches,
                )
    if len(trunk_steps) != len(full_steps):
        mismatches.append("optimizer_step_records: 数量不同")
    else:
        for index, (trunk_step, full_step) in enumerate(
            zip(trunk_steps, full_steps, strict=True)
        ):
            if recycle_divergence is not None and int(
                trunk_step["last_microbatch"]
            ) >= (recycle_divergence):
                break
            _compare_optimizer_step(
                trunk_step,
                full_step,
                index=index,
                expected_last_microbatch=(index + 1) * int(trunk["accumulate_steps"])
                - 1,
                atol=atol,
                rtol=rtol,
                optimizer_scalar_divergence=optimizer_scalar_divergence,
                signature_divergences=optimizer_signature_divergences,
                trunk_dense_reader=trunk_dense_reader,
                full_dense_reader=full_dense_reader,
                mismatches=mismatches,
            )

    numerical_diagnostics = {
        "voxel_outputs": voxel_output_divergence.report(),
        "common_losses": common_loss_divergence.report(),
        "accumulated_voxel_gradients": accumulated_gradient_divergence.report(),
        "accumulated_voxel_gradient_norm": accumulated_norm_divergence.report(),
        "optimizer_scalars": optimizer_scalar_divergence.report(),
        "optimizer_signatures": {
            key: optimizer_signature_divergences[key].report()
            for key in OPTIMIZER_SIGNATURE_LIMITS
        },
    }
    numerical_limits = {
        "voxel_outputs": VOXEL_OUTPUT_LIMITS,
        "common_losses": {
            "absolute_max": COMMON_LOSS_ABSOLUTE_MAX,
            "relative_max": COMMON_LOSS_RELATIVE_MAX,
        },
        "accumulated_voxel_gradients": ACCUMULATED_GRADIENT_LIMITS,
        "accumulated_voxel_gradient_norm": {
            "absolute_max": ACCUMULATED_NORM_ABSOLUTE_MAX,
            "relative_max": ACCUMULATED_NORM_RELATIVE_MAX,
        },
        "optimizer_scalars": {
            "absolute_max": OPTIMIZER_SCALAR_ABSOLUTE_MAX,
            "relative_max": OPTIMIZER_SCALAR_RELATIVE_MAX,
        },
        "optimizer_signatures": OPTIMIZER_SIGNATURE_LIMITS,
        "loss_value_range": [0.0, MAX_LOSS_VALUE],
    }
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "passed": not mismatches,
        "track": trunk["track"],
        "atol": atol,
        "rtol": rtol,
        "first_recycle_divergence": recycle_divergence,
        "eligible_microbatches": eligible_microbatches,
        "recycle_sequence_source_sha256": trunk.get("recycle_sequence_source_sha256"),
        "counterfactual_replay_sources": sorted(
            str(comparison.get("recycle_sequence_source_sha256"))
            for comparison in counterfactual_replays
            if comparison.get("passed") and comparison.get("track") == "replay"
        ),
        "trace_identity": {
            "trunk": _trace_identity(trunk),
            "full": _trace_identity(full),
        },
        "numerical_limits": numerical_limits,
        "numerical_diagnostics": numerical_diagnostics,
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
    dense_files = []
    for trace_path, trace in ((trunk_path, trunk), (full_path, full)):
        record = trace.get("dense_tensor_store")
        file_name = record.get("file_name") if isinstance(record, Mapping) else None
        if isinstance(file_name, str) and Path(file_name).name == file_name:
            dense_files.append(trace_path.parent / file_name)
        else:
            dense_files.append(None)
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
        trunk_dense_file=dense_files[0],
        full_dense_file=dense_files[1],
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
