from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processedPDB_EMDB_binder.utils.mrc_tools import load_map

DEFAULT_SIM_MAP_KEYS = ("sim_map_path", "sim_map_path_cryoatom")


@dataclass(frozen=True)
class DensityGridMeta:
    """
    记录推理路径中密度图重采样后的几何元数据。
    输入参数:
        - shape_zyx: tuple[int, int, int], 重采样后密度图形状, 轴顺序为 z/y/x
        - voxel_size_xyz: list[float], (3,), 重采样后体素大小, 轴顺序为 x/y/z
        - origin_xyz: list[float], (3,), 重采样后世界坐标原点, 轴顺序为 x/y/z
    """

    shape_zyx: tuple[int, int, int]
    voxel_size_xyz: list[float]
    origin_xyz: list[float]


@dataclass(frozen=True)
class DensityPairValidation:
    """
    表示一对真实密度图和模拟密度图的几何兼容性校验结果。
    输入参数:
        - ok: bool, 当前配对是否通过校验
        - map_path: str, 真实密度图路径
        - sim_map_path: str, 模拟密度图路径
        - sim_map_key: str, 样本 JSON 中对应的模拟图字段名
        - reason: str, 校验失败原因; 通过时为空字符串
        - exp_meta: DensityGridMeta | None, 真实密度图重采样后元数据
        - sim_meta: DensityGridMeta | None, 模拟密度图重采样后元数据
    """

    ok: bool
    map_path: str
    sim_map_path: str
    sim_map_key: str
    reason: str
    exp_meta: DensityGridMeta | None
    sim_meta: DensityGridMeta | None


def resolve_n_jobs(n_jobs: int) -> int:
    """
    将命令行并行数解析为实际 joblib worker 数。
    输入参数:
        - n_jobs: int, 标量; 0 表示读取 SLURM_CPUS_PER_TASK, 负数表示使用 os.cpu_count()

    输出:
        - resolved_n_jobs: int, 标量, 至少为 1 的实际并行 worker 数
    """
    if int(n_jobs) == 0:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if int(n_jobs) < 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(n_jobs))


def _model_grid_meta_from_raw(
    grid_shape_zyx: tuple[int, int, int],
    voxel_size_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    target_voxel_size: float,
) -> DensityGridMeta:
    """
    复刻 make_model_grid 的几何元数据计算, 但不执行 Fourier 重采样。
    输入参数:
        - grid_shape_zyx: tuple[int, int, int], 原始 load_map 后的密度图形状
        - voxel_size_xyz: np.ndarray, (3,), 原始体素大小, 轴顺序 x/y/z
        - origin_xyz: np.ndarray, (3,), 原始世界坐标原点, 轴顺序 x/y/z
        - target_voxel_size: float, 推理配置中的目标体素大小

    输出:
        - meta: DensityGridMeta, 与当前推理路径几何口径一致的重采样后元数据
    """
    # np.ndarray[int64], (3,), make_cubic 后的 z/y/x 形状
    raw_shape_zyx = np.asarray(grid_shape_zyx, dtype=np.int64)
    cubic_shape_zyx = raw_shape_zyx + raw_shape_zyx % 2
    shift_zyx = cubic_shape_zyx // 2 - raw_shape_zyx // 2
    origin_after_cubic = origin_xyz - shift_zyx.astype(np.float64) * voxel_size_xyz

    # np.ndarray[int64], (3,), normalize_voxel_size 内部使用的 x/y/z 形状
    in_size_xyz = np.asarray(
        [cubic_shape_zyx[2], cubic_shape_zyx[1], cubic_shape_zyx[0]],
        dtype=np.float64,
    )
    out_size_xyz = np.ceil(in_size_xyz * voxel_size_xyz / float(target_voxel_size)).astype(np.int64)
    for axis_index, axis_size in enumerate(out_size_xyz):
        if int(axis_size) % 2 == 0:
            continue
        voxel_if_plus = voxel_size_xyz[axis_index] * in_size_xyz[axis_index] / float(axis_size + 1)
        voxel_if_minus = voxel_size_xyz[axis_index] * in_size_xyz[axis_index] / float(axis_size - 1)
        if abs(voxel_if_plus - float(target_voxel_size)) < abs(voxel_if_minus - float(target_voxel_size)):
            out_size_xyz[axis_index] = axis_size + 1
        else:
            out_size_xyz[axis_index] = axis_size - 1

    voxel_size_after_resample = voxel_size_xyz * in_size_xyz / out_size_xyz.astype(np.float64)
    shape_after_resample_zyx = (
        int(out_size_xyz[2]),
        int(out_size_xyz[1]),
        int(out_size_xyz[0]),
    )
    return DensityGridMeta(
        shape_zyx=shape_after_resample_zyx,
        voxel_size_xyz=[float(v) for v in voxel_size_after_resample],
        origin_xyz=[float(v) for v in origin_after_cubic],
    )


def load_inference_grid_meta(map_path: str, target_voxel_size: float) -> DensityGridMeta:
    """
    按当前推理读取口径提取单张密度图的重采样后几何元数据。
    输入参数:
        - map_path: str, 标量, MRC/MAP 文件路径
        - target_voxel_size: float, 推理配置中的目标体素大小

    输出:
        - meta: DensityGridMeta, 重采样后 shape/voxel_size/origin 元数据
    """
    grid_raw, voxel_size_raw, origin_raw = load_map(map_path)
    return _model_grid_meta_from_raw(
        grid_shape_zyx=tuple(int(v) for v in grid_raw.shape),
        voxel_size_xyz=np.asarray(voxel_size_raw, dtype=np.float64).reshape(3),
        origin_xyz=np.asarray(origin_raw, dtype=np.float64).reshape(3),
        target_voxel_size=float(target_voxel_size),
    )


def validate_density_map_pair(
    map_path: str,
    sim_map_path: str,
    sim_map_key: str,
    target_voxel_size: float,
    origin_atol: float,
    voxel_rtol: float,
    voxel_atol: float,
) -> DensityPairValidation:
    """
    校验真实密度图和模拟密度图在当前推理口径下是否几何一致。
    输入参数:
        - map_path: str, 标量, 真实密度图路径
        - sim_map_path: str, 标量, 模拟密度图路径
        - sim_map_key: str, 标量, 样本 JSON 中的模拟图字段名
        - target_voxel_size: float, 推理配置中的目标体素大小
        - origin_atol: float, origin 绝对误差容忍度
        - voxel_rtol: float, voxel_size 相对误差容忍度
        - voxel_atol: float, voxel_size 绝对误差容忍度

    输出:
        - result: DensityPairValidation, 当前配对的校验结果和必要元数据
    """
    try:
        exp_meta = load_inference_grid_meta(map_path, target_voxel_size)
        sim_meta = load_inference_grid_meta(sim_map_path, target_voxel_size)
    except Exception as exc:
        return DensityPairValidation(
            ok=False,
            map_path=str(map_path),
            sim_map_path=str(sim_map_path),
            sim_map_key=str(sim_map_key),
            reason=f"load_error: {type(exc).__name__}: {exc}",
            exp_meta=None,
            sim_meta=None,
        )

    if exp_meta.shape_zyx != sim_meta.shape_zyx:
        reason = f"shape mismatch: exp={exp_meta.shape_zyx}, sim={sim_meta.shape_zyx}"
    elif not np.allclose(exp_meta.voxel_size_xyz, sim_meta.voxel_size_xyz, rtol=float(voxel_rtol), atol=float(voxel_atol)):
        reason = f"voxel_size mismatch: exp={exp_meta.voxel_size_xyz}, sim={sim_meta.voxel_size_xyz}"
    elif not np.allclose(exp_meta.origin_xyz, sim_meta.origin_xyz, rtol=1e-5, atol=float(origin_atol)):
        reason = f"origin mismatch: exp={exp_meta.origin_xyz}, sim={sim_meta.origin_xyz}"
    else:
        reason = ""

    return DensityPairValidation(
        ok=reason == "",
        map_path=str(map_path),
        sim_map_path=str(sim_map_path),
        sim_map_key=str(sim_map_key),
        reason=reason,
        exp_meta=exp_meta,
        sim_meta=sim_meta,
    )


def validate_sample_density_maps(
    sample: dict[str, Any],
    target_voxel_size: float,
    origin_atol: float,
    voxel_rtol: float,
    voxel_atol: float,
    sim_map_keys: tuple[str, ...] = DEFAULT_SIM_MAP_KEYS,
) -> list[DensityPairValidation]:
    """
    校验单个样本中所有已存在的模拟密度图字段。
    输入参数:
        - sample: dict[str, Any], 单个 JSON 样本条目, 至少包含 map_path
        - target_voxel_size: float, 推理配置中的目标体素大小
        - origin_atol: float, origin 绝对误差容忍度
        - voxel_rtol: float, voxel_size 相对误差容忍度
        - voxel_atol: float, voxel_size 绝对误差容忍度
        - sim_map_keys: tuple[str, ...], 需要尝试校验的模拟图字段名

    输出:
        - results: list[DensityPairValidation], 可变长度, 每个已存在模拟图字段对应一个校验结果
    """
    results: list[DensityPairValidation] = []
    for sim_map_key in sim_map_keys:
        sim_map_path = sample.get(sim_map_key)
        if sim_map_path is None:
            continue
        results.append(
            validate_density_map_pair(
                map_path=str(sample["map_path"]),
                sim_map_path=str(sim_map_path),
                sim_map_key=str(sim_map_key),
                target_voxel_size=float(target_voxel_size),
                origin_atol=float(origin_atol),
                voxel_rtol=float(voxel_rtol),
                voxel_atol=float(voxel_atol),
            )
        )
    return results


def _validate_for_cli(
    sample: dict[str, Any],
    target_voxel_size: float,
    origin_atol: float,
    voxel_rtol: float,
    voxel_atol: float,
    sim_map_keys: tuple[str, ...],
) -> dict[str, Any]:
    """
    为命令行批量校验构造可 JSON 序列化的单样本结果。
    输出:
        - result: dict[str, Any], 包含 sample、ok 和 validations 三个字段
    """
    validations = validate_sample_density_maps(
        sample=sample,
        target_voxel_size=target_voxel_size,
        origin_atol=origin_atol,
        voxel_rtol=voxel_rtol,
        voxel_atol=voxel_atol,
        sim_map_keys=sim_map_keys,
    )
    return {
        "sample": sample,
        "ok": all(item.ok for item in validations),
        "validations": [asdict(item) for item in validations],
    }


def main() -> None:
    """
    命令行入口: 过滤 JSON 中真实密度图和模拟密度图几何不匹配的样本。
    """
    parser = argparse.ArgumentParser(description="过滤真实密度图与模拟密度图几何不一致的 JSON 样本")
    parser.add_argument("--input_json", required=True, help="输入样本 JSON 路径")
    parser.add_argument("--output_json", required=True, help="过滤后的输出 JSON 路径")
    parser.add_argument("--report_json", default=None, help="可选, 写出逐样本校验报告")
    parser.add_argument("--target_voxel_size", type=float, default=1.0, help="推理重采样目标体素大小")
    parser.add_argument("--origin_atol", type=float, default=1e-4, help="origin 绝对误差容忍度")
    parser.add_argument("--voxel_rtol", type=float, default=1e-5, help="voxel_size 相对误差容忍度")
    parser.add_argument("--voxel_atol", type=float, default=1e-6, help="voxel_size 绝对误差容忍度")
    parser.add_argument("--n_jobs", type=int, default=0, help="joblib 并行数; 0 表示读取 SLURM_CPUS_PER_TASK")
    parser.add_argument("--sim_map_keys", nargs="+", default=list(DEFAULT_SIM_MAP_KEYS), help="需要校验的模拟图字段名")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8-sig") as file_obj:
        samples = json.load(file_obj)
    if not isinstance(samples, list):
        raise TypeError("input_json 顶层必须是 list[dict]")

    resolved_n_jobs = resolve_n_jobs(args.n_jobs)
    sim_map_keys = tuple(str(key) for key in args.sim_map_keys)
    if resolved_n_jobs == 1:
        results = [
            _validate_for_cli(sample, args.target_voxel_size, args.origin_atol, args.voxel_rtol, args.voxel_atol, sim_map_keys)
            for sample in samples
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=resolved_n_jobs, backend="loky")(
            delayed(_validate_for_cli)(sample, args.target_voxel_size, args.origin_atol, args.voxel_rtol, args.voxel_atol, sim_map_keys)
            for sample in samples
        )

    accepted_samples = [item["sample"] for item in results if bool(item["ok"])]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as file_obj:
        json.dump(accepted_samples, file_obj, ensure_ascii=False, indent=4)

    if args.report_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, ensure_ascii=False, indent=2)

    failed_count = len(samples) - len(accepted_samples)
    print(f"[density-map-validation] total={len(samples)} accepted={len(accepted_samples)} failed={failed_count} n_jobs={resolved_n_jobs}")


if __name__ == "__main__":
    main()
