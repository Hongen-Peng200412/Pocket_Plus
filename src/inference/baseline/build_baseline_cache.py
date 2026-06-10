from __future__ import annotations

"""
非深度学习 baseline 缓存生成过渡脚本。

职责(只做这些, 不做后处理/扫参/评估):
    1. 解析配置和样本 JSON(protein_40 / protein_110)。
    2. 复用 DL 管线的 load_from_raw_cif + GT loader, 拿到 hardmask / resampled_emdb / resampled_sim / origin / voxel_size / GT。
    3. 生成 baseline 整卷 raw score map:
        - density-channel baseline: build_density_channels 整卷一次性计算单通道(receptor_mask=hardmask>0)。
        - phenix baseline: 按 A2 约定派生路径读取已对齐到 cache 网格的 phenix 差图(对齐由 phenix 专属脚本负责, 详见 docs/test_pipline/phenix.md)。
    4. 对 raw score 做每样本保序直方图均衡到 [0,1](valid=hardmask==0, atom 区置 0)。
    5. 写出与 DL 预测缓存兼容的 .npz(ligand_pred=均衡后 score, receptor_pred=None, GT 同 DL)。
    6. 顺手写一份 raw 差图 sidecar MRC, 路径记入 meta["raw_diff_map_path"], 供统一可视化加载。

用法(单组 baseline×system×split):
    python src/inference/baseline/build_baseline_cache.py \
        --config configs/infer_or_eval/non_DL/baseline_base_stardard.yaml \
        baseline_name=posdiff_clipnorm_DoG1 system=stardard
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from src.datasets.density_channel_builder import DensityChannelConfig, build_density_channels
from src.inference.parse_input import load_from_raw_cif
from src.inference.voxel_gt import load_ligand_gt_from_labels_npz, load_ligand_gt_from_structure
from src.inference.voxel_tuning import save_voxel_prediction_cache
from src.inference.utils.utils import write_grid_as_map
from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs
from src.inference.main.voxel_pipeline import (
    _get_cfg,
    _gt_id_to_name,
    _merge_sample_pair_cfg,
    _resolve_cache_path,
    _resolve_forward_structure_path,
    _resolve_gt_receptor_path,
    _resolve_gt_source,
    _resolve_path_by_cfg_key,
    _resolve_sample_name,
)

# str, phenix 差图 baseline 名; 与 density-channel baseline 区分处理
PHENIX_BASELINE_NAME = "phenix_real_space_diff_map"
# list[str], 5 组 density-channel baseline 名; 必须与 density_channel_builder 支持的通道名一致
DENSITY_CHANNEL_BASELINES = [
    "diff_clipnorm_nopost",
    "posdiff_clipnorm_DoG1",
    "posdiff_clipnorm_DoG2",
    "posdiff_clipnorm_smooth1",
    "posdiff_clipnorm_smooth2",
]
# list[str], 固定六组 baseline 名
ALL_BASELINE_NAMES = [PHENIX_BASELINE_NAME, *DENSITY_CHANNEL_BASELINES]
# str, phenix 对齐差图的固定文件名(A2 约定派生; 生成端写入、消费端按同规则读取)
PHENIX_ALIGNED_MAP_NAME = "phenix_diff_aligned.mrc"


def derive_phenix_map_path(phenix_output_root: str, system: str, sample_name: str) -> str:
    """
    按约定派生 phenix 对齐差图路径(A2: 生成端与消费端共用此一处规则, JSON 不写 phenix 字段)。

    输入参数:
        - phenix_output_root: str, phenix 预生成产物根目录
        - system: str, 系统名(stardard / strict)
        - sample_name: str, 样本名

    输出:
        - map_path: str, 对齐到 cache 网格的 phenix 差图路径
    """
    return str(Path(phenix_output_root) / system / sample_name / PHENIX_ALIGNED_MAP_NAME)


def per_sample_rank_equalize(raw_score: np.ndarray, hardmask: np.ndarray) -> np.ndarray:
    """
    每样本内部保序直方图均衡: 把有效区 raw score 映射到 [0,1], atom 区置 0。

    映射: valid=hardmask==0; score_eq[valid]=(rank-1)/(N-1)(rank 用 average 处理 ties, 严格保序); score_eq[hardmask>0]=0。

    输入参数:
        - raw_score: np.ndarray, (D,H,W), baseline 原始 score map(可含负值)
        - hardmask: np.ndarray, (D,H,W), 受体原子占据掩码(0 为有效区)

    输出:
        - score_eq: np.ndarray, (D,H,W), float32, 均衡后 score, 取值范围 [0,1]
    """
    # np.ndarray, (D,H,W), float64, raw score
    raw_score = np.asarray(raw_score, dtype=np.float64)
    # np.ndarray, (D,H,W), bool, 有效区(非受体原子)
    valid_mask = np.asarray(hardmask) == 0
    if raw_score.shape != valid_mask.shape:
        raise ValueError(f"raw_score.shape={raw_score.shape} 与 hardmask.shape={valid_mask.shape} 不一致")
    # np.ndarray, (D,H,W), float32, 均衡输出; atom 区默认 0
    score_eq = np.zeros(raw_score.shape, dtype=np.float32)
    # np.ndarray, (N,), float64, 有效区 raw score
    valid_values = raw_score[valid_mask]
    # int, 有效体素数
    n_valid = int(valid_values.size)
    if n_valid == 0:
        raise ValueError("有效体素数为 0(hardmask 全占据), 无法做直方图均衡")
    if n_valid == 1:
        score_eq[valid_mask] = 1.0
        return score_eq
    # np.ndarray, (N,), float64, average 秩, 取值 [1, N]
    ranks = rankdata(valid_values, method="average")
    score_eq[valid_mask] = ((ranks - 1.0) / (float(n_valid) - 1.0)).astype(np.float32)
    return score_eq


def _compute_baseline_raw_score(
    sample_cfg: dict[str, Any],
    baseline_name: str,
    raw_data: dict[str, Any],
) -> np.ndarray:
    """
    计算 baseline 整卷 raw score map(均衡前)。

    输入参数:
        - sample_cfg: dict[str, Any], 当前样本配置; phenix baseline 需含 phenix_output_root/system(按约定派生差图路径)
        - baseline_name: str, baseline 名; phenix 或 density-channel 通道名
        - raw_data: dict[str, Any], load_from_raw_cif 输出, 提供 resampled_emdb/resampled_sim/hardmask

    输出:
        - raw_score: np.ndarray, (D,H,W), float32, baseline 原始 score map
    """
    if baseline_name == PHENIX_BASELINE_NAME:
        from processedPDB_EMDB_binder.utils.mrc_tools import load_map

        # str, 已对齐到 cache 网格的 phenix 差图路径(A2 约定派生; 由 generate_phenix_diff_maps.py 预生成, 详见 phenix.md)
        phenix_diff_map_path = derive_phenix_map_path(
            phenix_output_root=str(_get_cfg(sample_cfg, "phenix_output_root", True)),
            system=str(_get_cfg(sample_cfg, "system", True)),
            sample_name=_resolve_sample_name(sample_cfg),
        )
        if not os.path.exists(phenix_diff_map_path):
            raise FileNotFoundError(f"phenix 对齐差图不存在: {phenix_diff_map_path}; 请先运行 generate_phenix_diff_maps.py")
        # np.ndarray, (D,H,W), phenix 差图体素数据(对齐性不在此断言, 详见 phenix.md)
        grid_raw, _, _ = load_map(phenix_diff_map_path)
        return np.asarray(grid_raw, dtype=np.float32)

    if baseline_name not in DENSITY_CHANNEL_BASELINES:
        raise ValueError(f"未知 baseline_name: {baseline_name}; 允许 {ALL_BASELINE_NAMES}")
    if raw_data["resampled_sim"] is None:
        raise ValueError(f"density-channel baseline {baseline_name} 需要 resampled_sim, 但当前为 None")
    # np.ndarray, (1,D,H,W), float32, 整卷单通道 baseline score(receptor_mask=hardmask>0)
    channels = build_density_channels(
        exp_raw=raw_data["resampled_emdb"],
        sim_raw=raw_data["resampled_sim"],
        config=DensityChannelConfig(enabled_channels=[baseline_name]),
        receptor_mask=(np.asarray(raw_data["hardmask"]) > 0),
    )
    return np.asarray(channels[0], dtype=np.float32)


def _baseline_density_channel_config(baseline_name: str) -> dict[str, Any]:
    """
    构造传给 load_from_raw_cif 的 density_channel_config, 用于决定是否加载模拟密度图。

    输入参数:
        - baseline_name: str, baseline 名

    输出:
        - density_channel_config: dict[str, Any], enabled_channels 控制 needs_sim; phenix 仅触发 exp, 不需 sim
    """
    # str, 触发 sim 加载的通道名; density baseline 直接用自身通道名, phenix 用 exp 通道(不需 sim)
    trigger_channel = "exp_clipnorm_nopost" if baseline_name == PHENIX_BASELINE_NAME else baseline_name
    return {
        "clip_percentile": (0.001, 0.999),
        "fit_mask_percentile": 0.003,
        "enabled_channels": [trigger_channel],
    }


def build_one_baseline_cache(
    sample_cfg: dict[str, Any],
    baseline_name: str,
    system: str,
) -> str:
    """
    为单个样本生成一份 baseline 缓存(.npz)与 raw 差图 sidecar MRC。

    输入参数:
        - sample_cfg: dict[str, Any], 当前样本配置(已合并 pair 字段)
        - baseline_name: str, baseline 名
        - system: str, 系统名(stardard / strict), 仅写入 meta

    输出:
        - cache_path: str, 生成或复用的 .npz 缓存路径
    """
    # str, 当前样本名
    sample_name = _resolve_sample_name(sample_cfg)
    # str, 当前样本缓存路径
    cache_path = _resolve_cache_path(sample_cfg, sample_name)
    if bool(_get_cfg(sample_cfg, "use_cache", True)) and os.path.exists(cache_path):
        return cache_path

    # str, 当前 system 的结构输入路径(structure_input_source 选择)
    effective_cif_path = _resolve_forward_structure_path(sample_cfg)
    # dict[str, Any], 控制 load_from_raw_cif 是否加载 sim 的密度通道配置
    density_channel_config = _baseline_density_channel_config(baseline_name)
    # str | None, 当前 system 的模拟密度图路径; phenix baseline 不需要 sim
    effective_sim_map_path = None if baseline_name == PHENIX_BASELINE_NAME else _resolve_path_by_cfg_key(sample_cfg, "sim_map_source")

    raw_data = load_from_raw_cif(
        cif_path=effective_cif_path,
        map_path=str(_get_cfg(sample_cfg, "map_path", True)),
        sim_map_path=effective_sim_map_path,
        target_voxel_size=float(_get_cfg(sample_cfg, "target_voxel_size", True)),
        compute_density=bool(_get_cfg(sample_cfg, "compute_density", True)),
        select_first_model=bool(_get_cfg(sample_cfg, "select_first_model", True)),
        error_dir=_get_cfg(sample_cfg, "error_dir", False),
        density_channel_config=density_channel_config,
    )

    # np.ndarray, (D,H,W), float32, baseline 原始 score map(均衡前)
    raw_score = _compute_baseline_raw_score(sample_cfg, baseline_name, raw_data)
    if raw_score.shape != tuple(int(v) for v in raw_data["full_shape_zyx"]):
        raise ValueError(f"baseline raw score shape={raw_score.shape} 与 cache 网格 {raw_data['full_shape_zyx']} 不一致")
    # np.ndarray, (D,H,W), float32, 均衡后 ligand_pred(取值 [0,1])
    ligand_pred = per_sample_rank_equalize(raw_score, raw_data["hardmask"])

    # str, GT 来源类型, 与 DL 同口径
    gt_source = _resolve_gt_source(sample_cfg)
    # dict[str, Any], voxel GT 数据; eval_gt=false 时保持空值
    gt_data: dict[str, Any] = {
        "gt_ligand_mask": None,
        "gt_instance_label": None,
        "gt_ligand_mask_by_class_id": {},
        "gt_instance_label_by_class_id": {},
        "gt_instance_meta": None,
    }
    if gt_source == "labels_npz":
        gt_data = load_ligand_gt_from_labels_npz(
            labels_npz_path=str(_get_cfg(sample_cfg, "labels_npz_path", True)),
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
            class_mapping=_get_cfg(sample_cfg, "class_mapping", False),
            ligand_gt_distance_threshold=float(_get_cfg(sample_cfg, "ligand_gt_distance_threshold", True)),
        )
    elif gt_source == "structure":
        gt_data = load_ligand_gt_from_structure(
            ligand_structure_path=str(_get_cfg(sample_cfg, "cif_gt_path", True)),
            receptor_structure_path=_resolve_gt_receptor_path(sample_cfg),
            filter_preset=str(_get_cfg(sample_cfg, "filter_preset", True)),
            class_mapping=_get_cfg(sample_cfg, "class_mapping", False),
            select_first_model=bool(_get_cfg(sample_cfg, "select_first_model", True)),
            error_dir=_get_cfg(sample_cfg, "error_dir", False),
            origin=raw_data["origin"],
            voxel_size=raw_data["voxel_size"],
            grid_shape_zyx=tuple(int(v) for v in raw_data["full_shape_zyx"]),
            ligand_gt_distance_threshold=float(_get_cfg(sample_cfg, "ligand_gt_distance_threshold", True)),
        )

    # list[str], (C,), 当前任务类别名; 二分类默认 background/foreground
    class_names = [str(v) for v in (_get_cfg(sample_cfg, "class_names", False) or ["background", "foreground"])]
    # tuple[dict|None, dict|None], 类别名粒度逐类 GT
    gt_ligand_mask_by_class, gt_instance_label_by_class = _gt_id_to_name(gt_data, class_names)

    # str | None, raw 差图 sidecar MRC 路径; vis_output_root 为空时不写
    raw_diff_map_path = None
    vis_output_root = _get_cfg(sample_cfg, "vis_output_root", False)
    if vis_output_root is not None:
        # str, sidecar MRC 输出目录(与 best_outputs vis 同根, 按 sample 分目录)
        sample_vis_dir = os.path.join(str(vis_output_root), sample_name)
        os.makedirs(sample_vis_dir, exist_ok=True)
        raw_diff_map_path = write_grid_as_map(
            raw_score,
            os.path.join(sample_vis_dir, "raw_diff.mrc"),
            raw_data["origin"],
            raw_data["voxel_size"],
        )

    meta = {
        "sample_name": sample_name,
        "cache_path": cache_path,
        "baseline_name": baseline_name,
        "system": system,
        "score_source": baseline_name,
        "cif_path": effective_cif_path,
        "map_path": str(_get_cfg(sample_cfg, "map_path", True)),
        "sim_map_path": effective_sim_map_path,
        "cif_gt_path": _get_cfg(sample_cfg, "cif_gt_path", False),
        "labels_npz_path": _get_cfg(sample_cfg, "labels_npz_path", False),
        "phenix_diff_map_path": (
            derive_phenix_map_path(
                phenix_output_root=str(_get_cfg(sample_cfg, "phenix_output_root", True)),
                system=str(_get_cfg(sample_cfg, "system", True)),
                sample_name=sample_name,
            )
            if baseline_name == PHENIX_BASELINE_NAME
            else None
        ),
        "gt_source": gt_source,
        "class_names": class_names,
        "class_mapping": _get_cfg(sample_cfg, "class_mapping", False),
        "raw_diff_map_path": raw_diff_map_path,
    }

    save_voxel_prediction_cache(
        cache_path=cache_path,
        ligand_pred=ligand_pred,
        receptor_pred=None,
        hardmask=raw_data["hardmask"],
        resampled_emdb=raw_data["resampled_emdb"],
        origin=raw_data["origin"],
        voxel_size=raw_data["voxel_size"],
        meta=meta,
        gt_ligand_mask=gt_data["gt_ligand_mask"],
        gt_instance_label=gt_data["gt_instance_label"],
        gt_ligand_mask_by_class=gt_ligand_mask_by_class,
        gt_instance_label_by_class=gt_instance_label_by_class,
        gt_instance_meta=gt_data.get("gt_instance_meta"),
    )
    return cache_path


def build_baseline_cache_for_split(
    cfg_dict: dict[str, Any],
    baseline_name: str,
    system: str,
) -> list[str]:
    """
    对 cfg_dict.raw_pairs_json 指向的整个 split 生成 baseline 缓存。

    输入参数:
        - cfg_dict: dict[str, Any], baseline 配置; 含 raw_pairs_json / cache_root / 各 system 输入字段
        - baseline_name: str, baseline 名
        - system: str, 系统名(stardard / strict)

    输出:
        - cache_paths: list[str], 可变长度, 生成或复用的缓存路径列表
    """
    # list[dict[str, str|None]], 样本 pair 列表
    pairs = load_raw_pairs(str(_get_cfg(cfg_dict, "raw_pairs_json", True)))
    # list[dict[str, Any]], 当前 split 全部样本配置
    sample_cfgs = [_merge_sample_pair_cfg(cfg_dict, pair) for pair in pairs]
    # int, cache 构建并行 worker 数; 默认 1 保持原串行语义
    cache_n_jobs = int(_get_cfg(cfg_dict, "cache_n_jobs", False) or 1)
    # str, joblib 后端; 用户当前选择 loky 以使用多进程
    cache_backend = str(_get_cfg(cfg_dict, "cache_backend", False) or "loky")
    if cache_n_jobs <= 1:
        return [build_one_baseline_cache(sample_cfg, baseline_name, system) for sample_cfg in sample_cfgs]
    print(
        f"[build_baseline_cache] baseline={baseline_name} system={system} "
        f"samples={len(sample_cfgs)} cache_n_jobs={cache_n_jobs} cache_backend={cache_backend}",
        flush=True,
    )
    return Parallel(n_jobs=cache_n_jobs, backend=cache_backend)(
        delayed(build_one_baseline_cache)(sample_cfg, baseline_name, system)
        for sample_cfg in sample_cfgs
    )


def _load_cfg(config_name_or_path: str, overrides: list[str]) -> dict[str, Any]:
    """
    加载 baseline 配置并合并命令行覆盖项。

    输入参数:
        - config_name_or_path: str, 配置名(configs/infer_or_eval 下)或 YAML 路径
        - overrides: list[str], OmegaConf dotlist 覆盖项

    输出:
        - cfg_dict: dict[str, Any], 合并后的配置字典
    """
    config_path = Path(config_name_or_path)
    if not config_path.exists():
        config_name = config_name_or_path[:-5] if config_name_or_path.endswith(".yaml") else config_name_or_path
        config_path = PROJECT_ROOT / "configs" / "infer_or_eval" / f"{config_name}.yaml"
    merged_cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(overrides))
    cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"配置必须解析为 dict, 实际为 {type(cfg_dict)}")
    return cfg_dict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="生成单组 baseline×system×split 的 DL 兼容缓存。")
    parser.add_argument("--config", required=True, help="baseline 配置名或 YAML 路径(含 raw_pairs_json / cache_root 等)。")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist 覆盖项, 至少需指定 baseline_name= 与 system=。")
    args = parser.parse_args(argv)
    cfg_dict = _load_cfg(args.config, args.overrides)
    baseline_name = str(_get_cfg(cfg_dict, "baseline_name", True))
    system = str(_get_cfg(cfg_dict, "system", True))
    cache_paths = build_baseline_cache_for_split(cfg_dict, baseline_name, system)
    print(f"[build_baseline_cache] baseline={baseline_name} system={system} 生成 {len(cache_paths)} 份缓存")


if __name__ == "__main__":
    main()
