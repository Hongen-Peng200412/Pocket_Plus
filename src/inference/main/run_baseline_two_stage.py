from __future__ import annotations

"""
非深度学习 baseline 自动化入口。

枚举 6 个 baseline_name × 2 个 system, 对每个组合:
    1. 用 derive_baseline_paths 派生 cache_root / output_root / vis_output_root / error_dir(改这一处规则即可改全部落地目录)。
    2. 调 build_baseline_cache 为 protein_40 / protein_110 预生成 DL 兼容缓存(含 raw 差图 sidecar MRC)。
    3. 复用 two_stage_basic.run_two_stage_then_fixed_test 走 40 Stage1 -> 40 Stage2 -> 110 fixed test(model=None, 命中预生成 cache, 绝不 forward)。

用法:
    python src/inference/main/run_baseline_two_stage.py \
        --base_dir /home/penghongen/My_Project/EVAL_OUT \
        [--systems stardard strict] [--baselines posdiff_clipnorm_DoG1 ...] \
        [--val_json .../protein_40.json] [--test_json .../protein_110.json] [--device cuda:0]
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = str(PROJECT_ROOT)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from src.inference.baseline.build_baseline_cache import ALL_BASELINE_NAMES, build_baseline_cache_for_split
from src.inference.main.two_stage_basic import parse_extra_test_json_specs, run_two_stage_then_fixed_test

# list[str], 两个系统
SYSTEMS = ["stardard", "strict"]
# str, 默认落地根目录(infer_cache/infer_out/infer_vis/infer_error 的父目录)
DEFAULT_BASE_DIR = "/home/penghongen/My_Project/EVAL_OUT"
# str, 默认验证集(protein_40)样本列表
DEFAULT_VAL_JSON = "/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_40.json"
# str, 默认测试集(protein_110)样本列表
DEFAULT_TEST_JSON = "/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/protein_110.json"
# str, 默认 phenix 预生成差图根目录(由 generate_phenix_diff_maps.py 写入, A2 约定派生消费)
DEFAULT_PHENIX_OUTPUT_ROOT = "/home/penghongen/My_Project/EVAL_OUT/phenix_diff_maps"


def parse_bool(value: str) -> bool:
    """
    解析命令行布尔值。

    输入参数:
        - value: str, 支持 true/false/1/0/yes/no/on/off

    输出:
        - enabled: bool, 解析后的布尔值
    """
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def derive_baseline_paths(base_dir: str, baseline_name: str, system: str, split: str) -> dict[str, str]:
    """
    由 (baseline_name, system, split) 派生当前组合的落地路径(要改目录结构只改这一处)。

    输入参数:
        - base_dir: str, 落地根目录
        - baseline_name: str, baseline 名
        - system: str, 系统名(stardard / strict)
        - split: str, 数据划分标签("40" / "110")

    输出:
        - paths: dict[str, str], 含 cache_root / output_root / vis_output_root / error_dir
    """
    # str, 当前组合标签
    tag = f"{baseline_name}_{system}_{split}"
    return {
        "cache_root": str(Path(base_dir) / "infer_cache" / "baseline" / tag),
        "output_root": str(Path(base_dir) / "infer_out" / "baseline" / tag),
        "vis_output_root": str(Path(base_dir) / "infer_vis" / "baseline" / tag),
        "error_dir": str(Path(base_dir) / "infer_error" / "baseline" / tag),
    }


def _load_system_base_cfg(system: str) -> dict[str, Any]:
    """
    加载某系统的 baseline 基础配置。

    输入参数:
        - system: str, 系统名(stardard / strict)

    输出:
        - cfg_dict: dict[str, Any], baseline_base_<system>.yaml 解析结果
    """
    # Path, baseline 基础配置路径
    config_path = PROJECT_ROOT / "configs" / "infer_or_eval" / "non_DL" / f"baseline_base_{system}.yaml"
    cfg_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"baseline 基础配置必须解析为 dict, 实际为 {type(cfg_dict)}")
    return cfg_dict


def _make_split_cfg(
    base_cfg: dict[str, Any],
    baseline_name: str,
    system: str,
    split: str,
    raw_pairs_json: str,
    base_dir: str,
    phenix_output_root: str,
    cache_n_jobs: int,
    cache_backend: str,
    vis_enable: bool,
) -> dict[str, Any]:
    """
    从系统基础配置派生某 (baseline, system, split) 的完整配置。

    输入参数:
        - base_cfg: dict[str, Any], 系统基础配置
        - baseline_name: str, baseline 名
        - system: str, 系统名
        - split: str, 数据划分标签("40" / "110")
        - raw_pairs_json: str, 当前 split 的样本列表 JSON
        - base_dir: str, 落地根目录
        - phenix_output_root: str, phenix 预生成差图根目录(仅 phenix baseline 消费)
        - cache_n_jobs: int, baseline cache 构建并行 worker 数
        - cache_backend: str, joblib cache 构建后端
        - vis_enable: bool, 是否允许样本级可视化与 raw 差图 sidecar 写出

    输出:
        - cfg: dict[str, Any], 注入 baseline 身份/路径后的配置
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["baseline_name"] = baseline_name
    cfg["system"] = system
    cfg["split_label"] = split
    cfg["raw_pairs_json"] = raw_pairs_json
    cfg["phenix_output_root"] = phenix_output_root
    cfg["cache_n_jobs"] = int(cache_n_jobs)
    cfg["cache_backend"] = str(cache_backend)
    paths = derive_baseline_paths(base_dir, baseline_name, system, split)
    if not vis_enable:
        paths["vis_output_root"] = None
    cfg.update(paths)
    cfg["pipeline_vis_enable"] = bool(vis_enable)
    cfg["vis_enable"] = bool(vis_enable)
    return cfg


def _val_cache_required(val_cfg: dict[str, Any]) -> bool:
    """
    判断验证 split 是否需要预生成 baseline cache。

    输入参数:
        - val_cfg: dict[str, Any], 验证 split 配置

    输出:
        - required: bool, Stage1 或 Stage2 best_params.json 缺失时为 True
    """
    val_root = Path(str(val_cfg["output_root"]))
    return not (
        (val_root / "stage1_threshold_only" / "best_params.json").exists()
        and (val_root / "stage2_threshold_component_policy" / "best_params.json").exists()
    )


def _fixed_test_cache_required(test_cfg: dict[str, Any]) -> bool:
    """
    判断固定测试 split 是否需要预生成 baseline cache。

    输入参数:
        - test_cfg: dict[str, Any], 固定测试配置

    输出:
        - required: bool, best_summary.json 缺失时为 True
    """
    return not (Path(str(test_cfg["output_root"])) / "best_summary.json").exists()


def run_one_baseline(
    base_cfg: dict[str, Any],
    baseline_name: str,
    system: str,
    base_dir: str,
    val_json: str,
    test_json: str,
    extra_test_json_specs: list[tuple[str, str]],
    phenix_output_root: str,
    device: torch.device,
    cache_n_jobs: int,
    cache_backend: str,
    vis_enable: bool,
) -> dict[str, str]:
    """
    执行单个 baseline×system 的完整流程: 预生成 40/110 缓存 -> 三段流程。

    输入参数:
        - base_cfg: dict[str, Any], 当前系统基础配置
        - baseline_name: str, baseline 名
        - system: str, 系统名
        - base_dir: str, 落地根目录
        - val_json: str, protein_40 样本列表
        - test_json: str, protein_110 样本列表
        - extra_test_json_specs: list[tuple[str, str]], 追加固定测试 split, 如 nucleic_40
        - phenix_output_root: str, phenix 预生成差图根目录(仅 phenix baseline 消费)
        - device: torch.device, 占位设备(baseline 不 forward)
        - cache_n_jobs: int, baseline cache 构建并行 worker 数
        - cache_backend: str, joblib cache 构建后端
        - vis_enable: bool, 是否允许样本级可视化与 raw 差图 sidecar 写出

    输出:
        - result: dict[str, str], run_two_stage_then_fixed_test 的输出目录与摘要路径
    """
    # dict[str, Any], 验证(40)与测试(110)配置
    val_cfg = _make_split_cfg(base_cfg, baseline_name, system, "40", val_json, base_dir, phenix_output_root, cache_n_jobs, cache_backend, vis_enable)
    test_cfg = _make_split_cfg(base_cfg, baseline_name, system, "110", test_json, base_dir, phenix_output_root, cache_n_jobs, cache_backend, vis_enable)
    extra_test_cfgs = [
        _make_split_cfg(base_cfg, baseline_name, system, split_label, raw_pairs_json, base_dir, phenix_output_root, cache_n_jobs, cache_backend, vis_enable)
        for split_label, raw_pairs_json in extra_test_json_specs
    ]
    for extra_test_cfg in extra_test_cfgs:
        extra_test_cfg["test_label"] = str(extra_test_cfg["split_label"])

    print("=" * 72)
    print(f"[run_baseline_two_stage] baseline={baseline_name} system={system}")
    print(f"val_output_root: {val_cfg['output_root']}")
    print(f"test_output_root: {test_cfg['output_root']}")
    print(f"cache_n_jobs: {cache_n_jobs}")
    print(f"cache_backend: {cache_backend}")
    print("=" * 72)

    # 只为真正会运行的 split 预生成 baseline 缓存; 已有结果的旧 split 保持只读。
    if _val_cache_required(val_cfg):
        build_baseline_cache_for_split(val_cfg, baseline_name, system)
    else:
        print(f"[run_baseline_two_stage] skip val cache: found Stage1/Stage2 best_params under {val_cfg['output_root']}", flush=True)
    if _fixed_test_cache_required(test_cfg):
        build_baseline_cache_for_split(test_cfg, baseline_name, system)
    else:
        print(f"[run_baseline_two_stage] skip protein_110 cache: found best_summary under {test_cfg['output_root']}", flush=True)
    for extra_test_cfg in extra_test_cfgs:
        if _fixed_test_cache_required(extra_test_cfg):
            build_baseline_cache_for_split(extra_test_cfg, baseline_name, system)
        else:
            print(f"[run_baseline_two_stage] skip {extra_test_cfg.get('test_label')} cache: found best_summary under {extra_test_cfg['output_root']}", flush=True)

    # 复用 DL 三段流程; model=None, 命中预生成 cache 绝不 forward
    return run_two_stage_then_fixed_test(val_cfg, test_cfg, model=None, device=device, extra_test_cfgs=extra_test_cfgs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="枚举 6 baseline × 2 system, 走 40 选阈值 -> 110 固定测试。")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, help="落地根目录(infer_cache/infer_out/infer_vis/infer_error 父目录)。")
    parser.add_argument("--systems", nargs="*", default=SYSTEMS, help="要跑的系统, 默认 stardard strict。")
    parser.add_argument("--baselines", nargs="*", default=ALL_BASELINE_NAMES, help="要跑的 baseline 名, 默认全部 6 组。")
    parser.add_argument("--val_json", default=DEFAULT_VAL_JSON, help="protein_40 样本列表 JSON。")
    parser.add_argument("--test_json", default=DEFAULT_TEST_JSON, help="protein_110 样本列表 JSON。")
    parser.add_argument(
        "--extra_test_json",
        action="append",
        default=[],
        help="追加固定测试 split, 格式 label=/path/to/raw_pairs.json; 可重复传入。",
    )
    parser.add_argument("--phenix_output_root", default=DEFAULT_PHENIX_OUTPUT_ROOT, help="phenix 预生成差图根目录(仅 phenix baseline 消费)。")
    parser.add_argument("--device", default="cuda:0", help="占位设备; baseline 不 forward。")
    parser.add_argument("--cache_n_jobs", type=int, default=1, help="baseline cache 构建并行 worker 数; 默认 1 保持串行。")
    parser.add_argument("--cache_backend", default="loky", help="baseline cache 构建 joblib 后端, 如 loky/threads。")
    parser.add_argument("--vis_enable", type=parse_bool, default=True, help="是否允许样本级可视化与 raw 差图 sidecar 写出; 默认 true。")
    args = parser.parse_args(argv)

    device = torch.device(str(args.device))
    extra_test_json_specs = parse_extra_test_json_specs(args.extra_test_json)
    for system in args.systems:
        # dict[str, Any], 当前系统基础配置
        base_cfg = _load_system_base_cfg(system)
        for baseline_name in args.baselines:
            run_one_baseline(
                base_cfg=base_cfg,
                baseline_name=baseline_name,
                system=system,
                base_dir=str(args.base_dir),
                val_json=str(args.val_json),
                test_json=str(args.test_json),
                extra_test_json_specs=extra_test_json_specs,
                phenix_output_root=str(args.phenix_output_root),
                device=device,
                cache_n_jobs=int(args.cache_n_jobs),
                cache_backend=str(args.cache_backend),
                vis_enable=bool(args.vis_enable),
            )
    print("[run_baseline_two_stage] all baselines done.")


if __name__ == "__main__":
    main()
