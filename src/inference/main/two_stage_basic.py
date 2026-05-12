from __future__ import annotations

"""
Linux 服务器用法示例:
    -     sbatch sbatch/a100/1gpu.sbatch voxel_param_search "raw_pairs_json=/path/pairs.json ckpt_path=/path/model.ckpt output_root=inference_output/two_stage_basic cache_root=inference_output/voxel_cache stage1_objective_expr=avg_voxel_f1 stage2_objective_expr='avg_instance_f1 + 0.5*avg_voxel_f1'"
    -     sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch voxel_param_search "stage1_objective_expr='avg_voxel_f1' stage2_objective_expr='avg_instance_f1 + avg_voxel_f1'"
或者: 
#!/bin/bash
python Pocket_Plus/src/inference/main/two_stage_basic.py \
    --config="voxel_param_search" \
    stage1_objective_expr='avg_voxel_f1' \
    stage2_objective_expr='avg_instance_f1 + avg_voxel_f1'

命令含义:
    - voxel_param_search 是 configs/infer_or_eval/voxel_param_search.yaml 的配置名。
    - 引号内是 Hydra/OmegaConf 覆盖项, 通常至少需要指定 raw_pairs_json、ckpt_path、output_root。
    - stage1_objective_expr 控制第一阶段 threshold-only 搜索目标。
    - stage2_objective_expr 控制第二阶段 threshold/min_component_voxels/connectivity_policy 搜索目标。

工作内容:
    1. 读取基础 voxel_param_search YAML, 合并命令行覆盖项。
    2. 只加载一次 checkpoint 和训练配置。
    3. 第一阶段固定 basic、min_component_voxels=5、connectivity_policy=7_none, 搜索 threshold=0.05..1.00。
    4. 从第一阶段 best_params.json 读取最优 threshold。
    5. 第二阶段固定 basic, 搜索 threshold=[best-0.08,best+0.04]、min_component_voxels=5..10、connectivity_policy=7_none/19_none/27_none。

输出目录:
    - {output_root}/stage1_threshold_only: 第一阶段完整参数搜索产物, 包含 best_params.json、best_summary.json、per_sample_best_metrics.json、搜索历史 Excel 和 best_outputs/。
    - {output_root}/stage2_threshold_component_policy: 第二阶段完整参数搜索产物, 文件结构同第一阶段, 不会覆盖第一阶段。
    - {output_root}/stage1_best_threshold.txt: 第一阶段最优 threshold 的纯文本记录。
    - {output_root}/two_stage_basic_summary.json: 两阶段输出目录、第一阶段最优 threshold、第二阶段搜索空间和两个 objective_expr 的汇总。
"""

import argparse
import copy
import json
import os
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from src.inference.get_pred import load_model, load_training_config
from src.inference.main.voxel_pipeline import run_voxel_param_search

ADVANCED_SEARCH_PARAM_NAMES = [
    "sigma_nearby",
    "kernel_nearby",
    "sigma_response",
    "kernel_response",
    "score_add",
    "score_minus",
    "voxel_score_min",
    "instance_score_min",
]
STAGE1_OBJECTIVE_KEY = "stage1_objective_expr"
STAGE2_OBJECTIVE_KEY = "stage2_objective_expr"
DEFAULT_STAGE1_OBJECTIVE_EXPR = "avg_voxel_f1"
DEFAULT_STAGE2_OBJECTIVE_EXPR = "avg_instance_f1 + 0.5*avg_voxel_f1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage basic voxel parameter search.")
    parser.add_argument("--config", required=True, help="Config name under configs/infer_or_eval or YAML path.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. ckpt_path=... raw_pairs_json=...")
    return parser.parse_args(argv)


def load_base_config(config_name_or_path: str, overrides: list[str]) -> dict[str, Any]:
    config_path = Path(config_name_or_path)
    if not config_path.exists():
        config_name = config_name_or_path[:-5] if config_name_or_path.endswith(".yaml") else config_name_or_path
        config_path = PROJECT_ROOT / "configs" / "infer_or_eval" / f"{config_name}.yaml"

    base_cfg = OmegaConf.load(config_path)
    override_cfg = OmegaConf.from_dotlist(overrides)
    merged_cfg = OmegaConf.merge(base_cfg, override_cfg)
    cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"配置必须解析为 dict, 实际为 {type(cfg_dict)}")
    if str(cfg_dict.get("mode")) != "voxel_param_search":
        raise ValueError(f"two_stage_basic 仅支持 mode=voxel_param_search, 实际为 {cfg_dict.get('mode')}")
    return cfg_dict


def get_required_cfg(cfg_dict: dict[str, Any], key: str) -> Any:
    if key not in cfg_dict or cfg_dict[key] is None:
        raise KeyError(f"缺少必填配置: {key}")
    return cfg_dict[key]


def _stage_output_root(run_root: str, stage_name: str) -> str:
    return str(Path(run_root) / stage_name)


def build_stage1_cfg(base_cfg: dict[str, Any], run_root: str) -> dict[str, Any]:
    stage_cfg = copy.deepcopy(base_cfg)
    stage_cfg["output_root"] = _stage_output_root(run_root, "stage1_threshold_only")
    stage_cfg["filter_strength"] = "basic"
    stage_cfg["threshold"] = 0.0
    stage_cfg["min_component_voxels"] = 5 
    stage_cfg["connectivity_policy"] = "7_none"
    stage_cfg["search_strategy"] = "grid"
    stage_cfg["search_space"] = {
        "threshold": {"type": "float", "min": 0.05, "max": 1.0, "step": 0.01},
    }
    stage_cfg["objective_expr"] = str(stage_cfg.get(STAGE1_OBJECTIVE_KEY, DEFAULT_STAGE1_OBJECTIVE_EXPR))
    stage_cfg["fixed_search_params"] = [
        "min_component_voxels",
        "connectivity_policy",
        *ADVANCED_SEARCH_PARAM_NAMES,
    ]
    return stage_cfg


def build_stage2_cfg(base_cfg: dict[str, Any], run_root: str, best_threshold: float) -> dict[str, Any]:
    if not 0.0 <= float(best_threshold) <= 1.0:
        raise ValueError(f"best_threshold 必须在 [0,1], 实际为 {best_threshold}")

    threshold_min = round(max(0.0, float(best_threshold) - 0.08), 2)
    threshold_max = round(min(1.0, float(best_threshold) + 0.04), 2)
    stage_cfg = copy.deepcopy(base_cfg)
    stage_cfg["output_root"] = _stage_output_root(run_root, "stage2_threshold_component_policy")
    stage_cfg["filter_strength"] = "basic"
    stage_cfg["threshold"] = float(best_threshold)
    stage_cfg["min_component_voxels"] = 5
    stage_cfg["connectivity_policy"] = "7_none"
    stage_cfg["search_strategy"] = "grid"
    stage_cfg["search_space"] = {
        "threshold": {"type": "float", "min": threshold_min, "max": threshold_max, "step": 0.01},
        "min_component_voxels": {"type": "int", "min": 5, "max": 10, "step": 1},
        "connectivity_policy": {"values": ["7_none", "19_none", "27_none"]},
    }
    stage_cfg["objective_expr"] = str(stage_cfg.get(STAGE2_OBJECTIVE_KEY, DEFAULT_STAGE2_OBJECTIVE_EXPR))
    stage_cfg["fixed_search_params"] = list(ADVANCED_SEARCH_PARAM_NAMES)
    return stage_cfg


def read_best_threshold(stage1_output_root: str) -> float:
    best_params_path = Path(stage1_output_root) / "best_params.json"
    with best_params_path.open("r", encoding="utf-8") as f:
        best_params = json.load(f)
    best_threshold = float(best_params["threshold"])
    if not 0.0 <= best_threshold <= 1.0:
        raise ValueError(f"best threshold 必须在 [0,1], 实际为 {best_threshold}")
    return best_threshold


def write_two_stage_summary(
    run_root: str,
    stage1_cfg: dict[str, Any],
    stage2_cfg: dict[str, Any],
    best_threshold: float,
) -> None:
    os.makedirs(run_root, exist_ok=True)
    threshold_path = Path(run_root) / "stage1_best_threshold.txt"
    threshold_path.write_text(f"{best_threshold:.6f}\n", encoding="utf-8")

    summary = {
        "stage1_output_root": stage1_cfg["output_root"],
        "stage2_output_root": stage2_cfg["output_root"],
        "stage1_best_threshold": float(best_threshold),
        "stage1_objective_expr": stage1_cfg["objective_expr"],
        "stage2_objective_expr": stage2_cfg["objective_expr"],
        "stage2_threshold_search_space": stage2_cfg["search_space"]["threshold"],
        "stage2_search_space": stage2_cfg["search_space"],
    }
    summary_path = Path(run_root) / "two_stage_basic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _print_stage_cfg(stage_name: str, stage_cfg: dict[str, Any]) -> None:
    print("=" * 72)
    print(f"[two_stage_basic] {stage_name}")
    print(f"output_root: {stage_cfg['output_root']}")
    print(f"objective_expr: {stage_cfg['objective_expr']}")
    print(f"search_space: {json.dumps(stage_cfg['search_space'], ensure_ascii=False, sort_keys=True)}")
    print(f"fixed_search_params: {stage_cfg['fixed_search_params']}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_cfg = load_base_config(args.config, args.overrides)
    run_root = str(get_required_cfg(base_cfg, "output_root"))
    os.makedirs(run_root, exist_ok=True)

    ckpt_path = str(get_required_cfg(base_cfg, "ckpt_path"))
    device_value = base_cfg.get("device")
    device_str = str(device_value) if device_value is not None else "cuda:0"
    device = torch.device(device_str)

    print("=" * 72)
    print("[two_stage_basic] 双 basic voxel 参数搜索")
    print(f"config: {args.config}")
    print(f"run_root: {run_root}")
    print(f"device: {device_str}")
    print(f"checkpoint: {ckpt_path}")
    print("=" * 72)

    backbone_override = base_cfg.get("backbone_override")
    model = load_model(ckpt_path, device, backbone_override=backbone_override)
    train_cfg = load_training_config(ckpt_path)
    base_cfg["_train_dataset_cfg"] = train_cfg["dataset"]

    stage1_cfg = build_stage1_cfg(base_cfg, run_root)
    _print_stage_cfg("stage1_threshold_only", stage1_cfg)
    run_voxel_param_search(stage1_cfg, model, device)

    best_threshold = read_best_threshold(str(stage1_cfg["output_root"]))
    stage2_cfg = build_stage2_cfg(base_cfg, run_root, best_threshold)
    _print_stage_cfg("stage2_threshold_component_policy", stage2_cfg)
    run_voxel_param_search(stage2_cfg, model, device)

    write_two_stage_summary(run_root, stage1_cfg, stage2_cfg, best_threshold)
    print(f"[two_stage_basic] done. summary: {Path(run_root) / 'two_stage_basic_summary.json'}")


if __name__ == "__main__":
    main()
