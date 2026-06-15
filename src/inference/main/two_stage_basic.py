from __future__ import annotations

"""
固定自动化测试 pipeline 入口: 在验证集(protein_40)上选阈值, 在测试集(protein_110)上固定评估。

支持两种运行情况:
    1. 40 + 110(默认): 用 protein_40 走 Stage1 -> Stage2 选阈值, 再到 protein_110 固定测试。
       即只给 --val_config / --test_config, 不传 --extra_test_json。
    2. 40 + 110 + 40(追加固定测试): 在情况 1 之上, 用 --extra_test_json 追加一个或多个固定测试 split
       (如 nucleic_40), 复用 protein_40 选出的 Stage2 best 参数在追加 split 上再做固定测试。

用法:
    # 情况 1: 40 + 110
    python /home/penghongen/My_Project/Pocket_Plus/src/inference/main/two_stage_basic.py \
        --val_config unet_c1_stardard_40 \
        --test_config unet_c1_stardard \
        [key=value 覆盖项...]

    # 情况 2: 40 + 110 + 40(追加固定测试 split, --extra_test_json 可重复传入)
    python /home/penghongen/My_Project/Pocket_Plus/src/inference/main/two_stage_basic.py \
        --val_config unet_c1_stardard_40 \
        --test_config unet_c1_stardard \
        --extra_test_json nucleic_40=/home/penghongen/My_Project/Pocket_Plus/src/inference/utils/nucleic_40.json \
        [key=value 覆盖项...]

命令含义:
    - --val_config: 验证配置名(或 YAML 路径), 对应 protein_40, 用于扫 threshold。
    - --test_config: 测试配置名(或 YAML 路径), 对应 protein_110, 只做固定评估、不扫参。
    - --extra_test_json: 追加固定测试 split, 格式 label=/path/to/raw_pairs.json, 可重复传入(情况 2 专用; 情况 1 省略即可)。
      其 output_root/cache_root/vis_output_root/error_dir 由 --test_config 的同名字段追加 _{label} 后缀派生, 复用 Stage2 best 参数固定评估。
    - 末尾 key=value 覆盖项同时套用到 val/test(及追加 split)配置(常见: ckpt_path、device、stage1_objective_expr、stage2_objective_expr、pipeline_vis_enable);
      路径/身份字段(output_root、cache_root、raw_pairs_json、vis_output_root、error_dir)请在各自 YAML 配好, 不要统一覆盖。

固定三段流程(均只加载一次 checkpoint, 验证与测试共用同一模型):
    1. protein_40 Stage 1: 固定 basic / mcv=10 / 7_none / merge_dist=0.0 / vis=false, 只扫 threshold=0.00..1.00, objective=avg_voxel_f1, 不做 instance 运算。
    2. protein_40 Stage 2: 固定 basic / mcv=10 / 7_none / merge_dist=5.0 / vis 由 pipeline_vis_enable 控制, 在 Stage1 best 阈值 ±0.10 内只扫 threshold, objective 含 global instance F1。
    3. protein_110 fixed test: search_space={}, 用 Stage2 best 后处理参数固定评估, vis 由 pipeline_vis_enable 控制, 不扫参。
       情况 2 下, 每个 --extra_test_json 追加 split 复用同一 Stage2 best 参数, 各自再做一次固定测试。

输出目录:
    - {val_output_root}/stage1_threshold_only: 第一阶段参数搜索产物。
    - {val_output_root}/stage2_threshold_component_policy: 第二阶段参数搜索产物。
    - {test_output_root}: 第三阶段(protein_110)固定测试产物(直接写测试配置根目录)。
    - {test_output_root}_{label}: 情况 2 下每个追加固定测试 split 的产物(如 {test_output_root}_nucleic_40)。
    - {val_output_root}/stage1_best_threshold.txt: 第一阶段最优 threshold 纯文本。
    - {val_output_root}/two_stage_basic_summary.json: 三段输出目录、Stage1 最优 threshold、Stage2 best 参数与测试侧字段汇总(含 fixed_tests 列出全部固定测试 split)。
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
    "merge_dist",
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
DEFAULT_STAGE2_OBJECTIVE_EXPR = "avg_voxel_f1 + global_instance_f1_cov03 + global_instance_f1_cov06"

# dict[str, float | str], Stage1 threshold 默认搜索空间(DL); baseline 由配置 stage1_search_space 覆盖成高分位网格
DEFAULT_STAGE1_THRESHOLD_SPACE = {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01}
# float, Stage2 局部窗口默认半宽(DL ±0.10); baseline 由配置 stage2_threshold_window_halfwidth 覆盖
DEFAULT_STAGE2_WINDOW_HALFWIDTH = 0.10
# float, Stage2 threshold 默认步长(DL 0.01); baseline 由配置 stage2_threshold_step 覆盖
DEFAULT_STAGE2_THRESHOLD_STEP = 0.01

# list[str], _postprocess_params_from_cfg() 读取的后处理参数名; test 阶段据此从 Stage2 best 复制固定参数
POSTPROCESS_PARAM_NAMES = [
    "threshold",
    "min_component_voxels",
    "filter_strength",
    "connectivity_policy",
    "sigma_nearby",
    "kernel_nearby",
    "sigma_response",
    "kernel_response",
    "score_add",
    "score_minus",
    "voxel_score_min",
    "instance_score_min",
    "merge_dist",
]



def build_stage1_cfg(base_cfg: dict[str, Any], run_root: str) -> dict[str, Any]:
    stage_cfg = copy.deepcopy(base_cfg)
    stage_cfg["output_root"] = _stage_output_root(run_root, "stage1_threshold_only")
    stage_cfg["filter_strength"] = "basic"
    stage_cfg["threshold"] = 0.0
    stage_cfg["min_component_voxels"] = 10
    stage_cfg["connectivity_policy"] = "7_none"
    stage_cfg["merge_dist"] = 0.0
    stage_cfg["search_strategy"] = "grid"
    stage_cfg["vis_enable"] = False
    stage_cfg["compute_instance_metrics"] = False
    # dict[str, float | str], Stage1 threshold 搜索空间; baseline 用配置 stage1_search_space 覆盖成高分位网格, DL 缺省 0..1 step 0.01
    stage1_threshold_space = base_cfg.get("stage1_search_space") or DEFAULT_STAGE1_THRESHOLD_SPACE
    stage_cfg["search_space"] = {"threshold": dict(stage1_threshold_space)}
    stage_cfg["objective_expr"] = str(stage_cfg.get(STAGE1_OBJECTIVE_KEY, DEFAULT_STAGE1_OBJECTIVE_EXPR))
    stage_cfg["fixed_search_params"] = [
        "min_component_voxels",
        "connectivity_policy",
        *ADVANCED_SEARCH_PARAM_NAMES,
    ]
    return stage_cfg

def _pipeline_vis_enabled(base_cfg: dict[str, Any]) -> bool:
    """
    解析 two_stage_basic 的总可视化开关。

    输入参数:
        - base_cfg: dict[str, Any], 基础 voxel_param_search 配置, 可包含 pipeline_vis_enable

    输出:
        - enabled: bool, Stage2 与 fixed test 是否导出可视化资产
    """
    return bool(base_cfg.get("pipeline_vis_enable", True))


def build_stage2_cfg(base_cfg: dict[str, Any], run_root: str, best_threshold: float | dict[str, float]) -> dict[str, Any]:
    """
    构造 two_stage_basic 第二阶段参数搜索配置。

    输入参数:
        - base_cfg: dict[str, Any], 基础 voxel_param_search 配置
        - run_root: str, 两阶段输出根目录
        - best_threshold: float | dict[str, float], 第一阶段最优 threshold; 多分类时为类别名到 threshold 的映射

    输出:
        - stage_cfg: dict[str, Any], 第二阶段 voxel_param_search 配置
    """
    stage_cfg = copy.deepcopy(base_cfg)
    stage_cfg["output_root"] = _stage_output_root(run_root, "stage2_threshold_component_policy")
    stage_cfg["filter_strength"] = "basic"
    stage_cfg["min_component_voxels"] = 10
    stage_cfg["connectivity_policy"] = "7_none"
    stage_cfg["merge_dist"] = 5.0
    stage_cfg["search_strategy"] = "grid"
    stage_cfg["vis_enable"] = _pipeline_vis_enabled(base_cfg)
    stage_cfg["compute_instance_metrics"] = True
    # float, Stage2 局部窗口半宽; baseline 用配置 stage2_threshold_window_halfwidth 覆盖
    window_halfwidth = float(base_cfg.get("stage2_threshold_window_halfwidth") or DEFAULT_STAGE2_WINDOW_HALFWIDTH)
    # float, Stage2 threshold 步长; baseline 用配置 stage2_threshold_step 覆盖
    window_step = float(base_cfg.get("stage2_threshold_step") or DEFAULT_STAGE2_THRESHOLD_STEP)
    if isinstance(best_threshold, dict):
        if len(best_threshold) == 0:
            raise ValueError("best_threshold by_class 映射不能为空")
        # dict[str, dict[str, Any]], class_name -> 当前类局部 threshold 搜索空间
        search_space_by_class = {
            str(class_name): {"threshold": _threshold_window(float(threshold), window_halfwidth, window_step)}
            for class_name, threshold in best_threshold.items()
        }
        # float, 用于保留 stage_cfg.threshold 的全类平均阈值, 实际逐类搜索会使用 search_space_by_class
        stage_cfg["threshold"] = float(sum(float(v) for v in best_threshold.values()) / len(best_threshold))
        stage_cfg["search_space"] = {
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
            # "min_component_voxels": {"type": "int", "min": 5, "max": 10, "step": 1},
            # "connectivity_policy": {"values": ["7_none", "19_none", "27_none"]},
        }
        stage_cfg["search_space_by_class"] = search_space_by_class
    else:
        stage_cfg["threshold"] = float(best_threshold)
        stage_cfg["search_space"] = {
            "threshold": _threshold_window(float(best_threshold), window_halfwidth, window_step),
            # "min_component_voxels": {"type": "int", "min": 5, "max": 10, "step": 1},
            # "connectivity_policy": {"values": ["7_none", "19_none", "27_none"]},
        }
        stage_cfg["search_space_by_class"] = {}
    stage_cfg["objective_expr"] = str(stage_cfg.get(STAGE2_OBJECTIVE_KEY, DEFAULT_STAGE2_OBJECTIVE_EXPR))
    stage_cfg["fixed_search_params"] = list(ADVANCED_SEARCH_PARAM_NAMES)
    return stage_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定三段流程: protein_40 选阈值, protein_110 固定测试。")
    parser.add_argument("--val_config", required=True, help="验证配置名(protein_40)或 YAML 路径; 用于 Stage1/Stage2 扫阈值。")
    parser.add_argument("--test_config", required=True, help="测试配置名(protein_110)或 YAML 路径; 用于固定测试。")
    parser.add_argument(
        "--extra_test_json",
        action="append",
        default=[],
        help="追加固定测试 split, 格式 label=/path/to/raw_pairs.json; 可重复传入, 例如 nucleic_40=.../nucleic_40.json。",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist 覆盖项, 同时套用到两个配置, 例如 ckpt_path=... device=...")
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


def _threshold_step_decimals(step: float) -> int:
    """
    由步长推断窗口边界 round 的小数位数。

    输入参数:
        - step: float, threshold 步长, 如 0.01 / 0.0001

    输出:
        - decimals: int, 小数位数, 至少 2; 用于对窗口上下界做 round 避免浮点毛刺
    """
    # str, 步长字符串的小数部分(如 "0.0001" -> "0001")
    fractional = repr(float(step)).split(".")[-1] if "." in repr(float(step)) else ""
    return max(2, len(fractional))


def _threshold_window(best_threshold: float, halfwidth: float, step: float) -> dict[str, float | str]:
    """
    基于第一阶段最优 threshold 构造第二阶段局部搜索窗口。

    输入参数:
        - best_threshold: float, 第一阶段最优 threshold, 取值范围 [0,1]
        - halfwidth: float, 窗口半宽(DL 默认 0.10, baseline 0.001)
        - step: float, threshold 步长(DL 默认 0.01, baseline 0.0001)

    输出:
        - search_space: dict[str, float | str], threshold 的 grid 搜索空间配置
    """
    if not 0.0 <= float(best_threshold) <= 1.0:
        raise ValueError(f"best_threshold 必须在 [0,1], 实际为 {best_threshold}")
    # int, 窗口边界 round 小数位
    decimals = _threshold_step_decimals(step)
    # float, 第二阶段 threshold 搜索下界
    threshold_min = round(max(0.0, float(best_threshold) - float(halfwidth)), decimals)
    # float, 第二阶段 threshold 搜索上界
    threshold_max = round(min(1.0, float(best_threshold) + float(halfwidth)), decimals)
    return {"type": "float", "min": threshold_min, "max": threshold_max, "step": float(step)}


def read_stage1_best_thresholds(stage1_output_root: str) -> float | dict[str, float]:
    """
    读取第一阶段 best_params.json 中的最优 threshold。

    输入参数:
        - stage1_output_root: str, 第一阶段输出目录

    输出:
        - best_threshold: float | dict[str, float], 二分类为单个 threshold, 多分类为类别名到 threshold 的映射
    """
    best_params_path = Path(stage1_output_root) / "best_params.json"
    with best_params_path.open("r", encoding="utf-8") as f:
        best_params = json.load(f)
    if "by_class" in best_params:
        # dict[str, float], class_name -> 第一阶段最优 threshold
        best_threshold_by_class: dict[str, float] = {}
        for class_name, class_params in best_params["by_class"].items():
            if "threshold" not in class_params:
                raise KeyError(f"类别 {class_name} 的 best_params 缺少 threshold")
            threshold = float(class_params["threshold"])
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"类别 {class_name} 的 best threshold 必须在 [0,1], 实际为 {threshold}")
            best_threshold_by_class[str(class_name)] = threshold
        return best_threshold_by_class
    best_threshold = float(best_params["threshold"])
    if not 0.0 <= best_threshold <= 1.0:
        raise ValueError(f"best threshold 必须在 [0,1], 实际为 {best_threshold}")
    return best_threshold


def read_best_threshold(stage1_output_root: str) -> float:
    """
    读取二分类第一阶段 best_params.json 中的最优 threshold。

    输入参数:
        - stage1_output_root: str, 第一阶段输出目录

    输出:
        - best_threshold: float, 二分类第一阶段最优 threshold
    """
    best_threshold = read_stage1_best_thresholds(stage1_output_root)
    if isinstance(best_threshold, dict):
        raise ValueError("read_best_threshold 只支持二分类平铺 best_params; 多分类请使用 read_stage1_best_thresholds")
    return float(best_threshold)


def read_best_params(stage_output_root: str) -> dict[str, Any]:
    """
    读取某阶段 best_params.json 的完整后处理参数(二分类平铺)。

    输入参数:
        - stage_output_root: str, 阶段输出目录

    输出:
        - best_params: dict[str, Any], 完整后处理参数; 含 best threshold 与各固定后处理项
    """
    best_params_path = Path(stage_output_root) / "best_params.json"
    with best_params_path.open("r", encoding="utf-8") as f:
        best_params = json.load(f)
    if "by_class" in best_params:
        raise ValueError("read_best_params 只支持二分类平铺 best_params; 多分类 test 暂不支持")
    return best_params


def build_test_cfg(test_base_cfg: dict[str, Any], stage2_best_params: dict[str, Any]) -> dict[str, Any]:
    """
    构造 protein_110 固定测试配置: 用 Stage2 best 后处理参数固定评估一次, 不扫参。

    输入参数:
        - test_base_cfg: dict[str, Any], 测试配置(protein_110); output_root 直接作为测试产物根目录
        - stage2_best_params: dict[str, Any], Stage2 选出的完整后处理参数(含 best threshold)

    输出:
        - stage_cfg: dict[str, Any], 测试阶段 voxel_param_search 配置; search_space 为空
    """
    stage_cfg = copy.deepcopy(test_base_cfg)
    # 固定使用 Stage2 best 的全部后处理参数(best threshold + mcv=10 + 7_none + merge_dist=5.0 + basic + advanced 占位)
    for name in POSTPROCESS_PARAM_NAMES:
        if name not in stage2_best_params:
            raise KeyError(f"Stage2 best_params 缺少后处理参数: {name}")
        stage_cfg[name] = stage2_best_params[name]
    stage_cfg["search_strategy"] = "grid"
    stage_cfg["search_space"] = {}
    stage_cfg["search_space_by_class"] = {}
    stage_cfg["fixed_search_params"] = []
    stage_cfg["vis_enable"] = _pipeline_vis_enabled(test_base_cfg)
    stage_cfg["compute_instance_metrics"] = True
    # PR-AUC 阈值无关, 只在 110 fixed test 算一次(search_space={} 单次评估, 无冗余)
    stage_cfg["compute_pr_auc"] = True
    stage_cfg["objective_expr"] = str(stage_cfg.get(STAGE2_OBJECTIVE_KEY, DEFAULT_STAGE2_OBJECTIVE_EXPR))
    return stage_cfg


def write_two_stage_summary(
    run_root: str,
    stage1_cfg: dict[str, Any],
    stage2_cfg: dict[str, Any],
    test_records: list[dict[str, Any]],
    best_threshold: float | dict[str, float],
    stage2_best_params: dict[str, Any],
    write_stage1_threshold: bool,
) -> None:
    """
    写出三段流程摘要(写在验证 run_root)。

    输入参数:
        - run_root: str, 验证侧两阶段输出根目录
        - stage1_cfg: dict[str, Any], 第一阶段参数搜索配置
        - stage2_cfg: dict[str, Any], 第二阶段参数搜索配置
        - test_records: list[dict[str, Any]], 固定测试 split 摘要, 第一项为主 protein_110 测试
        - best_threshold: float | dict[str, float], 第一阶段最优 threshold
        - stage2_best_params: dict[str, Any], 第二阶段最优后处理参数(测试阶段实际使用)
        - write_stage1_threshold: bool, 是否写出 stage1_best_threshold.txt; Stage1 跳过时保持旧文件不动

    输出:
        - None, 写出 stage1_best_threshold.txt 和 two_stage_basic_summary.json
    """
    os.makedirs(run_root, exist_ok=True)
    threshold_path = Path(run_root) / "stage1_best_threshold.txt"
    if write_stage1_threshold:
        if isinstance(best_threshold, dict):
            threshold_path.write_text(json.dumps(best_threshold, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            threshold_path.write_text(f"{best_threshold:.6f}\n", encoding="utf-8")

    primary_test = test_records[0] if test_records else {}
    summary = {
        "stage1_output_root": stage1_cfg["output_root"],
        "stage2_output_root": stage2_cfg["output_root"],
        "test_output_root": primary_test.get("output_root"),
        "test_raw_pairs_json": primary_test.get("raw_pairs_json"),
        "fixed_tests": test_records,
        "stage1_objective_expr": stage1_cfg["objective_expr"],
        "stage2_objective_expr": stage2_cfg["objective_expr"],
        "stage2_search_space": stage2_cfg["search_space"],
        "stage2_search_space_by_class": stage2_cfg.get("search_space_by_class", {}),
        "stage2_best_params": stage2_best_params,
        "test_threshold": stage2_best_params.get("threshold"),
    }
    if isinstance(best_threshold, dict):
        summary["stage1_best_threshold_by_class"] = best_threshold
    else:
        summary["stage1_best_threshold"] = float(best_threshold)
        summary["stage2_threshold_search_space"] = stage2_cfg["search_space"]["threshold"]
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


def _fmt_metric(summary: dict[str, Any], key: str) -> str:
    """
    从 best_summary 取一个指标并格式化; 缺失返回 'NA'。

    输入参数:
        - summary: dict[str, Any], test 阶段 best_summary
        - key: str, 指标键名

    输出:
        - text: str, 6 位小数文本或 'NA'
    """
    # Any, 指标原始值; 缺失或 None 时返回 NA
    value = summary.get(key)
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _print_final_test_metrics(test_label: str, best_summary: dict[str, Any], test_threshold: Any) -> None:
    """
    在三段流程末尾打印某个固定测试 split 的紧凑最终指标(只到 stdout, 不写文件)。

    输入参数:
        - test_label: str, 当前固定测试 split 标签
        - best_summary: dict[str, Any], test 阶段 run_voxel_param_search 返回的 best_summary
        - test_threshold: Any, 实际使用的测试 threshold(来自 Stage2 best params)
    """
    print("=" * 72)
    print(f"[two_stage_basic] {test_label} fixed test 最终指标")
    print(f"test_threshold: {test_threshold}")
    print(f"avg_voxel_f1:   {_fmt_metric(best_summary, 'avg_voxel_f1')}")
    print(f"pr_auc_macro:   {_fmt_metric(best_summary, 'pr_auc_macro')}  (有效样本数 {best_summary.get('pr_auc_num_valid', 'NA')})")
    for tag in ("03", "06"):
        # str, 当前覆盖率阈值下的 global instance F1/P/R 文本
        f1 = _fmt_metric(best_summary, f"global_instance_f1_cov{tag}")
        precision = _fmt_metric(best_summary, f"global_instance_precision_cov{tag}")
        recall = _fmt_metric(best_summary, f"global_instance_recall_cov{tag}")
        print(f"global_instance cov{tag}: F1={f1} P={precision} R={recall}")
        # str, 当前覆盖率阈值下的 loose global instance F1/P/R 文本
        f1_loose = _fmt_metric(best_summary, f"global_instance_f1_loose_cov{tag}")
        precision_loose = _fmt_metric(best_summary, f"global_instance_precision_loose_cov{tag}")
        recall_loose = _fmt_metric(best_summary, f"global_instance_recall_loose_cov{tag}")
        print(f"global_instance loose cov{tag}: F1={f1_loose} P={precision_loose} R={recall_loose}")
    for tag in ("03", "06"):
        # str, 当前覆盖率阈值下 top3/4/5 成功率文本
        topk_text = "  ".join(f"top{k}={_fmt_metric(best_summary, f'top{k}_success_ratio_cov{tag}')}" for k in (3, 4, 5))
        print(f"topK cov{tag}: {topk_text}")
    print("=" * 72)


def _read_json_dict(path: str | Path) -> dict[str, Any]:
    """
    读取 JSON 文件并按 dict 返回。

    输入参数:
        - path: str | Path, JSON 文件路径

    输出:
        - data: dict[str, Any], JSON 顶层对象
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON 顶层对象必须为 dict: {path}")
    return data


def _best_params_path(output_root: str) -> Path:
    return Path(output_root) / "best_params.json"


def _best_summary_path(output_root: str) -> Path:
    return Path(output_root) / "best_summary.json"


def parse_extra_test_json_specs(values: list[str]) -> list[tuple[str, str]]:
    """
    解析追加固定测试 split 的命令行规格。

    输入参数:
        - values: list[str], 每项为 label=/path/to/raw_pairs.json 或 raw_pairs.json

    输出:
        - specs: list[tuple[str, str]], 每项为 (split_label, raw_pairs_json)
    """
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            label, raw_pairs_json = value.split("=", 1)
            split_label = label.strip()
            split_json = raw_pairs_json.strip()
        else:
            split_json = value.strip()
            split_label = Path(split_json).stem
        if not split_label or not split_json:
            raise ValueError(f"--extra_test_json 格式错误: {value}")
        specs.append((split_label, split_json))
    return specs


def _append_path_leaf_suffix(path_value: Any, suffix: str) -> str | None:
    """
    给路径最后一级目录追加 split 后缀。

    输入参数:
        - path_value: Any, 原始路径值; None 保持为 None
        - suffix: str, 追加到最后一级目录名后的后缀

    输出:
        - new_path: str | None, 后缀派生后的路径
    """
    if path_value is None:
        return None
    path = Path(str(path_value))
    return str(path.with_name(f"{path.name}_{suffix}"))


def derive_extra_test_cfgs(test_cfg: dict[str, Any], specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """
    从主测试配置派生追加固定测试配置。

    输入参数:
        - test_cfg: dict[str, Any], 主测试配置, 通常对应 protein_110
        - specs: list[tuple[str, str]], 每项为 (split_label, raw_pairs_json)

    输出:
        - extra_cfgs: list[dict[str, Any]], 每项为独立 output/cache/error 路径的追加测试配置
    """
    extra_cfgs: list[dict[str, Any]] = []
    for split_label, raw_pairs_json in specs:
        cfg = copy.deepcopy(test_cfg)
        cfg["test_label"] = split_label
        cfg["raw_pairs_json"] = raw_pairs_json
        for key in ("output_root", "cache_root", "vis_output_root", "error_dir"):
            if key in cfg:
                cfg[key] = _append_path_leaf_suffix(cfg.get(key), split_label)
        if not _pipeline_vis_enabled(cfg):
            cfg["vis_output_root"] = None
        extra_cfgs.append(cfg)
    return extra_cfgs


def _run_param_search_unless_best_params(stage_name: str, stage_cfg: dict[str, Any], model: Any, device: torch.device) -> bool:
    """
    按 best_params.json 是否存在决定运行或跳过参数搜索。

    输入参数:
        - stage_name: str, 当前阶段名称, 仅用于日志
        - stage_cfg: dict[str, Any], run_voxel_param_search 配置
        - model: Any, DL 模型; baseline 路径可为 None
        - device: torch.device, 推理设备

    输出:
        - skipped_existing: bool, 是否因为已有 best_params.json 而跳过
    """
    best_params_path = _best_params_path(str(stage_cfg["output_root"]))
    if best_params_path.exists():
        print(f"[two_stage_basic] skip {stage_name}: found {best_params_path}", flush=True)
        return True
    run_voxel_param_search(stage_cfg, model, device)
    return False


def _run_or_read_fixed_test(
    test_label: str,
    test_base_cfg: dict[str, Any],
    stage2_best_params: dict[str, Any],
    model: Any,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    执行或读取一个固定测试 split。

    输入参数:
        - test_label: str, 当前固定测试 split 标签
        - test_base_cfg: dict[str, Any], 测试基底配置
        - stage2_best_params: dict[str, Any], Stage2 选出的固定后处理参数
        - model: Any, DL 模型; baseline 路径可为 None
        - device: torch.device, 推理设备

    输出:
        - test_stage_cfg: dict[str, Any], 已注入固定后处理参数的测试配置
        - record: dict[str, Any], 当前 split 的输出索引与是否跳过
    """
    test_stage_cfg = build_test_cfg(test_base_cfg, stage2_best_params)
    _print_stage_cfg(f"{test_label} fixed_test", test_stage_cfg)

    output_root = str(test_stage_cfg["output_root"])
    best_summary_path = _best_summary_path(output_root)
    skipped_existing = best_summary_path.exists()
    if skipped_existing:
        print(f"[two_stage_basic] skip {test_label} fixed_test: found {best_summary_path}", flush=True)
        best_summary = _read_json_dict(best_summary_path)
    else:
        test_result = run_voxel_param_search(test_stage_cfg, model, device)
        best_summary = test_result["best_summary"]

    _print_final_test_metrics(test_label, best_summary, stage2_best_params.get("threshold"))
    return test_stage_cfg, {
        "label": test_label,
        "output_root": output_root,
        "raw_pairs_json": test_stage_cfg.get("raw_pairs_json"),
        "best_summary_path": str(best_summary_path),
        "skipped_existing": skipped_existing,
        "test_threshold": stage2_best_params.get("threshold"),
    }


def run_two_stage_then_fixed_test(
    val_cfg: dict[str, Any],
    test_cfg: dict[str, Any],
    model: Any,
    device: torch.device,
    extra_test_cfgs: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """
    执行固定三段流程: protein_40 Stage1 -> protein_40 Stage2 -> protein_110 fixed test。

    输入参数:
        - val_cfg: dict[str, Any], 验证配置(protein_40); 已注入 _train_dataset_cfg
        - test_cfg: dict[str, Any], 测试配置(protein_110); 已注入 _train_dataset_cfg
        - model: torch.nn.Module | None, stage1 模型; baseline 复用预生成 cache 时可为 None(命中缓存绝不 forward)
        - device: torch.device, 推理设备
        - extra_test_cfgs: list[dict[str, Any]] | None, 追加固定测试配置, 如 nucleic_40

    输出:
        - result: dict[str, str], 三段输出目录与摘要路径
    """
    # str, 验证侧两阶段输出根目录
    val_run_root = str(get_required_cfg(val_cfg, "output_root"))
    os.makedirs(val_run_root, exist_ok=True)

    stage1_cfg = build_stage1_cfg(val_cfg, val_run_root)
    _print_stage_cfg("protein_40 stage1_threshold_only", stage1_cfg)
    stage1_skipped_existing = _run_param_search_unless_best_params("protein_40 stage1_threshold_only", stage1_cfg, model, device)

    best_threshold = read_stage1_best_thresholds(str(stage1_cfg["output_root"]))
    stage2_cfg = build_stage2_cfg(val_cfg, val_run_root, best_threshold)
    _print_stage_cfg("protein_40 stage2_threshold_component_policy", stage2_cfg)
    _run_param_search_unless_best_params("protein_40 stage2_threshold_component_policy", stage2_cfg, model, device)

    # dict[str, Any], Stage2 最优后处理参数(含 best threshold), 供 protein_110 固定测试复用
    stage2_best_params = read_best_params(str(stage2_cfg["output_root"]))
    test_records: list[dict[str, Any]] = []
    primary_test_stage_cfg, primary_test_record = _run_or_read_fixed_test(
        "protein_110",
        test_cfg,
        stage2_best_params,
        model,
        device,
    )
    test_records.append(primary_test_record)
    for extra_test_cfg in extra_test_cfgs or []:
        test_label = str(extra_test_cfg.get("test_label") or Path(str(extra_test_cfg.get("raw_pairs_json", "extra_test"))).stem)
        _, extra_test_record = _run_or_read_fixed_test(
            test_label,
            extra_test_cfg,
            stage2_best_params,
            model,
            device,
        )
        test_records.append(extra_test_record)

    write_two_stage_summary(
        val_run_root,
        stage1_cfg,
        stage2_cfg,
        test_records,
        best_threshold,
        stage2_best_params,
        write_stage1_threshold=not stage1_skipped_existing,
    )
    return {
        "val_run_root": val_run_root,
        "stage1_output_root": str(stage1_cfg["output_root"]),
        "stage2_output_root": str(stage2_cfg["output_root"]),
        "test_output_root": str(primary_test_stage_cfg["output_root"]),
        "summary_path": str(Path(val_run_root) / "two_stage_basic_summary.json"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    val_cfg = load_base_config(args.val_config, args.overrides)
    test_cfg = load_base_config(args.test_config, args.overrides)
    extra_test_cfgs = derive_extra_test_cfgs(test_cfg, parse_extra_test_json_specs(args.extra_test_json))

    ckpt_path = str(get_required_cfg(val_cfg, "ckpt_path"))
    device_value = val_cfg.get("device")
    device_str = str(device_value) if device_value is not None else "cuda:0"
    device = torch.device(device_str)

    print("=" * 72)
    print("[two_stage_basic] 固定三段流程: protein_40 选阈值 -> protein_110 固定测试")
    print(f"val_config: {args.val_config}  raw_pairs_json: {val_cfg.get('raw_pairs_json')}")
    print(f"test_config: {args.test_config}  raw_pairs_json: {test_cfg.get('raw_pairs_json')}")
    for extra_test_cfg in extra_test_cfgs:
        print(f"extra_test: {extra_test_cfg.get('test_label')}  raw_pairs_json: {extra_test_cfg.get('raw_pairs_json')}")
    print(f"val_output_root: {val_cfg.get('output_root')}")
    print(f"test_output_root: {test_cfg.get('output_root')}")
    print(f"device: {device_str}")
    print(f"checkpoint: {ckpt_path}")
    print("=" * 72)

    backbone_override = val_cfg.get("backbone_override")
    model = load_model(ckpt_path, device, backbone_override=backbone_override)
    train_cfg = load_training_config(ckpt_path)
    val_cfg["_train_dataset_cfg"] = train_cfg["dataset"]
    test_cfg["_train_dataset_cfg"] = train_cfg["dataset"]
    for extra_test_cfg in extra_test_cfgs:
        extra_test_cfg["_train_dataset_cfg"] = train_cfg["dataset"]

    result = run_two_stage_then_fixed_test(val_cfg, test_cfg, model, device, extra_test_cfgs=extra_test_cfgs)
    print(f"[two_stage_basic] done. summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
