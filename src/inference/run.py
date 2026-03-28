"""
run.py - 推断与评估的统一入口

通过 Hydra 管理所有配置参数, 串联 parse_input → get_pred → postprocess → evaluator 四个模块。

支持的运行模式 (mode):
    - "single":       单样本推断 + 可选评估 (已处理的 BOX .npz 文件)
    - "batch":        批量推断 + 可选评估 + Excel 汇总
    - "param_search":     使用BOX, 进行网格调参搜索 + Excel 对比

    - "raw_single":       单样本原始文件推断 (直接读取 .cif + .map, 无需预处理)
    - "raw_batch":        批量原始文件推断 + 可选评估 + Excel 汇总
    - "raw_param_search": 使用原始数据, 进行网格调参搜索 + Excel 对比

支持两种评估场景:
    场景A : cif_path = 真实结构，cif_gt_path = None, 模型输入特征和 GT 标签均来自同一 .cif 文件
    场景B (分离): cif_path = 建模结构（AF3/CryoAtom 等），cif_gt_path = 真实实验解析结构. 模型输入特征来自 cif_path，GT 标签来自 cif_gt_path

启动命令示例:
    # 单样本推断 (有 GT 评估)
    python src/inference/run.py --config=infer \\
        +class_folder="small_molecule" +sample_name="9f3f_0_0_0_0_C" \\
        +ckpt_path="feedback/logs/.../last.ckpt"

    # 批量推断 + 评估
    python src/inference/run.py --config=infer mode=batch \\
        +ckpt_path="feedback/logs/.../last.ckpt"

    # 网格调参搜索
    python src/inference/run.py --config=search_best_param \\
        mode=param_search +ckpt_path="..."
"""

import os
import sys
import argparse
import itertools 
import numpy as np
import torch
import openpyxl
from openpyxl.styles import PatternFill, Font
import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path
POCKET_ROOT = Path(__file__).resolve().parent.parent.parent  # Pocket/
if str(POCKET_ROOT) in sys.path:
    sys.path.remove(str(POCKET_ROOT))
sys.path.insert(0, str(POCKET_ROOT))


def _build_vis_bundle_safe(
    cfg_dict: dict,
    output_root: str,   # 没用; （可视化根目录由 vis_output_root 显式指定）
    cif_path: str,
    map_path: str,
    cif_gt_path: str,
    pred_atom_coords: np.ndarray,
    prob_threshold: float,
    filter_preset: str,
    class_mapping: list,
    select_first_model: bool,
    pdb_id: str = None,
) -> dict:
    """
    安全封装可视化生成流程：统一处理开关、输出路径与异常捕获。    
    输入参数:
        - cfg_dict: dict, 当前 Hydra 配置的扁平字典
        - output_root: str, 推断主输出目录（可视化根目录由 vis_output_root 显式指定）
        - cif_path: str, 推断用结构文件路径
        - map_path: str, 对应密度图路径
        - cif_gt_path: str | None, 真实结构路径（可为空）
        - pred_atom_coords: np.ndarray | None, (N_pred, 3), 预测为正类的原子点云（世界坐标）
        - prob_threshold: float | None, 预测阈值(仅用于写文件名)
        - filter_preset: str | None, 配体筛选预设
        - class_mapping: list | None, 标签类别映射
        - select_first_model: bool | None, 是否仅使用第一个 model
        - pdb_id: str | None, 样本 ID（为空则自动从路径推断）
    输出:
        - result: dict | None, build_infer_vis_bundle 的返回汇总；失败或关闭时为 None
    """
    # bool, 是否启用可视化
    vis_enable = cfg_dict.get("vis_enable", True)
    if not vis_enable:
        return None

    # str | None, 自定义可视化输出根目录
    vis_output_root = cfg_dict.get("vis_output_root")
    if not vis_output_root:
        print("[Vis] 未设置 vis_output_root，跳过可视化")
        return None
    # see me: vis_subdir, str | None, 如果存在, 那么输出目录就变为 os.path.join(vis_output_root, vis_subdir) 而不是 vis_output_root
    vis_subdir = cfg_dict.get("vis_subdir", None)
    if vis_subdir:
        vis_output_root = os.path.join(vis_output_root, vis_subdir)

    if not cif_path or not map_path:
        print("[Vis] 缺少 cif_path/map_path，跳过可视化")
        return None
    if filter_preset is None:
        print("[Vis] 未设置 filter_preset，跳过可视化")
        return None

    try:
        _pred_atom_coords = pred_atom_coords
        _class_mapping = class_mapping

        result = build_infer_vis_bundle(
            output_root=vis_output_root,
            cif_path=cif_path,
            map_path=map_path,
            cif_gt_path=cif_gt_path,
            pred_atom_coords=_pred_atom_coords,
            prob_threshold=prob_threshold,
            filter_preset=filter_preset,
            class_mapping=_class_mapping,
            pdb_id=pdb_id,
            select_first_model=select_first_model,
        )
        if result and result.get("root_dir"):
            print(f"[Vis] 已生成: {result['root_dir']}")
        return result
    except Exception as e:
        print(f"[Vis] 生成失败: {e}")
        return None


def _get_config_name() -> str:
    """
    从命令行参数中解析 --config 参数，用于指定 Hydra 配置文件名。
    用法: python src/inference/run.py --config=infer
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Hydra config file name (without .yaml)")
    args, _ = parser.parse_known_args()
    # 清理 sys.argv，防止 hydra.main 报错无法识别 --config
    cleaned = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    return args.config


_CONFIG_NAME = _get_config_name()
# 导入推断模块
from src.inference.parse_input import (
    load_from_raw_cif, load_gt_from_structure,
)
from src.inference.get_pred import load_model, run_inference
from src.inference.postprocess import assign_prob_to_atoms, point_semantic_segment
from src.inference.evaluator import semantic_evaluate, print_metrics
from src.inference.utils.utils import (
    write_batch_excel,
    write_param_search_excel,
    generate_param_grid,
    build_infer_vis_bundle,
)

from src.inference.utils.pipline_for_box import (
    run_single_mode,
    run_batch_mode,
    run_param_search_mode,
)









# =============================================================================
# 二. 对于原始样本
# =============================================================================
# 原始样本推断（直接读 .cif + .map）
def run_raw_pipeline(
    model: torch.nn.Module,
    device: torch.device,
    cif_path: str,
    map_path: str,
    cif_gt_path: str = None,  # 可选，真实结构路径（仅用于 GT 标签提取评估）；None 则回退到 cif_path

    target_voxel_size: float = None,  # 0.7
    compute_density: bool = None,     # True
    select_first_model: bool = None,  # True
    eval_gt: bool = None,             # False
    filter_preset: str = None,        # "five_class"
    class_mapping: list = None,       # half模式下为 [0,1,0,0,1]
    threshold: float = None,          # 0.5, 对点概率截断的阈值
    dist_threshold: float = None,     # 3.0, hit的容忍阈值

    point_assign_radius: float = None,        # 1.5
    point_assign_sigma: float = None,         # 1.0（ #NOTE？）
    point_cat_weight_home: float = None,      # 0.5
    point_cat_weight_has_atom: float = None,  # 0.35
    point_cat_weight_no_atom: float = None,   # 0.15

    stride: int = None,               # 36
    windows_size: int = None,         # 72
    batch_size: int = None,           # 3
    core_offset: int = None,          # 12
    output_dir: str = None,
    show_progress: bool = True,
    error_dir: str = None,
) -> dict:
    """
    单样本原始文件推断流水线: load_from_raw_cif → get_pred → postprocess → (可选)evaluator。
    与 run_single_pipeline() 逻辑完全对齐，不同之处在于数据来源是原始 .cif + .map。

    支持两种评估场景:
      场景A : cif_path = 真实结构，cif_gt_path = None, 模型输入特征和 GT 标签均来自同一 .cif 文件
      场景B (分离): cif_path = 建模结构（AF3/CryoAtom 等），cif_gt_path = 真实实验解析结构. 模型输入特征来自 cif_path，GT 标签来自 cif_gt_path

    Args:
        - model:               nn.Module,    已加载的 backbone
        - device:              torch.device, 推断设备
        - cif_path:            str,          原始 .cif / .pdb 文件路径（用于提取模型输入特征）
        - map_path:            str,          对应的 EMDB 密度图路径 (.map / .mrc)
        - cif_gt_path:         str | None,   可选，真实结构文件路径（仅用于决定合格的配体）. 若为 None，则 GT 提取时回退使用 cif_path（即场景A）。

        - target_voxel_size:   float,        重采样目标体素大小 (Å)
        - compute_density:     bool,         是否计算原子局部密度特征
        - select_first_model:  bool,         多模型 CIF 时是否仅取第一个
        - eval_gt:             bool,         是否提取 GT 标签进行评估
        - filter_preset:       str,          配体筛选预设名 (用于 GT 提取)
        - class_mapping:       list[int]|None, 标签类别映射表
        - threshold:           float,        语义分割阈值
        - dist_threshold:      float,        点云评估(hit)距离阈值 (Å)

        
        - point_assign_radius: float,        为原子赋概率时的搜索半径 (Å)
        - point_assign_sigma:  float,        为原子赋概率时的高斯衰减核sigma (Å)
        - point_cat_weight_home:     float,  所在体素类别权重
        - point_cat_weight_has_atom: float,  含原子体素类别权重
        - point_cat_weight_no_atom:  float,  不含原子体素类别权重
        

        - stride:              int,          滑窗步幅
        - windows_size:        int,          滑窗边长
        - batch_size:          int,          推断 batch size
        - core_offset:         int,          丢弃每个 block 边缘的体素数
        - output_dir:          str|None,     输出目录(保存 pred_prob.npz, pred_atom_coords.npz, 可选的 gt_points.npz); None 则不保存
        - show_progress:       bool,         是否显示进度条
        - error_dir:           str|None,     错误日志目录

    Returns:
        - result: dict, 与 run_single_pipeline() 返回格式相同，包含:
            - "sample_name":   str
            - "class_folder":  str, 强制指定为 "raw"
            - "pred_prob":     np.ndarray, (D,H,W), 概率图
            - "pred_atom_coords": np.ndarray, (N_pred,3), 预测为正类的原子坐标
            - "atom_coords":   np.ndarray, (N_atom,3), 所有原子坐标
            - "hardmask":      np.ndarray, (D,H,W), 硬掩膜
            - "origin":        np.ndarray, (3,), origin
            - "voxel_size":    np.ndarray, (3,), voxel_size
            - "metrics":       dict | None, 评估指标（eval_gt=True 且结构中有配体时）
            - "error":         str | None, 错误信息
    """
    from pathlib import Path as _Path
    sample_name = _Path(cif_path).stem
    result = {
        "sample_name":  sample_name,
        "class_folder": "raw",
        "metrics":      None,
        "error":        None,
    }

    try:
        # ---- 1. 从原始文件提取特征 ----
        data = load_from_raw_cif(
            cif_path=cif_path,
            map_path=map_path,
            target_voxel_size=target_voxel_size,
            compute_density=compute_density,
            select_first_model=select_first_model,
            error_dir=error_dir,
        )
        # np.ndarray, float32, (C, D, H, W)
        grid     = data["grid"]
        # np.ndarray, int64, (D, H, W), 1=有原子 0=无原子
        hardmask = data["hardmask"]
        origin   = data["origin"]       # np.ndarray, 形状 (3,), 世界坐标原点
        voxel_sz = data["voxel_size"]   # np.ndarray, 形状 (3,), 体素大小
        atom_coords = data["atom_coords"] # np.ndarray, 形状 (N_atom, 3), 原子坐标


        # ---- 2. 可选: 提取 GT 标签 ----
        gt_data = None
        if eval_gt:
            # str, 场景A(无 cif_gt_path)时回退到 cif_path、场景B(有 cif_gt_path)时使用真实结构
            _effective_gt_cif = cif_gt_path if cif_gt_path else cif_path
            gt_data = load_gt_from_structure(
                cif_path=cif_path,              # 受体结构（场景B时为预测结构）
                cif_gt_path=_effective_gt_cif,  # 真实结构（含配体信息）；场景A时与 cif_path 相同
                map_path=map_path,
                target_voxel_size=target_voxel_size,
                filter_preset=filter_preset,
                class_mapping=class_mapping,
                select_first_model=select_first_model,
                error_dir=error_dir,
            ) 


        # ---- 3. 滑窗推断 ----
        # np.ndarray, float32, (D, H, W)
        pred_prob = run_inference(
            model=model, device=device, show_progress=show_progress, grid=grid,
            stride=stride,
            windows_size=windows_size,
            batch_size=batch_size,
            core_offset=core_offset,
        )


        # ---- 4. 后处理 ----
        # 4a. 体素概率 → 原子概率
        atom_probs = assign_prob_to_atoms(
            pred_prob=pred_prob,
            atom_coords=atom_coords,
            origin=origin,
            voxel_size=voxel_sz,
            hardmask=hardmask,
            radius=point_assign_radius,
            sigma=point_assign_sigma,
            cat_weight_home=point_cat_weight_home,
            cat_weight_has_atom=point_cat_weight_has_atom,
            cat_weight_no_atom=point_cat_weight_no_atom,
        )

        # 4b. 原子概率 → 语义分割 (二值化)
        pred_atom_coords = point_semantic_segment(
            atom_probs=atom_probs,
            atom_coords=atom_coords,
            threshold=threshold,
        )
        
        result["pred_prob"]  = pred_prob
        result["pred_atom_coords"] = pred_atom_coords
        result["atom_coords"] = atom_coords
        result["hardmask"]   = hardmask
        result["origin"]     = origin
        result["voxel_size"] = voxel_sz


        # ---- 5. 评估 (可选) ----
        if gt_data is not None:
            metrics = semantic_evaluate(
                pred_atom_coords=pred_atom_coords,
                atom_gt=gt_data["atom_gt"],
                dist_threshold=dist_threshold,
            )
            result["metrics"] = metrics


        # ---- 6. 保存结果(可视化逻辑内置于下面的函数) ----
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(output_dir, "pred_prob.npz"),
                pred_prob=pred_prob,
            )
            np.savez_compressed(
                os.path.join(output_dir, "pred_atom_coords.npz"),
                pred_atom_coords=pred_atom_coords,
            )
            # 保存 GT 点云（若有）
            if gt_data is not None:
                np.savez_compressed(
                    os.path.join(output_dir, "gt_points.npz"),
                    atom_gt=gt_data["atom_gt"],
                    atom_coords=gt_data["atom_coords"],
                )

    except Exception as e:
        result["error"] = str(e)
        import traceback
        print(f"  ❌ 推断失败: {e}")
        traceback.print_exc()
    return result
















# ----------------------------------------------- 主入口 -----------------------------------------------
@hydra.main(version_base="1.3", config_path="../../configs/infer_or_eval", config_name=_CONFIG_NAME)
def main(cfg: DictConfig):
    """
    推断与评估的统一主入口。
    按 cfg.mode 分流执行:
        - single       → _run_single_mode()       (已处理 BOX .npz)
        - batch        → _run_batch_mode()        (已处理 BOX .npz 批量)
        - param_search → _run_param_search_mode() (超参数搜索)
        - raw          → _run_raw_mode()          (原始 .cif + .map 单样本)
        - raw_batch    → _run_raw_batch_mode()    (原始 .cif + .map 批量)
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ckpt_path = cfg_dict.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "[错误] 未指定 ckpt_path，请通过命令行传入: "
            "+ckpt_path=\"feedback/logs/.../checkpoints/last.ckpt\""
        )

    # ---- 读取公共配置 ----
    device_str = cfg_dict.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    mode       = cfg_dict["mode"]
    output_root = cfg_dict["output_root"]
    base_infer_params = {
        "stride":       cfg_dict["stride"],
        "windows_size": cfg_dict["windows_size"],
        "batch_size":   cfg_dict["batch_size"],
        "threshold":    cfg_dict["threshold"],
        "core_offset":  cfg_dict["core_offset"],
    }
    os.makedirs(output_root, exist_ok=True)
    print("=" * 60)
    print(f"  [inference/run] 推断与评估")
    print(f"  模式: {mode}  |  配置: {_CONFIG_NAME}.yaml")
    print(f"  设备: {device_str}  |  checkpoint: {ckpt_path}")
    print("=" * 60)
    # ---- 加载模型 (一次) ----
    backbone_override = cfg_dict.get("backbone_override", None)
    model = load_model(ckpt_path, device, backbone_override=backbone_override)



    # ---- 按 mode 执行 ----
    if mode == "single":
        all_data_path      = cfg_dict["all_data_path"]
        data_folder_names  = cfg_dict["data_folder_names"]
        class_mapping      = cfg_dict.get("class_mapping")
        run_single_mode(cfg_dict, model, device, all_data_path,
                         data_folder_names, class_mapping, base_infer_params,
                         output_root)
    elif mode == "batch":
        all_data_path      = cfg_dict["all_data_path"]
        data_folder_names  = cfg_dict["data_folder_names"]
        class_folder_names = cfg_dict["class_folder_names"]
        class_mapping      = cfg_dict.get("class_mapping")
        run_batch_mode(cfg_dict, model, device, all_data_path,
                        data_folder_names, class_folder_names, class_mapping,
                        base_infer_params, output_root)
    elif mode == "param_search":
        all_data_path      = cfg_dict["all_data_path"]
        data_folder_names  = cfg_dict["data_folder_names"]
        class_folder_names = cfg_dict["class_folder_names"]
        class_mapping      = cfg_dict.get("class_mapping")
        run_param_search_mode(cfg_dict, model, device, all_data_path,
                               data_folder_names, class_folder_names, class_mapping,
                               base_infer_params, output_root)
    elif mode == "raw_single":
        _run_raw_mode(cfg_dict, model, device, base_infer_params, output_root)
    elif mode == "raw_batch":
        _run_raw_batch_mode(cfg_dict, model, device, base_infer_params, output_root)
    elif mode == "raw_param_search":
        _run_raw_param_search_mode(cfg_dict, model, device, base_infer_params, output_root)
    else:
        raise ValueError(
            f"[错误] 未知 mode: '{mode}'，"
            "请在配置中设置 mode = single / batch / param_search / raw_single / raw_batch / raw_param_search"
        )






# -----------------------------------------------原始样本 -----------------------------------------------
def _run_raw_mode(cfg_dict, model, device, infer_params, output_root):
    """mode=raw_single: 单样本原始文件推断 + 可选 GT 评估"""
    cif_path = cfg_dict.get("cif_path")
    map_path = cfg_dict.get("map_path")
    if not cif_path or not map_path:
        raise ValueError(
            "[错误] mode=raw_single 需要指定 cif_path 和 map_path。\n"
            "用法: +cif_path=\"/path/to/xxx.cif\" +map_path=\"/path/to/emd_xxx.map\""
        )
    # str | None, 标量, 可选的真实结构路径（仅用于 GT 评估）；None 则回退到 cif_path
    cif_gt_path = cfg_dict.get("cif_gt_path", None)
    from pathlib import Path as _Path
    sample_name = _Path(cif_path).stem

    # dict, 原子赋概率/后处理默认参数（与 raw_* 配置保持一致）
    postprocess_kwargs = {
        "point_assign_radius":        cfg_dict.get("point_assign_radius"),
        "point_assign_sigma":         cfg_dict.get("point_assign_sigma"),
        "point_cat_weight_home":      cfg_dict.get("point_cat_weight_home"),
        "point_cat_weight_has_atom":  cfg_dict.get("point_cat_weight_has_atom"),
        "point_cat_weight_no_atom":   cfg_dict.get("point_cat_weight_no_atom"),
    }

    result = run_raw_pipeline(
        model=model,
        device=device,
        cif_path=cif_path,
        map_path=map_path,
        cif_gt_path=cif_gt_path,
        target_voxel_size=cfg_dict.get("target_voxel_size"),
        compute_density=cfg_dict.get("compute_density"),
        select_first_model=cfg_dict.get("select_first_model"),
        eval_gt=cfg_dict.get("eval_gt"),
        filter_preset=cfg_dict.get("filter_preset"),
        class_mapping=cfg_dict.get("class_mapping"),
        dist_threshold=cfg_dict.get("dist_threshold"),

        # 为了避免 batch 下打印一万次 GT 解析报错，可以给个 error_dir=os.path.join(output_root, sample_name),
        error_dir=cfg_dict.get("error_dir"),
        output_dir=os.path.join(output_root, sample_name),
        **postprocess_kwargs,
        **infer_params,
    )

    if result["error"]:
        print(f"\n❌ 推断失败: {result['error']}")
        return

    pred_prob = result["pred_prob"]
    print(f"\n[Raw] 样本: {sample_name}")
    print(f"[Raw] 概率图统计:")
    print(f"  shape = {pred_prob.shape}")
    print(f"  min={pred_prob.min():.6f}  max={pred_prob.max():.6f}  "
          f"mean={pred_prob.mean():.6f}")
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"  {q*100:.0f}% 分位数 = {np.quantile(pred_prob, q):.6f}")
    _pred_atom_coords = result.get("pred_atom_coords")
    print(f"  预测正类原子数 = {len(_pred_atom_coords) if _pred_atom_coords is not None else 0}")

    if result["metrics"]:
        print(f"\n[Raw] 评估结果 (与 GT 对比):")
        print_metrics(result["metrics"])
    else:
        print(f"\n[Raw] 未启用 GT 评估 (eval_gt=false)")
        
    # np.ndarray | None, (N_pred, 3)
    _pred_atom_coords = result.get("pred_atom_coords")
    _build_vis_bundle_safe(
        cfg_dict=cfg_dict,
        output_root=output_root,
        cif_path=cif_path,
        map_path=map_path,
        cif_gt_path=cif_gt_path,
        pred_atom_coords=_pred_atom_coords,
        prob_threshold=cfg_dict.get("threshold"),
        filter_preset=cfg_dict.get("filter_preset"),
        class_mapping=cfg_dict.get("class_mapping"),
        select_first_model=cfg_dict.get("select_first_model"),
        pdb_id=sample_name,
    )
    print(f"\n✅ 原始文件推断完毕！输出: {os.path.join(output_root, sample_name)}")

def _run_raw_batch_mode(cfg_dict, model, device, infer_params, output_root):
    """
    mode=raw_batch: 批量原始文件推断 + 可选 GT 评估。

    输入方式:
      raw_pairs: JSON 文件路径，每项 dict 含 {"cif_path": ..., "map_path": ...,   None|"cif_gt_path": ...}
      透过 yield_json_from_raw_sample.py 预先生成该 JSON，支持筛选和场景B (cif_gt_path)。
    """
    # ---- 读取 raw_pairs JSON ----
    import time
    # list[tuple(str, str, str|None)], 路径三元组 (cif_path, map_path, cif_gt_path)
    pairs = []
    raw_pairs = cfg_dict.get("raw_pairs", None)
    if not raw_pairs:
        raise ValueError(
            "[错误] mode=raw_batch 需要设置 raw_pairs（JSON 文件路径），"
            "格式为 [{\"cif_path\": ..., \"map_path\": ...[, \"cif_gt_path\": ...]}]\n"
            "可用 src/inference/utils/yield_json_from_raw_sample.py 预先生成该 JSON。"
        )
    if isinstance(raw_pairs, str) and raw_pairs.endswith(".json"):
        import json
        with open(raw_pairs, "r", encoding="utf-8") as f:
            raw_pairs = json.load(f)
    for p in raw_pairs:
        pairs.append((p["cif_path"], p["map_path"], p.get("cif_gt_path", None)))

    if not pairs:
        raise RuntimeError("[错误] mode=raw_batch 未找到任何有效的 (cif, map) 文件对")
    print(f"[raw_batch] 共找到 {len(pairs)} 个样本")


    # ---- 公共参数 ----
    # 从配置中提取原始文件推断的共享参数
    raw_kwargs = {
        "target_voxel_size":  cfg_dict.get("target_voxel_size"),           # 0.7
        "compute_density":    cfg_dict.get("compute_density"),             # true
        "select_first_model": cfg_dict.get("select_first_model"),          # true
        "eval_gt":            cfg_dict.get("eval_gt"),                     # true
        "filter_preset":      cfg_dict.get("filter_preset"),               # "five_class"
        "class_mapping":      cfg_dict.get("class_mapping"),               # [0, 1, 0, 0, 1]
        "point_assign_radius":        cfg_dict.get("point_assign_radius"),
        "point_assign_sigma":         cfg_dict.get("point_assign_sigma"),
        "point_cat_weight_home":      cfg_dict.get("point_cat_weight_home"),
        "point_cat_weight_has_atom":  cfg_dict.get("point_cat_weight_has_atom"),
        "point_cat_weight_no_atom":   cfg_dict.get("point_cat_weight_no_atom"),
        "dist_threshold":     cfg_dict.get("dist_threshold"),              # 3.0
        "error_dir":          cfg_dict.get("error_dir", "error_raw_batch"),
        **infer_params,
    }

    # ---- 批量推断 ----
    all_results = []
    for i, (cif_path, map_path, cif_gt_path) in enumerate(pairs):
        from pathlib import Path as _Path
        sample_name = _Path(cif_path).stem
        print(f"\n[raw_batch {i+1}/{len(pairs)}] === {sample_name} ===")
        if cif_gt_path:
            print(f"  [GT] 使用独立 GT 结构: {cif_gt_path}")
        sample_dir = os.path.join(output_root, sample_name)
        
        start_time = time.time()
        result = run_raw_pipeline(
            model=model, device=device,
            cif_path=cif_path, map_path=map_path,
            cif_gt_path=cif_gt_path,
            output_dir=sample_dir,
            show_progress=False,
            **raw_kwargs,
        )

        if result["error"] is None:
            # np.ndarray | None, (N_pred, 3)
            _pred_atom_coords = result.get("pred_atom_coords")
            result["num_pred_pos"] = len(_pred_atom_coords) if _pred_atom_coords is not None else 0
            
            elapsed = time.time() - start_time
            if result["metrics"] is not None:
                m = result["metrics"]
                print(f"  ✅ [耗时 {elapsed:.1f}s] P={m['precision']:.4f}  R={m['recall']:.4f}  "
                      f"F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
                      f"正类原子={result['num_pred_pos']}")
            else:
                print(f"  ✅ [耗时 {elapsed:.1f}s] 推断完成, 预测正类原子数={result['num_pred_pos']}")
                
            # np.ndarray | None, (N_pred, 3)
            _pred_atom_coords = result.get("pred_atom_coords")
            _build_vis_bundle_safe(
                cfg_dict=cfg_dict,
                output_root=output_root,
                cif_path=cif_path,
                map_path=map_path,
                cif_gt_path=cif_gt_path,
                pred_atom_coords=_pred_atom_coords,
                prob_threshold=cfg_dict.get("threshold"),
                filter_preset=cfg_dict.get("filter_preset"),
                class_mapping=cfg_dict.get("class_mapping"),
                select_first_model=cfg_dict.get("select_first_model"),
                pdb_id=sample_name,
            )
        else:
            result["num_pred_pos"] = 0

        # 释放大数组节省内存
        for key in ["pred_prob", "pred_atom_coords", "atom_coords", "hardmask"]:
            result.pop(key, None)

        all_results.append(result)
    write_batch_excel(all_results, output_root)
    print("\n 批量原始文件推断完毕！\n")


def _run_raw_param_search_mode(cfg_dict, model, device, base_params, output_root):
    """
    mode=raw_param_search: 使用原始 .cif + .map 文件进行网格调参搜索。

    与 mode=param_search 逻辑对齐，区别在于数据来自原始文件而非预处理 BOX:
    - 数据加载: load_from_raw_cif() + load_gt_from_structure()   (而非 load_from_npz_dirs)
    - 单样本推断: run_raw_pipeline()                             (而非 run_single_pipeline)
    - 优化: 若仅搜索 threshold，则复用概率图缓存，只变更阈值

    输入方式:
      raw_pairs: JSON 文件路径，每项 dict 含 {"cif_path": ..., "map_path": ...,  None|"cif_gt_path": ...}
      透过 yield_json_from_raw_sample.py 预先生成该 JSON，支持筛选和场景B (cif_gt_path)。
    """
    # ---- 0. 校验 param_sweep 配置 ----
    param_sweep_cfg = cfg_dict.get("param_sweep", None)
    if not param_sweep_cfg:
        raise ValueError(
            "[错误] mode=raw_param_search 需要配置 param_sweep 节，例如:\n"
            "param_sweep:\n"
            "  - {name: threshold, min: 0.3, max: 0.7, step: 0.05}"
        )

    # ---- 1. 读取 raw_pairs JSON ----
    # list[tuple(str, str, str|None)], 路径三元组 (cif_path, map_path, cif_gt_path)
    pairs = []
    raw_pairs = cfg_dict.get("raw_pairs", None)
    if not raw_pairs:
        raise ValueError(
            "[错误] mode=raw_param_search 需要设置 raw_pairs（JSON 文件路径），"
            "格式为 [{\"cif_path\": ..., \"map_path\": ...[, \"cif_gt_path\": ...]}]\n"
            "可用 src/inference/utils/yield_json_from_raw_sample.py 预先生成该 JSON。"
        )
    if isinstance(raw_pairs, str) and raw_pairs.endswith(".json"):
        import json
        with open(raw_pairs, "r", encoding="utf-8") as f:
            raw_pairs = json.load(f)
    for p in raw_pairs:
        pairs.append((p["cif_path"], p["map_path"], p.get("cif_gt_path", None)))

    if not pairs:
        raise RuntimeError("[错误] mode=raw_param_search 未找到任何有效的 (cif, map) 文件对")
    print(f"[raw_param_search] 共找到 {len(pairs)} 个样本")


    # ---- 2. 生成参数网格 ----
    # list[dict], 每项为一组参数组合
    param_grid  = generate_param_grid(param_sweep_cfg)
    # list[str], 被 sweep 的参数名
    param_names = list(param_grid[0].keys()) if param_grid else []
    # list[dict], 汇总的评估结果
    summary = []

    # 公共调用参数（不含 threshold/stride 等被 sweep 的参数）
    # dict, 原始文件推断的固定参数
    raw_fixed_kwargs = {
        "target_voxel_size":  cfg_dict.get("target_voxel_size"),            # 0.7
        "compute_density":    cfg_dict.get("compute_density"),              # True
        "select_first_model": cfg_dict.get("select_first_model"),           # False
        "eval_gt":            True,   # 调参必须有 GT
        "filter_preset":      cfg_dict.get("filter_preset"),              # "five_class"
        "class_mapping":      cfg_dict.get("class_mapping"),              # [0,1,0,0,1]
        "dist_threshold":     cfg_dict.get("dist_threshold"),             # 3.0
        "point_assign_radius":        cfg_dict.get("point_assign_radius"),
        "point_assign_sigma":         cfg_dict.get("point_assign_sigma"),
        "point_cat_weight_home":      cfg_dict.get("point_cat_weight_home"),
        "point_cat_weight_has_atom":  cfg_dict.get("point_cat_weight_has_atom"),
        "point_cat_weight_no_atom":   cfg_dict.get("point_cat_weight_no_atom"),
        "error_dir":          cfg_dict.get("error_dir"),                  # None
    }
    # dict, 后处理默认参数（仅后处理搜索时作为兜底）
    postprocess_defaults = {
        "point_assign_radius":        cfg_dict.get("point_assign_radius"),
        "point_assign_sigma":         cfg_dict.get("point_assign_sigma"),
        "point_cat_weight_home":      cfg_dict.get("point_cat_weight_home"),
        "point_cat_weight_has_atom":  cfg_dict.get("point_cat_weight_has_atom"),
        "point_cat_weight_no_atom":   cfg_dict.get("point_cat_weight_no_atom"),
    }


    # ---- 3. 判断是否复用概率图缓存加速----
    # 后处理参数名集合（不影响模型 forward 的参数）
    POST_INFERENCE_PARAMS = {
        "threshold", 
        "point_assign_radius", 
        "point_assign_sigma",
        "point_cat_weight_home", 
        "point_cat_weight_has_atom",
        "point_cat_weight_no_atom"
    }
    only_sweep_post_params = all(name in POST_INFERENCE_PARAMS for name in param_names)
    
    if only_sweep_post_params:
        print("\n[raw_param_search] 💡 仅搜索后处理参数，启用推断缓存加速！")
        npz_root_path = cfg_dict.get("npz_root_path", os.path.join(output_root, "cache"))
        os.makedirs(npz_root_path, exist_ok=True)
        # bool, 跑完是否删除缓存文件
        delete_cache = cfg_dict.get("delete_cache", False)

        # ---- 阶段 1: 对每个样本做一次全图推断，缓存概率图 + GT ----
        # list[dict], 每项包含样本的基础信息与 .npz 保存路径 (不再在内存中保存大体积张量)
        cached_info = []
        print("[raw_param_search] --- 阶段 1: 全量推断落盘缓存 ---")
        for i, (cif_path, map_path, cif_gt_path) in enumerate(pairs):
            from pathlib import Path as _Path
            import re
            # str, 从 map_path 提取数字作为 emdb 编号, e.g. "40631"
            emdb_match = re.search(r'\d+', _Path(map_path).name)
            emdb_id = emdb_match.group() if emdb_match else "unknown"
            # str, 从 cif_path 提取无后缀受体文件名并转为小写作为 pdb 编号, e.g. "9qad"
            pdb_id = _Path(cif_path).stem.lower()
            
            cache_name = f"{emdb_id}_{pdb_id}.npz"
            npz_path = os.path.join(npz_root_path, cache_name)
            sname = _Path(cif_path).stem
            # dict, 保存路径信息以便下游读取
            sample_info = {
                "name": sname,
                "cif_path": cif_path,
                "map_path": map_path,
                "cif_gt_path": cif_gt_path,
                "npz_path": npz_path,
                "error": None
            }
            
            print(f"  [{i+1}/{len(pairs)}] {sname}")
            if cif_gt_path:
                print(f"    [GT] 使用独立 GT 结构: {cif_gt_path}")
            
            # 检查是否已有对应的 .npz 缓存文件存在
            if os.path.exists(npz_path):
                print(f"    [Cache] 命中已有硬盘缓存，跳过推断: {npz_path}")
                cached_info.append(sample_info)
                continue
                
            try:
                data = load_from_raw_cif(
                    cif_path=cif_path,
                    map_path=map_path,
                    target_voxel_size=raw_fixed_kwargs["target_voxel_size"],
                    compute_density=raw_fixed_kwargs["compute_density"],
                    select_first_model=raw_fixed_kwargs["select_first_model"],
                )
                # np.ndarray, float32, (D, H, W)
                pred_prob = run_inference(
                    model=model, device=device, show_progress=False, grid=data["grid"],
                    stride=base_params["stride"],
                    windows_size=base_params["windows_size"],
                    batch_size=base_params["batch_size"],
                    core_offset=base_params["core_offset"],
                )
                # str, 场景A(无 cif_gt_path)时回退到 cif_path、场景B(有 cif_gt_path)时使用真实结构
                _effective_gt_cif = cif_gt_path if cif_gt_path else cif_path
                # dict, 点云 GT 数据
                gt_data = load_gt_from_structure(
                    cif_path=cif_path,              # 受体结构（场景B: 预测结构；场景A: 与 gt 相同）
                    cif_gt_path=_effective_gt_cif,  # 真实结构（含配体信息）；场景A时与 cif_path 相同
                    map_path=map_path,
                    target_voxel_size=raw_fixed_kwargs["target_voxel_size"],
                    filter_preset=raw_fixed_kwargs["filter_preset"],
                    class_mapping=raw_fixed_kwargs["class_mapping"],
                    select_first_model=raw_fixed_kwargs["select_first_model"],
                    error_dir=raw_fixed_kwargs["error_dir"],
                )
                
                # dict, 要存入 .npz 的结构
                save_dict = {
                    "pred_prob":   pred_prob,
                    "atom_coords": data["atom_coords"],
                    "hardmask":    data["hardmask"],
                    "origin":      data["origin"],
                    "voxel_size":  data["voxel_size"],
                }
                if gt_data is not None:
                    save_dict["gt_atom_coords"] = gt_data["atom_coords"]
                    save_dict["gt_atom_gt"]     = gt_data["atom_gt"]
                    
                # 存入硬盘 (未压缩)
                np.savez(npz_path, **save_dict)
                print(f"    [Cache] 推断完成，已落盘至: {npz_path}")
                cached_info.append(sample_info)
                
            except Exception as e:
                import traceback
                print(f"  ❌ 推断失败 {sname}: {e}")
                traceback.print_exc()
                # 发生错误时不保存 .npz 缓存文件
                sample_info["error"] = str(e)
                cached_info.append(sample_info)




        # ---- 阶段 2: 遍历参数组合，复用缓存概率图 ----         # TODO: 注意看 Pocket\sbatch\infer\search_v0_half1.sbatch, 一个a100可以配套16核cpu, 那么能否用joblib让cpu"并行地"处理样本，搜索参数？
        print("\n[raw_param_search] --- 阶段 2: 后处理评估 ---")
        
        # 为了记录逐样本结果，判断并生成 sample indices
        excel_sample_n = cfg_dict.get("excel_sample_n", 0)
        selected_sample_indices = []
        if excel_sample_n > 0 and len(cached_info) > 0:
            np.random.seed(42)  # 固定种子保证可重复性
            selected_sample_indices = np.random.choice(len(cached_info), min(excel_sample_n, len(cached_info)), replace=False).tolist()

        from src.inference.postprocess import assign_prob_to_atoms, point_semantic_segment
            
        for idx, override_params in enumerate(param_grid):
            # 获取本组完整的后处理参数
            iter_params = {**postprocess_defaults, **base_params, **override_params}
            th = iter_params.get("threshold")
            r = iter_params.get("point_assign_radius")
            sig = iter_params.get("point_assign_sigma")
            wp1 = iter_params.get("point_cat_weight_home")
            wp2 = iter_params.get("point_cat_weight_has_atom")
            wp3 = iter_params.get("point_cat_weight_no_atom")
            
            param_str = "  ".join(f"{k}={v}" for k, v in override_params.items())
            print(f"[{idx+1}/{len(param_grid)}] {param_str}")
            
            # list[dict], 本轮所有样本的评估结果
            eval_results = []
            
            # 用于逐样本记录的数据
            per_sample_records = []
            
            for c_i, info in enumerate(cached_info):

                if info["error"] is not None: continue
                if not os.path.exists(info["npz_path"]): continue
                with np.load(info["npz_path"], allow_pickle=True) as cache:
                    if "gt_atom_gt" not in cache:
                        continue
                        
                    # 4a. 体素概率 → 原子概率
                    atom_probs = assign_prob_to_atoms(
                        pred_prob=cache["pred_prob"],
                        atom_coords=cache["atom_coords"],
                        origin=cache["origin"],
                        voxel_size=cache["voxel_size"],
                        hardmask=cache["hardmask"],
                        radius=r,
                        sigma=sig,
                        cat_weight_home=wp1,
                        cat_weight_has_atom=wp2,
                        cat_weight_no_atom=wp3,
                    )
                    
                    # 4b. 原子概率 → 二值化
                    pred_atom_coords = point_semantic_segment(
                        atom_probs=atom_probs,
                        atom_coords=cache["atom_coords"],
                        threshold=th,
                    )

                    metrics = semantic_evaluate(
                        pred_atom_coords=pred_atom_coords,
                        atom_gt=cache["gt_atom_gt"],
                        dist_threshold=raw_fixed_kwargs["dist_threshold"],
                    )
                eval_results.append(metrics)
                
                if c_i in selected_sample_indices:
                    per_sample_records.append({
                        "sample_name": info["name"],
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1": metrics.get("f1", 0.0),
                        "iou": metrics.get("iou", 0.0),
                    })

            if eval_results:
                avg_p   = float(np.mean([r["precision"] for r in eval_results]))
                avg_r   = float(np.mean([r["recall"]    for r in eval_results]))
                avg_f1  = float(np.mean([r["f1"]        for r in eval_results]))
                avg_iou = float(np.mean([r["iou"]       for r in eval_results]))
            else:
                avg_p = avg_r = avg_f1 = avg_iou = 0.0

            # dict, 本组参数的评估汇总
            row = {**override_params, "avg_P": avg_p, "avg_R": avg_r,
                   "avg_F1": avg_f1, "avg_IoU": avg_iou}
            if per_sample_records:
                row["_per_sample"] = per_sample_records
                
            summary.append(row)
            print(f"  → avg_P={avg_p:.4f}  avg_R={avg_r:.4f}  "
                  f"avg_F1={avg_f1:.4f}  avg_IoU={avg_iou:.4f}")


    else:   # 非纯后处理搜索 → 每组参数都跑全量推断
        excel_sample_n = cfg_dict.get("excel_sample_n", 0)
        selected_sample_indices = []   # 生成"excel_sample_n"个样本的详细信息
        if excel_sample_n > 0 and len(pairs) > 0:
            np.random.seed(42)
            selected_sample_indices = np.random.choice(len(pairs), min(excel_sample_n, len(pairs)), replace=False).tolist()

        import time
        for idx, override_params in enumerate(param_grid):
            # dict, 本轮使用的完整推断参数（base + override）
            iter_infer_params = {**base_params, **override_params}
            param_str = "  ".join(f"{k}={v}" for k, v in override_params.items())
            print(f"\n{'=' * 60}")
            print(f"[raw_param_search {idx+1}/{len(param_grid)}] {param_str}")
            
            eval_results = []   # list[dict], 本轮各样本的评估结果
            per_sample_records = []
            
            for i, (cif_path, map_path, cif_gt_path) in enumerate(pairs):
                from pathlib import Path as _Path
                sname = _Path(cif_path).stem
                
                # 记录本样本的耗时
                start_time = time.time()
                result = run_raw_pipeline(
                    model=model, device=device,
                    cif_path=cif_path, map_path=map_path,
                    cif_gt_path=cif_gt_path,
                    output_dir=None,          # 调参时不保存到磁盘
                    show_progress=False,
                    **raw_fixed_kwargs,
                    **iter_infer_params,
                )
                elapsed = time.time() - start_time
                
                if result["error"] is None and result["metrics"] is not None:
                    m = result["metrics"]
                    eval_results.append(m)
                    
                    if i in selected_sample_indices:
                        per_sample_records.append({
                            "sample_name": sname,
                            "precision": m.get("precision", 0.0),
                            "recall": m.get("recall", 0.0),
                            "f1": m.get("f1", 0.0),
                            "iou": m.get("iou", 0.0),
                        })
                    print(f"  [{i+1}/{len(pairs)}] {sname} [耗时 {elapsed:.1f}s] F1={m.get('f1', 0.0):.4f}")
                else:
                    if result["error"]:
                        print(f"  [{i+1}/{len(pairs)}] {sname} [耗时 {elapsed:.1f}s] ❌ {result['error']}")
                    else:
                        print(f"  [{i+1}/{len(pairs)}] {sname} [耗时 {elapsed:.1f}s] ✅ 无评估结果")

            if eval_results:
                avg_p   = float(np.mean([r["precision"] for r in eval_results]))
                avg_r   = float(np.mean([r["recall"]    for r in eval_results]))
                avg_f1  = float(np.mean([r["f1"]        for r in eval_results]))
                avg_iou = float(np.mean([r["iou"]       for r in eval_results]))
            else:
                avg_p = avg_r = avg_f1 = avg_iou = 0.0

            row = {**override_params, "avg_P": avg_p, "avg_R": avg_r,
                   "avg_F1": avg_f1, "avg_IoU": avg_iou}
            if per_sample_records:
                row["_per_sample"] = per_sample_records
                
            summary.append(row)
            print(f"  → avg_P={avg_p:.4f}  avg_R={avg_r:.4f}  "
                  f"avg_F1={avg_f1:.4f}  avg_IoU={avg_iou:.4f}")

    # ---- 4. 排序 + 写 Excel + 打印最优 ----
    summary.sort(key=lambda r: r["avg_F1"], reverse=True)
    best = summary[0]
    write_param_search_excel(summary, param_names, output_root, output_name=cfg_dict.get("output_name"))
    print("\n" + "=" * 60)
    print("  [raw_param_search] 🏆 最优参数组合:")
    for k, v in best.items():
        print(f"    {k}: {round(v, 4) if isinstance(v, float) else v}")
    print("=" * 60)



    # ---- 5. 可视化 ----
    vis_enable = cfg_dict.get("vis_enable")
    vis_after_search = cfg_dict.get("vis_after_search")
    if vis_enable and vis_after_search:
        print("\n[raw_param_search] 开始可视化")
        if only_sweep_post_params:
            best_threshold = best.get("threshold", cfg_dict.get("threshold"))
            best_radius = best.get("point_assign_radius", cfg_dict.get("point_assign_radius"))
            best_sigma = best.get("point_assign_sigma", cfg_dict.get("point_assign_sigma"))
            best_wp1 = best.get("point_cat_weight_home", cfg_dict.get("point_cat_weight_home"))
            best_wp2 = best.get("point_cat_weight_has_atom", cfg_dict.get("point_cat_weight_has_atom"))
            best_wp3 = best.get("point_cat_weight_no_atom", cfg_dict.get("point_cat_weight_no_atom"))
            for info in cached_info:
                if info["error"] is not None:
                    continue
                if not os.path.exists(info["npz_path"]):
                    continue
                # 从硬盘重新读取推断的部分结果
                with np.load(info["npz_path"], allow_pickle=True) as cache:
                    # 使用最优后处理参数从缓存的 pred_prob 生成点云
                    atom_probs = assign_prob_to_atoms(
                        pred_prob=cache.get("pred_prob"),
                        atom_coords=cache.get("atom_coords"),
                        origin=cache.get("origin"),
                        voxel_size=cache.get("voxel_size"),
                        hardmask=cache.get("hardmask"),
                        radius=best_radius,
                        sigma=best_sigma,
                        cat_weight_home=best_wp1,
                        cat_weight_has_atom=best_wp2,
                        cat_weight_no_atom=best_wp3,
                    )
                    pred_atom_coords = point_semantic_segment(
                        atom_probs=atom_probs,
                        atom_coords=cache.get("atom_coords"),
                        threshold=best_threshold,
                    )
                _build_vis_bundle_safe(
                    cfg_dict=cfg_dict,
                    output_root=output_root,
                    cif_path=info.get("cif_path"),
                    map_path=info.get("map_path"),
                    cif_gt_path=info.get("cif_gt_path"),
                    pred_atom_coords=pred_atom_coords,
                    prob_threshold=best_threshold,
                    filter_preset=raw_fixed_kwargs.get("filter_preset"),
                    class_mapping=raw_fixed_kwargs.get("class_mapping"),
                    select_first_model=raw_fixed_kwargs.get("select_first_model"),
                    pdb_id=info.get("name"),
                )
        else:
            _param_names = list(param_names)
            best_override = {k: best[k] for k in _param_names}
            iter_infer_params = {**base_params, **best_override}
            best_threshold = iter_infer_params.get("threshold", cfg_dict.get("threshold"))
            for i, (cif_path, map_path, cif_gt_path) in enumerate(pairs):
                from pathlib import Path as _Path
                sname = _Path(cif_path).stem
                print(f"[raw_param_search][正在可视化 {i+1}/{len(pairs)}] {sname}")
                result = run_raw_pipeline(
                    model=model, device=device,
                    cif_path=cif_path, map_path=map_path,
                    cif_gt_path=cif_gt_path,
                    output_dir=None,
                    show_progress=False,
                    **raw_fixed_kwargs,
                    **iter_infer_params,
                )
                if result["error"] is not None:
                    print(f"  [Vis] 错误: {result['error']}")
                    continue
                _pred_atom_coords = result.get("pred_atom_coords")
                _build_vis_bundle_safe(
                    cfg_dict=cfg_dict,
                    output_root=output_root,
                    cif_path=cif_path,
                    map_path=map_path,
                    cif_gt_path=cif_gt_path,
                    pred_atom_coords=_pred_atom_coords,
                    prob_threshold=best_threshold,
                    filter_preset=raw_fixed_kwargs.get("filter_preset"),
                    class_mapping=raw_fixed_kwargs.get("class_mapping"),
                    select_first_model=raw_fixed_kwargs.get("select_first_model"),
                    pdb_id=sname,
                )
                _release_keys = ["pred_prob", "pred_atom_coords", "hardmask"]
                for key in _release_keys:
                    result.pop(key, None)

    # ---- 6. 清理临时缓存文件 ----
    if only_sweep_post_params and delete_cache:
        print(f"\n[raw_param_search] 🧹 清除临时硬盘缓存 ({npz_root_path})...")
        for info in cached_info:
            if info["error"] is None and os.path.exists(info["npz_path"]):
                try:
                    os.remove(info["npz_path"])
                except Exception as e:
                    print(f"  ❌ 删除缓存失败 {info['npz_path']}: {e}")
                    
    print("\n🎉 原始数据网格调参搜索完毕！\n")


if __name__ == "__main__":
    main()
 