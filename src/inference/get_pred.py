"""
get_pred.py — 模型加载与点云级推断模块

负责:
    1. 从 Lightning checkpoint 加载完整 VolumePointStage1Model
    2. 对已组装好的 batch dict 执行模型前向, 返回 atom-level 原始 logits

旧版体素推断逻辑已迁移至 src/inference/legacy/get_pred_voxel.py
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



# =============================================================================
# 1. 模型加载
# =============================================================================
def _extract_backbone_cfg(config_obj: Any) -> dict[str, Any] | None:
    """
    从用户传入配置、checkpoint 超参数或训练配置快照中提取 VolumePointStage1Model 的 Hydra 配置。

    输入参数:
        - config_obj: Any, 可能是 DictConfig、dict、wrapper 级配置或 backbone 级配置

    输出:
        - backbone_cfg: dict[str, Any] | None, 可直接传给 hydra.instantiate() 的 backbone 配置
    """
    if config_obj is None:
        return None

    if OmegaConf.is_config(config_obj):
        config_obj = OmegaConf.to_container(config_obj, resolve=True)
    if not isinstance(config_obj, dict):
        return None

    # 直接就是 backbone 配置
    target_name = config_obj.get("_target_")
    if target_name is not None and "model" not in config_obj and "backbone" not in config_obj:
        return config_obj

    # 训练快照通常是全局配置，backbone 位于 cfg.model.backbone
    model_cfg = config_obj.get("model")
    if isinstance(model_cfg, dict):
        model_backbone_cfg = _extract_backbone_cfg(model_cfg)
        if model_backbone_cfg is not None:
            return model_backbone_cfg

    # wrapper 级配置通常是 cfg.model，backbone 位于其下一级
    nested_backbone_cfg = config_obj.get("backbone")
    if isinstance(nested_backbone_cfg, dict):
        direct_backbone_cfg = _extract_backbone_cfg(nested_backbone_cfg)
        if direct_backbone_cfg is not None:
            return direct_backbone_cfg

    return None

def _load_backbone_cfg_from_training_snapshot(ckpt_path: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    从 checkpoint 同一 run 目录下保存的训练配置快照中恢复 backbone 配置。

    输入参数:
        - ckpt_path: str, Lightning checkpoint 路径

    输出:
        - backbone_cfg: dict[str, Any] | None, 恢复出的 backbone 配置
        - config_path: str | None, 实际命中的配置文件路径
    """
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) < 2:
        return None, None

    run_dir = ckpt_file.parents[1]
    candidate_paths = (
        run_dir / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
    )
    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        loaded_cfg = OmegaConf.load(config_path)
        backbone_cfg = _extract_backbone_cfg(loaded_cfg)
        if backbone_cfg is not None:
            return backbone_cfg, str(config_path)

    return None, None

# 总函数
def load_model(
    ckpt_path: str,
    device: str | torch.device,
    backbone_override: dict | None = None,
) -> torch.nn.Module:
    """
    从 Lightning checkpoint 加载完整 VolumePointStage1Model。

    输入参数:
        - ckpt_path: str, checkpoint 文件路径 (.ckpt)
        - device: str | torch.device, 推断设备
        - backbone_override: dict | None, 若提供则用此配置覆盖 checkpoint 中的 backbone 配置; 格式为 Hydra instantiate 所需的字典（含 _target_ 等）

    输出:
        - model: nn.Module, eval 模式的完整 VolumePointStage1Model, 位于指定 device
    """
    print(f"[get_pred] 正在加载模型: {ckpt_path}")

    # dict, Lightning checkpoint 完整内容
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise RuntimeError(
            f"[get_pred] checkpoint 中未找到 'state_dict'。"
            f"请确认文件是 Lightning .ckpt 格式: {ckpt_path}"
        )

    # ---- 0. 注入训练时的代码快照 (如果存在) ----
    # 当 checkpoint 所在 run 目录下存在 src_snapshot 时，模型类必须从快照中加载，以保证 state_dict key 与模型结构一致。
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) >= 2:
        run_dir = ckpt_file.parents[1]
        snapshot_dir = run_dir / "src_snapshot"
        snapshot_src_dir = snapshot_dir / "src"
        if snapshot_src_dir.exists():
            # str, snapshot 中的 src/ 目录绝对路径
            snapshot_src_str = str(snapshot_src_dir.resolve())

            # list[str], 需要从 sys.modules 中移除的 src.model.* 缓存条目
            stale_module_keys = [
                mod_name for mod_name in sys.modules
                if mod_name == "src.model" or mod_name.startswith("src.model.")
            ]
            for mod_name in stale_module_keys:
                del sys.modules[mod_name]
            # 将 snapshot/src 插入 src 包的 __path__ 最前面, 使后续 import src.model.* 优先从 snapshot 解析
            import src as _src_pkg
            if snapshot_src_str not in _src_pkg.__path__:
                _src_pkg.__path__.insert(0, snapshot_src_str)

            print(f"[get_pred] 已注入代码快照: {snapshot_dir}")
            if stale_module_keys:
                print(f"[get_pred] 已清除 {len(stale_module_keys)} 个 src.model.* 缓存模块")

    # ---- 1. 构建模型 ----
    if backbone_override is not None:
        backbone_cfg = _extract_backbone_cfg(backbone_override)
        if backbone_cfg is None:
            raise ValueError(
                "[get_pred] backbone_override 必须是 VolumePointStage1Model 配置，"
                "或包含 model.backbone / backbone 的完整训练配置。"
            )
        from hydra.utils import instantiate
        model = instantiate(backbone_cfg)
        print("[get_pred] 使用用户提供的 backbone 配置")
    elif "hyper_parameters" in ckpt and "backbone" in ckpt["hyper_parameters"]:
        hp = ckpt["hyper_parameters"]
        backbone_cfg = hp["backbone"]
        from hydra.utils import instantiate
        model = instantiate(backbone_cfg)
        target_name = backbone_cfg.get("_target_", "unknown") if isinstance(backbone_cfg, dict) else "unknown"
        print(f"[get_pred] 从 checkpoint hyper_parameters 自动推断 backbone: {target_name}")
    else:
        backbone_cfg, config_path = _load_backbone_cfg_from_training_snapshot(ckpt_path)
        if backbone_cfg is not None:
            from hydra.utils import instantiate
            model = instantiate(backbone_cfg)
            target_name = backbone_cfg.get("_target_", "unknown")
            print(f"[get_pred] 从训练配置快照恢复 backbone: {target_name}")
            print(f"[get_pred] 配置来源: {config_path}")
        else:
            raise RuntimeError(
                "[get_pred] checkpoint 中无 hyper_parameters.backbone，且未在 checkpoint 邻近目录找到可用训练配置。"
                "请提供 backbone_override，或确认 run_dir/config.yaml 存在且包含 model.backbone。"
            )


    # ---- 3. 提取并加载 backbone 权重 ----
    # dict[str, torch.Tensor], 仅剥掉开头 "backbone." 前缀后的权重字典
    state_dict = ckpt["state_dict"]
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = state_dict
        print("[get_pred] ⚠️ 未找到 backbone.* 前缀, 尝试直接加载全部权重")

    # strict=True: 任何 key 不匹配立即报错, 避免静默加载随机权重
    model.load_state_dict(backbone_state, strict=True)

    model.to(device)
    model.eval()
    print(f"[get_pred] 模型加载成功, device={device}")
    return model

"""
=============================================================================
1.5 训练配置加载
=============================================================================
默认从训练.yaml加载的参数(可覆盖):

A. 训练契约类 —— 应自动从训练 config 的 dataset.* 继承
   ┌────────────────────┬──────────────────────────┐
   │ 参数               │ 训练 config 路径          │
   ├────────────────────┼──────────────────────────┤
   │ data_folder_names  │ dataset.data_folder_names │
   │ class_mapping      │ dataset.class_mapping     │
   │ atom_buffer_radius │ dataset.atom_buffer_radius│
   │ valid_crop_margin  │ dataset.valid_crop_margin │
   │ density_channel_config │ dataset.density_channel_config │
   │ (未来新增参数)      │ dataset.*                 │
   └────────────────────┴──────────────────────────┘
   → 由 load_training_config() 自动提取, 推理管线直接使用。
   → 如需覆盖, 可在推理 YAML 中显式设置同名字段 (_get_cfg 优先级最高)。
"""

def load_training_config(ckpt_path: str) -> dict[str, Any]:
    """
    从 checkpoint 所在的训练 run 目录中提取完整训练配置。

    输入参数:
        - ckpt_path: str, Lightning checkpoint 路径

    输出:
        - train_cfg: dict[str, Any], 训练时的完整配置(顶层 dict, 含 model/dataset/train 等子键)

    异常:
        - FileNotFoundError: 若无法从 checkpoint 路径推断 run 目录, 或 run 目录下无 config.yaml
    """
    ckpt_file = Path(ckpt_path).resolve()
    if len(ckpt_file.parents) < 2:
        raise FileNotFoundError(
            f"无法从 checkpoint 路径推断 run 目录: {ckpt_path}"
        )
    run_dir = ckpt_file.parents[1]
    for config_path in (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml"):
        if config_path.exists():
            loaded = OmegaConf.load(config_path)
            cfg_dict = OmegaConf.to_container(loaded, resolve=True)
            print(f"[get_pred] 已加载训练配置: {config_path}")
            return cfg_dict
    raise FileNotFoundError(
        f"在 run 目录 {run_dir} 下未找到 config.yaml 或 .hydra/config.yaml"
    )









# =============================================================================
# 正文： 推断整体(全图/全蛋白)结果
# =============================================================================


# ---------------------------------------- 工具函数 ----------------------------------------
def move_batch_to_device(
    batch_dict: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """
    将 batch dict 中的所有 torch.Tensor 移至指定设备, 非 tensor 字段保持不变。

    输入参数:
        - batch_dict: dict[str, Any], box_point_collate 输出的 batch dict
        - device: str | torch.device, 目标设备

    输出:
        - batch_dict: dict[str, Any], tensor 已移至 device 的 batch dict (原地修改并返回)
    """
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.to(device)
    return batch_dict

_VOXEL_HEAD_TO_OUTPUT_KEY: dict[str, str] = {
    "ligand": "voxel_logits_ligand",
    "receptor": "voxel_logits_aux",
}

def extract_voxel_head_logits(
    outputs: dict[str, Any],
    output_heads: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """
    从 stage1 模型输出中提取请求的 voxel logits。

    输入参数:
        - outputs: dict[str, Any], run_inference() 返回的模型输出
        - output_heads: tuple[str, ...], 需要提取的 head 名称, 允许 ligand/receptor

    输出:
        - head_logits: dict[str, torch.Tensor], 按请求 head 名称组织的 logits 字典, 包含:
            - "ligand": torch.Tensor, (B, 1, D, H, W), float32, ligand voxel head 的原始 logits; 仅当请求 ligand 时存在
            - "receptor": torch.Tensor, (B, 1, D, H, W), float32, receptor voxel head 的原始 logits; 仅当请求 receptor 时存在
    """
    if len(output_heads) == 0:
        raise ValueError("output_heads 不能为空")

    head_logits: dict[str, torch.Tensor] = {}
    for head_name in output_heads:
        if head_name not in _VOXEL_HEAD_TO_OUTPUT_KEY:
            raise ValueError(f"未知 voxel output head: {head_name}")
        output_key = _VOXEL_HEAD_TO_OUTPUT_KEY[head_name]
        logits = outputs[output_key]
        if logits is None:
            raise ValueError(f"模型输出 {output_key} 为 None, 但 output_heads 请求了 {head_name}")
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"模型输出 {output_key} 必须是 torch.Tensor, 实际为 {type(logits)}")
        if logits.ndim != 5 or int(logits.shape[1]) != 1:
            raise ValueError(f"模型输出 {output_key} 形状必须为 (B, 1, D, H, W), 实际为 {tuple(logits.shape)}")
        head_logits[head_name] = logits
    return head_logits

def run_inference(
    model: torch.nn.Module,
    device: str | torch.device,
    batch_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    对一个已组装好的 batch dict 执行模型前向, 返回原始 atom logits。

    输入参数:
        - model: nn.Module, eval 模式的 VolumePointStage1Model
        - device: str | torch.device, 推断设备
        - batch_dict: dict[str, Any], 由 box_point_collate() 产出的 batch dict, 所有 tensor 已在 device 上

    输出:
        - outputs: dict[str, Any], 模型前向输出字典, 至少包含:
            - "atom_logits": torch.Tensor, (sumN_final, C_logit), 原始 atom 分类 logits, 未经 sigmoid
            - "voxel_logits_ligand": torch.Tensor | None, (B, 1, D, H, W), ligand voxel head 的原始 logits; 未启用时可为 None
            - "voxel_logits_aux": torch.Tensor | None, (B, 1, D, H, W), receptor voxel head 的原始 logits; 未启用时可为 None
            - 以及与展平原子序列对齐的辅助字段, 如 "atom_counts"、"atom_coord_local_voxel"、"atom_is_in_core_box" 等
    """
    with torch.no_grad():
        # dict[str, Any], 模型前向输出
        outputs = model(batch_dict)

    return outputs






# ---------------------------------------- 合并BOX ---------------------------------------- 
def _compute_box_slices_for_merge(
    box_position_zyx: tuple[int, int, int],
    box_shape_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
    core_offset: int,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    """
    计算通过 core_offset 裁剪后的 BOX 切片。

    输入参数:
        - box_position_zyx: tuple[int, int, int], BOX 在整图中的起点(z,y,x)
        - box_shape_zyx: tuple[int, int, int], BOX 概率图形状(D_box,H_box,W_box)
        - full_shape_zyx: tuple[int, int, int], 整图形状(D,H,W)
        - core_offset: int, 非边界侧裁掉的体素厚度; 0 表示不裁剪

    输出:
        - box_slices: tuple[slice, slice, slice], BOX 内有效局部切片
        - full_slices: tuple[slice, slice, slice], 整图目标切片
    """
    box_slices: list[slice] = []
    full_slices: list[slice] = []
    for axis, (start, box_size, full_size) in enumerate(zip(box_position_zyx, box_shape_zyx, full_shape_zyx)):
        if start < 0:
            raise ValueError(f"BOX 起点不能为负数: axis={axis}, start={start}")
        if box_size <= 0 or full_size <= 0:
            raise ValueError(f"BOX/full shape 必须为正数: box_size={box_size}, full_size={full_size}")

        # int, 当前轴 BOX 与整图重叠的局部终点
        local_end = min(int(box_size), int(full_size) - int(start))
        # int, 当前轴 BOX 内参与合并的局部起点
        local_start = int(core_offset) if start > 0 else 0
        if start + box_size < full_size:
            local_end = min(local_end, int(box_size) - int(core_offset))
        if local_start >= local_end:
            raise ValueError(
                "core_offset 导致 BOX 无有效合并区域: "
                f"axis={axis}, position={box_position_zyx}, box_shape={box_shape_zyx}, "
                f"full_shape={full_shape_zyx}, core_offset={core_offset}"
            )

        full_start = int(start) + local_start
        full_end = int(start) + local_end
        box_slices.append(slice(local_start, local_end))
        full_slices.append(slice(full_start, full_end))

    return tuple(box_slices), tuple(full_slices)

def _build_gaussian_weight(
    box_shape_zyx: tuple[int, int, int],
    gaussian_sigma_ratio: float,
) -> np.ndarray:
    """
    构造 BOX 内中心高、边缘低的 3D Gaussian 权重。

    输入参数:
        - box_shape_zyx: tuple[int, int, int], BOX 概率图形状(D,H,W)
        - gaussian_sigma_ratio: float, sigma 与最大边长的比例

    输出:
        - weight: np.ndarray, (D,H,W), float32, Gaussian 合并权重
    """
    if gaussian_sigma_ratio <= 0:
        raise ValueError(f"gaussian_sigma_ratio 必须大于 0, 实际为 {gaussian_sigma_ratio}")
    depth, height, width = [int(v) for v in box_shape_zyx]
    # np.ndarray, (D,H,W), float32, 三轴体素坐标网格
    zz, yy, xx = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    center_z = 0.5 * float(depth - 1)
    center_y = 0.5 * float(height - 1)
    center_x = 0.5 * float(width - 1)
    sigma = float(gaussian_sigma_ratio) * float(max(depth, height, width))
    # np.ndarray, (D,H,W), float32, 到 BOX 中心的平方距离
    dist2 = (zz - center_z) ** 2 + (yy - center_y) ** 2 + (xx - center_x) ** 2
    return np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)

def _merge_box_probability_into_full(
    value_sum: np.ndarray,
    weight_sum: np.ndarray,
    box_prob: np.ndarray,
    box_position_zyx: tuple[int, int, int],
    full_shape_zyx: tuple[int, int, int],
    merge_mode: str,
    core_offset: int,
    gaussian_sigma_ratio: float | None,
) -> None:
    """
    将单个 BOX 概率图按 mean/max 语义原地合并到整图累加数组。

    输入参数:
        - value_sum: np.ndarray, (D,H,W), float32, mean 时为概率加权和, max 时为当前最大概率
        - weight_sum: np.ndarray, (D,H,W), float32, mean 时为权重累计图, max 时为覆盖权重图
        - box_prob: np.ndarray, (D_box,H_box,W_box), float32, 单 BOX 概率图
        - box_position_zyx: tuple[int,int,int], BOX 在整图中的起点(z,y,x)
        - full_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - merge_mode: str, 整图聚合方式, 可选 mean/max
        - core_offset: int, 非边界侧裁剪厚度; 0 表示不裁剪
        - gaussian_sigma_ratio: float | None, Gaussian 衰减 sigma 比例; None 表示不启用衰减

    输出:
        - None, 原地更新 value_sum 和 weight_sum
    """
    # tuple[int,int,int], 当前 BOX 概率图形状(D_box,H_box,W_box)
    box_shape_zyx = tuple(int(v) for v in box_prob.shape)
    box_slices, full_slices = _compute_box_slices_for_merge(
        box_position_zyx=box_position_zyx,
        box_shape_zyx=box_shape_zyx,
        full_shape_zyx=full_shape_zyx,
        core_offset=core_offset,
    )
    if gaussian_sigma_ratio is None:
        # np.ndarray, 与 box_prob[box_slices] 相同形状, 不启用 Gaussian 时等权
        weight = np.ones_like(box_prob[box_slices], dtype=np.float32)
    else:
        # np.ndarray, 与 box_prob[box_slices] 相同形状, BOX 中心高、边缘低的空间权重
        weight = _build_gaussian_weight(box_shape_zyx, gaussian_sigma_ratio)[box_slices]

    # np.ndarray, 当前 BOX 参与合并的概率子块
    prob_patch = box_prob[box_slices].astype(np.float32, copy=False)
    if merge_mode == "max":
        # np.ndarray, 与 prob_patch 相同形状, Gaussian 衰减后的候选最大概率
        weighted_prob_patch = prob_patch * weight
        value_sum[full_slices] = np.maximum(value_sum[full_slices], weighted_prob_patch)
        weight_sum[full_slices] = np.maximum(weight_sum[full_slices], weight)
        return

    if merge_mode != "mean":
        raise ValueError(f"未知 voxel merge_mode: {merge_mode}")

    value_sum[full_slices] += prob_patch * weight
    weight_sum[full_slices] += weight

def get_voxel_pred(
    model: torch.nn.Module,
    device: str | torch.device,
    box_dicts: list[dict[str, Any]],
    full_shape_zyx: tuple[int, int, int],
    hardmask: np.ndarray,
    batch_size: int,
    output_heads: tuple[str, ...],
    merge_mode: str,
    core_offset: int,
    gaussian_sigma_ratio: float | None,
    show_progress: bool,
) -> dict[str, Any]:
    """
    对全部 BOX 执行 voxel head 推理并合并为整图概率图。

    输入参数:
        - model: torch.nn.Module, eval 模式的 stage1 模型
        - device: str | torch.device, 推理设备
        - box_dicts: list[dict[str, Any]], split_volume_to_boxes() 返回的 BOX 样本列表
        - full_shape_zyx: tuple[int,int,int], 整图形状(D,H,W)
        - hardmask: np.ndarray, (D,H,W), 受体原子落点掩码
        - batch_size: int, 每批 BOX 数
        - output_heads: tuple[str, ...], 需要输出的 head, 允许 ligand/receptor
        - merge_mode: str, 整图聚合方式, 可选 mean/max
        - core_offset: int, 非边界侧裁剪厚度; 0 表示不裁剪
        - gaussian_sigma_ratio: float | None, Gaussian 衰减 sigma 比例; None 表示不启用衰减
        - show_progress: bool, 是否显示 tqdm 进度条

    输出:
        - result: dict[str, Any], 整图概率输出, 包含:
            - "ligand_pred": np.ndarray, (D,H,W), float32, ligand head 合并后的整图概率; 仅当请求 ligand 时存在
            - "receptor_pred": np.ndarray, (D,H,W), float32, receptor head 合并后的整图概率; 仅当请求 receptor 时存在
            - "head_weight_maps": dict[str, np.ndarray], 各 head 的整图权重图, 包含:
                - "ligand": np.ndarray, (D,H,W), float32, ligand head 的累计权重图; 仅当请求 ligand 时存在
                - "receptor": np.ndarray, (D,H,W), float32, receptor head 的累计权重图; 仅当请求 receptor 时存在
    """
    from src.inference.parse_input import prepare_batched_boxes

    if len(box_dicts) == 0:
        raise ValueError("box_dicts 不能为空")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0, 实际为 {batch_size}")
    if int(core_offset) < 0:
        raise ValueError(f"core_offset 不能为负数, 实际为 {core_offset}")
    if merge_mode not in {"mean", "max"}:
        raise ValueError(f"未知 voxel merge_mode: {merge_mode}")

    full_shape_zyx = tuple(int(v) for v in full_shape_zyx)
    # np.ndarray, (D,H,W), float32, 用于概率约束的 hardmask
    hardmask_float = np.asarray(hardmask, dtype=np.float32)
    if tuple(hardmask_float.shape) != full_shape_zyx:
        raise ValueError(f"hardmask.shape={hardmask_float.shape} 与 full_shape_zyx={full_shape_zyx} 不一致")

    # dict[str, np.ndarray], head 名称 → 整图概率加权和
    value_sums = {head_name: np.zeros(full_shape_zyx, dtype=np.float32) for head_name in output_heads}
    # dict[str, np.ndarray], head 名称 → 整图权重累计
    weight_sums = {head_name: np.zeros(full_shape_zyx, dtype=np.float32) for head_name in output_heads}

    batched_iter = prepare_batched_boxes(box_dicts=box_dicts, batch_size=batch_size, device=device)
    if show_progress:
        try:
            from tqdm import tqdm
            total_batches = (len(box_dicts) + batch_size - 1) // batch_size
            batched_iter = tqdm(batched_iter, total=total_batches, desc="voxel boxes")
        except Exception:
            pass

    for batch_dict in batched_iter:
        outputs = run_inference(model=model, device=device, batch_dict=batch_dict)
        head_logits = extract_voxel_head_logits(outputs=outputs, output_heads=output_heads)
        # list[dict], 长度 B, 每个 BOX 的起点元信息
        box_meta_list = batch_dict["_box_meta"]
        for head_name, logits in head_logits.items():
            # np.ndarray, (B,D_box,H_box,W_box), float32, 当前 head 的 BOX 概率
            box_probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy().astype(np.float32)
            for box_index, box_prob in enumerate(box_probs):
                _merge_box_probability_into_full(
                    value_sum=value_sums[head_name],
                    weight_sum=weight_sums[head_name],
                    box_prob=box_prob,
                    box_position_zyx=tuple(int(v) for v in box_meta_list[box_index]["box_position_zyx"]),
                    full_shape_zyx=full_shape_zyx,
                    merge_mode=merge_mode,
                    core_offset=core_offset,
                    gaussian_sigma_ratio=gaussian_sigma_ratio,
                )

    result: dict[str, Any] = {"head_weight_maps": weight_sums}
    for head_name in output_heads:
        if merge_mode == "max":
            # np.ndarray, (D,H,W), float32, max 合并后的概率图
            pred = value_sums[head_name]
        else:
            # np.ndarray, (D,H,W), float32, 加权平均后的概率图
            pred = np.divide(
                value_sums[head_name],
                weight_sums[head_name],
                out=np.zeros_like(value_sums[head_name], dtype=np.float32),
                where=weight_sums[head_name] > 0,
            )
        if head_name == "ligand":
            result["ligand_pred"] = (pred * (1.0 - hardmask_float)).astype(np.float32)
        elif head_name == "receptor":
            result["receptor_pred"] = (pred * hardmask_float).astype(np.float32)

    return result
