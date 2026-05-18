"""
Anchor-based P pseudo atom 的 mixed layout 工具。

对齐契约（修改时必须全量同步）:
    - 本段、CLAUDE/plans/implement/tri_ligand_sparse_refine/00-master.md、Stage1 调用点和 tests/model/test_pseudo_atoms.py 必须同步更新。
    - B 表示 batch 内 BOX 数; N_real=sum(real_counts); N_pseudo=sum(pseudo_counts); N_all=N_real+N_pseudo。
    - mixed layout 固定为每个 BOX 内 `[real_i..., pseudo_i...]`, 不允许把 pseudo 打散到 real 序列中间。

pseudo_dict 字段契约:
    - pseudo_counts: torch.Tensor, (B,), int64/long, 每个 BOX 的 P anchor 数, sum 等于 N_pseudo。
    - pseudo_batch_index: torch.Tensor, (N_pseudo,), int64/long, 每个 P anchor 所属 BOX 索引, 必须按 BOX 分组且与 pseudo_counts 一致。
    - pseudo_coord_centered_world: torch.Tensor, (N_pseudo, 3), floating, P anchor 以 BOX 中心为原点的世界坐标, 轴顺序 (x, y, z)。
    - pseudo_coord_local_voxel: torch.Tensor, (N_pseudo, 3), floating, P anchor 的 corner 语义连续局部体素坐标, 轴顺序 (x, y, z)。
    - pseudo_coord_world: torch.Tensor, (N_pseudo, 3), floating, P anchor 的绝对世界坐标, 轴顺序 (x, y, z)。
    - pseudo_feat: torch.Tensor, (N_pseudo, F_atom), floating, P anchor 初始点特征, F_atom 必须等于 real_batch["atom_feat"].shape[1]。
    - pseudo_is_in_core_box: torch.Tensor, (N_pseudo,), bool, 可选字段, P anchor 是否在 core box 内; 缺省时本模块按全 True 处理。
    - pseudo_anchor_class: torch.Tensor, (N_pseudo,), int64/long, 可选字段, P anchor 的候选类别索引, 类别顺序由候选 C 模块定义。
    - pseudo_anchor_voxel_zyx: torch.Tensor, (N_pseudo, 3), int64/long, 可选字段, P anchor 来源体素坐标, 轴顺序 (z, y, x)。
    - pseudo_source_candidate_index: torch.Tensor, (N_pseudo,), int64/long, 可选字段, P anchor 对应的候选 C 行索引。

inject_pseudo_atoms 输出契约:
    - mixed_batch["real_mask"]: torch.Tensor, (N_all,), bool, True 表示 real atom。
    - mixed_batch["pseudo_mask"]: torch.Tensor, (N_all,), bool, True 表示 P anchor。
    - mixed_batch["atom_valid_mask"]: torch.Tensor, (N_all,), bool, P anchor 槽位固定为 False, 不参与 real atom 监督。
    - mixed_batch["atom_label"]: torch.Tensor, (N_all,), long, P anchor 槽位固定为 0, 仅作占位。
    - mixed_batch["atom_global_indices"]: torch.Tensor, (N_all,), long, P anchor 槽位固定为 -1, 表示无真实原子全局索引。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch


# --------------------------------------------------- 计数 ---------------------------------------------------
@dataclass(frozen=True)
class PseudoAtomLayout:
    """
    描述 real atom 与 P anchor 在 mixed batch 中的逐 BOX 布局。

    输入参数:
        - real_counts: torch.Tensor, (B,), long, 每个 BOX 的真实原子数
        - pseudo_counts: torch.Tensor, (B,), long, 每个 BOX 的 P anchor 数

    属性输出:
        - mixed_counts: torch.Tensor, (B,), long, 每个 BOX 的 mixed 点数
        - batch_size: int, BOX 数量
    """

    real_counts: torch.Tensor
    pseudo_counts: torch.Tensor

    @property
    def mixed_counts(self) -> torch.Tensor:
        return self.real_counts + self.pseudo_counts

    @property
    def batch_size(self) -> int:
        return int(self.real_counts.shape[0])

def compute_atom_counts(batch: dict[str, Any]) -> torch.Tensor:
    """
    从 real-only batch 中读取或计算每个 BOX 的真实原子数。

    输入参数:
        - batch: dict[str, Any], collate 后的 real-only batch, 必须包含
            - atom_batch_index: torch.Tensor, (sumN,), long,  指明展平的点云序列中，每个点属于当前 batch 内哪个样本 (0 ~ B-1)
            - box_shape_zyx: torch.Tensor, (B,3), long, 每个BOX的体素形状，顺序为(Z,Y,X)对应(depth,height,width)

    输出:
        - atom_counts: torch.Tensor, (B,), long, 每个 BOX 的真实原子数
    """
    # torch.Tensor | None, (B,), long, batch 中已有的真实原子计数
    atom_counts = batch.get("atom_counts")
    # torch.Tensor, (N_real,), long, 真实原子所属 BOX 索引
    atom_batch_index = batch["atom_batch_index"].to(dtype=torch.long)
    if atom_counts is not None and int(atom_counts.sum().item()) == int(atom_batch_index.shape[0]):
        return atom_counts.to(device=atom_batch_index.device, dtype=torch.long)

    # int, batch 内 BOX 数量
    batch_size = int(batch["box_shape_zyx"].shape[0])
    if atom_batch_index.numel() == 0:
        return torch.zeros(batch_size, dtype=torch.long, device=atom_batch_index.device)
    return torch.bincount(atom_batch_index, minlength=batch_size).to(dtype=torch.long)

def build_layout(real_batch: dict[str, Any], pseudo_dict: dict[str, Any]) -> PseudoAtomLayout:
    """
    根据 real-only batch 与 anchor-based pseudo_dict 构造 mixed layout。

    输入参数:
        - real_batch: dict[str, Any], real-only batch, 提供真实原子计数
        - pseudo_dict: dict[str, Any], anchor sampler 输出后的 P anchor 字典, 必须包含 pseudo_counts

    输出:
        - layout: PseudoAtomLayout, (B,), real/pseudo/mixed 逐 BOX 计数
    """
    # torch.Tensor, (B,), long, 每个 BOX 的真实原子数
    real_counts = compute_atom_counts(real_batch)
    # torch.Tensor, (B,), long, 每个 BOX 的 P anchor 数
    pseudo_counts = pseudo_dict["pseudo_counts"].to(device=real_counts.device, dtype=torch.long)
    if pseudo_counts.shape[0] != real_counts.shape[0]:
        raise RuntimeError("pseudo_counts 的 batch 维度必须与 real_counts 一致。")
    return PseudoAtomLayout(real_counts=real_counts, pseudo_counts=pseudo_counts)

def build_real_mask(layout: PseudoAtomLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """
    按 `[real_i, pseudo_i]` mixed 顺序构造真实原子掩码。

    输入参数:
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数
        - device: torch.device | None, 输出掩码设备; None 表示使用 layout.real_counts.device

    输出:
        - real_mask: torch.Tensor, (sum(real_counts + pseudo_counts),), bool, True 表示真实原子
    """
    output_device = layout.real_counts.device if device is None else device
    # int, mixed 点总数
    total_mixed = int(layout.mixed_counts.sum().item())
    # torch.Tensor, (sumN_all,), bool, mixed 布局下的真实原子掩码
    real_mask = torch.zeros(total_mixed, dtype=torch.bool, device=output_device)
    offset = 0
    for real_count, pseudo_count in zip(layout.real_counts.tolist(), layout.pseudo_counts.tolist()):
        real_mask[offset : offset + int(real_count)] = True
        offset += int(real_count) + int(pseudo_count)
    return real_mask

def build_pseudo_mask(layout: PseudoAtomLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """
    按 `[real_i, pseudo_i]` mixed 顺序构造 P anchor 掩码。

    输入参数:
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数
        - device: torch.device | None, 输出掩码设备; None 表示使用 layout.real_counts.device

    输出:
        - pseudo_mask: torch.Tensor, (sum(real_counts + pseudo_counts),), bool, True 表示 P anchor
    """
    return ~build_real_mask(layout, device=device)





#  --------------------------------------------------- 张量管理  ---------------------------------------------------
def interleave_real_and_pseudo_tensor(
    real_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
    pseudo_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    将 real-only 张量与 pseudo-only 张量按 `[real_i, pseudo_i]` 拼成 mixed 张量。

    输入参数:
        - real_tensor: torch.Tensor | None, (sumN_real, ...), real-only 张量; 如果为 None 则直接返回 None
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数
        - pseudo_tensor: torch.Tensor | None, (sumP, ...), pseudo-only 张量; 如果为 None 则用 0 填充 P anchor 槽位

    输出:
        - mixed_tensor: torch.Tensor | None, (sumN_real + sumP, ...), mixed 布局张量
    """
    if real_tensor is None:
        return None
    # int, real-only 点数
    total_real = int(layout.real_counts.sum().item())
    # int, pseudo-only 点数
    total_pseudo = int(layout.pseudo_counts.sum().item())
    if int(real_tensor.shape[0]) != total_real:
        raise RuntimeError(f"real_tensor 必须是 real-only 长度 {total_real}，当前为 {real_tensor.shape[0]}。")
    if pseudo_tensor is not None and int(pseudo_tensor.shape[0]) != total_pseudo:
        raise RuntimeError(f"pseudo_tensor 必须是 pseudo-only 长度 {total_pseudo}，当前为 {pseudo_tensor.shape[0]}。")

    # tuple[int, ...], 张量除点维之外的后缀形状
    suffix_shape = tuple(real_tensor.shape[1:])
    # list[torch.Tensor], 按 BOX 顺序排列的 real/pseudo 片段
    chunks: list[torch.Tensor] = []
    real_offset = 0
    pseudo_offset = 0
    for real_count, pseudo_count in zip(layout.real_counts.tolist(), layout.pseudo_counts.tolist()):
        real_count_int = int(real_count)
        pseudo_count_int = int(pseudo_count)
        if real_count_int > 0:
            chunks.append(real_tensor[real_offset : real_offset + real_count_int])
            real_offset += real_count_int
        if pseudo_count_int > 0:
            if pseudo_tensor is None:
                chunks.append(real_tensor.new_zeros((pseudo_count_int,) + suffix_shape))
            else:
                chunks.append(
                    pseudo_tensor[pseudo_offset : pseudo_offset + pseudo_count_int].to(
                        device=real_tensor.device,
                        dtype=real_tensor.dtype,
                    )
                )
            pseudo_offset += pseudo_count_int
    if chunks:
        return torch.cat(chunks, dim=0)
    return real_tensor.new_empty((0,) + suffix_shape)

def _interleave_required_field(
    real_batch: dict[str, Any],
    pseudo_dict: dict[str, Any],
    layout: PseudoAtomLayout,
    real_field: str,
    pseudo_field: str,
) -> torch.Tensor:
    """
    交错拼接 real batch 与 pseudo dict 的必需字段。

    输入参数:
        - real_batch: dict[str, Any], real-only batch
        - pseudo_dict: dict[str, Any], P anchor 字典
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数
        - real_field: str, real batch 字段名
        - pseudo_field: str, pseudo dict 字段名

    输出:
        - mixed_tensor: torch.Tensor, (sumN_real + sumP, ...), mixed 字段张量
    """
    return interleave_real_and_pseudo_tensor(real_batch[real_field], layout, pseudo_dict[pseudo_field])

def extract_real_tensor_from_mixed(
    mixed_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
) -> torch.Tensor | None:
    """
    从 mixed 张量中提取真实原子子张量。

    输入参数:
        - mixed_tensor: torch.Tensor | None, (sumN_real + sumP, ...), mixed 布局张量
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数

    输出:
        - real_tensor: torch.Tensor | None, (sumN_real, ...), real-only 子张量
    """
    if mixed_tensor is None:
        return None
    # torch.Tensor, (sumN_all,), bool, True 表示真实原子
    real_mask = build_real_mask(layout, device=mixed_tensor.device)
    if int(mixed_tensor.shape[0]) != int(real_mask.shape[0]):
        raise RuntimeError(f"mixed_tensor 长度必须为 {real_mask.shape[0]}，当前为 {mixed_tensor.shape[0]}。")
    return mixed_tensor[real_mask]

def extract_pseudo_tensor_from_mixed(
    mixed_tensor: torch.Tensor | None,
    layout: PseudoAtomLayout,
) -> torch.Tensor | None:
    """
    从 mixed 张量中提取 P anchor 子张量。

    输入参数:
        - mixed_tensor: torch.Tensor | None, (sumN_real + sumP, ...), mixed 布局张量
        - layout: PseudoAtomLayout, (B,), real/pseudo 逐 BOX 计数

    输出:
        - pseudo_tensor: torch.Tensor | None, (sumP, ...), pseudo-only 子张量
    """
    if mixed_tensor is None:
        return None
    # torch.Tensor, (sumN_all,), bool, True 表示 P anchor
    pseudo_mask = build_pseudo_mask(layout, device=mixed_tensor.device)
    if int(mixed_tensor.shape[0]) != int(pseudo_mask.shape[0]):
        raise RuntimeError(f"mixed_tensor 长度必须为 {pseudo_mask.shape[0]}，当前为 {mixed_tensor.shape[0]}。")
    return mixed_tensor[pseudo_mask]





#  --------------------------------------------------- (对batch的)点级管理  ---------------------------------------------------
def inject_pseudo_atoms(
    real_batch: dict[str, Any],
    pseudo_dict: dict[str, Any],
) -> tuple[dict[str, Any], PseudoAtomLayout]:
    """
    将 anchor sampler 准备好的 P anchors 注入 real-only batch, 供最后一轮 point backbone 消费。

    输入参数:
        - real_batch: dict[str, Any], real-only batch, 来自 embed head 裁剪后的 canonical 视图
        - pseudo_dict: dict[str, Any], P anchor 字典, 后续由 candidate/anchor pipeline 构造

    输出:
        - mixed_batch: dict[str, Any], (sumN_real + sumP, ...), 按 `[real_i, pseudo_i]` 排列的 point backbone 输入 batch
        - layout: PseudoAtomLayout, (B,)
    """
    # PseudoAtomLayout, (B,), real/pseudo/mixed 逐 BOX 计数
    layout = build_layout(real_batch, pseudo_dict)
    # torch.device, mixed batch 主设备
    device = real_batch["atom_coord_centered_world"].device
    # int, P anchor 总数
    total_pseudo = int(layout.pseudo_counts.sum().item())

    # dict[str, Any], mixed batch 浅拷贝; batch-level 字段沿用 real_batch
    mixed_batch = {**real_batch}
    mixed_batch["atom_coord_centered_world"] = _interleave_required_field(
        real_batch, pseudo_dict, layout, "atom_coord_centered_world", "pseudo_coord_centered_world"
    )
    mixed_batch["atom_coord_local_voxel"] = _interleave_required_field(
        real_batch, pseudo_dict, layout, "atom_coord_local_voxel", "pseudo_coord_local_voxel"
    )
    mixed_batch["atom_coord_world"] = _interleave_required_field(
        real_batch, pseudo_dict, layout, "atom_coord_world", "pseudo_coord_world"
    )
    mixed_batch["atom_feat"] = _interleave_required_field(real_batch, pseudo_dict, layout, "atom_feat", "pseudo_feat")

    # torch.Tensor, (sumP,), bool, P anchor 不参与 real atom 监督
    pseudo_valid_mask = torch.zeros(total_pseudo, dtype=torch.bool, device=device)
    # torch.Tensor, (sumP,), long, P anchor 的占位 atom label
    pseudo_label = torch.zeros(total_pseudo, dtype=real_batch["atom_label"].dtype, device=device)
    # torch.Tensor, (sumP,), bool, P anchor 默认视为 core 内点; 后续 anchor pipeline 可显式覆盖
    pseudo_core = pseudo_dict.get(
        "pseudo_is_in_core_box",
        torch.ones(total_pseudo, dtype=torch.bool, device=device),
    )
    # torch.Tensor, (sumP,), long, P anchor 无真实 atom 全局索引
    pseudo_global_indices = torch.full(
        (total_pseudo,),
        fill_value=-1,
        dtype=real_batch["atom_global_indices"].dtype,
        device=device,
    )

    mixed_batch["atom_valid_mask"] = interleave_real_and_pseudo_tensor(
        real_batch["atom_valid_mask"], layout, pseudo_valid_mask
    )
    mixed_batch["atom_label"] = interleave_real_and_pseudo_tensor(real_batch["atom_label"], layout, pseudo_label)
    mixed_batch["atom_is_in_core_box"] = interleave_real_and_pseudo_tensor(
        real_batch["atom_is_in_core_box"], layout, pseudo_core
    )
    mixed_batch["atom_global_indices"] = interleave_real_and_pseudo_tensor(
        real_batch["atom_global_indices"], layout, pseudo_global_indices
    )

    # torch.Tensor, (B,), long, mixed 点数累积 offset
    mixed_batch["atom_counts"] = layout.mixed_counts.to(device=device)
    mixed_batch["atom_offsets"] = torch.cumsum(mixed_batch["atom_counts"], dim=0)
    # torch.Tensor, (sumN_all,), long, mixed 点所属 BOX 索引
    mixed_batch["atom_batch_index"] = torch.repeat_interleave(
        torch.arange(layout.batch_size, dtype=torch.long, device=device),
        mixed_batch["atom_counts"],
    )
    mixed_batch["real_mask"] = build_real_mask(layout, device=device)
    mixed_batch["pseudo_mask"] = build_pseudo_mask(layout, device=device)

    for optional_key in ("pseudo_anchor_class", "pseudo_anchor_voxel_zyx", "pseudo_source_candidate_index"):
        if optional_key in pseudo_dict:
            mixed_batch[optional_key] = pseudo_dict[optional_key]
    return mixed_batch, layout

def remove_pseudo_atoms(
    mixed_batch: dict[str, Any],
    layout: PseudoAtomLayout,
) -> dict[str, Any]:
    """
    从 mixed batch 中移除 P anchors, 恢复 wrapper-facing real-only batch 字段。

    输入参数:
        - mixed_batch: dict[str, Any], inject_pseudo_atoms 输出的 mixed batch
        - layout: PseudoAtomLayout, (B,), mixed layout 计数

    输出:
        - real_batch: dict[str, Any], (sumN_real, ...), real-only batch 视图
    """
    # torch.Tensor, (sumN_all,), bool, True 表示真实原子
    real_mask = build_real_mask(layout, device=mixed_batch["atom_coord_centered_world"].device)
    # dict[str, Any], real-only batch 浅拷贝
    real_batch = {**mixed_batch}
    for field_name in (
        "atom_coord_centered_world",
        "atom_coord_local_voxel",
        "atom_coord_world",
        "atom_feat",
        "atom_valid_mask",
        "atom_label",
        "atom_is_in_core_box",
        "atom_global_indices",
    ):
        if field_name in mixed_batch and mixed_batch[field_name] is not None:
            real_batch[field_name] = mixed_batch[field_name][real_mask]
    real_batch["atom_counts"] = layout.real_counts.to(device=real_mask.device)
    real_batch["atom_offsets"] = torch.cumsum(real_batch["atom_counts"], dim=0)
    real_batch["atom_batch_index"] = torch.repeat_interleave(
        torch.arange(layout.batch_size, dtype=torch.long, device=real_mask.device),
        real_batch["atom_counts"],
    )
    for mixed_only_key in (
        "real_mask",
        "pseudo_mask",
        "pseudo_anchor_class",
        "pseudo_anchor_voxel_zyx",
        "pseudo_source_candidate_index",
    ):
        real_batch.pop(mixed_only_key, None)
    return real_batch





#  --------------------------------------------- 特化函数: 只为了主模型最后输出的整理  ---------------------------------------------
def filter_point_state_with_mask(    # 下面函数的工具函数
    point_state: dict[str, Any],
    keep_mask: torch.Tensor,
    counts: torch.Tensor,
) -> dict[str, Any]:
    """
    用点级 keep_mask 裁剪 point_state 并重建 offset。

    输入参数:
        - point_state: dict[str, Any], point backbone 输出的点状态(具体可见 src\model\stage1_point_backbone.py, 就是 coords, batch, offset, grid_size, grid_coord)
        - keep_mask: torch.Tensor, (N,), bool, True 表示保留该点
        - counts: torch.Tensor, (B,), long, 裁剪后每个 BOX 的点数

    输出:
        - filtered_state: dict[str, Any], 与 keep_mask 对齐后的点状态
    """
    # dict[str, Any], 裁剪后的 point_state 浅拷贝
    filtered_state = {**point_state}
    filtered_state["coord"] = point_state["coord"][keep_mask]
    filtered_state["batch"] = point_state["batch"][keep_mask]
    filtered_state["offset"] = torch.cumsum(counts.to(device=point_state["coord"].device), dim=0)
    if "grid_coord" in point_state and point_state["grid_coord"] is not None:
        filtered_state["grid_coord"] = point_state["grid_coord"][keep_mask]
    if "pseudo_mask" in point_state and point_state["pseudo_mask"] is not None:
        filtered_state["pseudo_mask"] = point_state["pseudo_mask"][keep_mask]
    return filtered_state

def extract_real_point_output(    # 特化函数
    mixed_batch: dict[str, Any],
    fused_point_feat: torch.Tensor,
    point_output_dict: dict[str, Any],
    layout: PseudoAtomLayout,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], dict[str, Any]]:
    """
    从最后一轮 mixed point backbone 输出中提取 real-only 视图, 供 recycle 与 wrapper 继续消费。

    输入参数:
        - mixed_batch: dict[str, Any], inject_pseudo_atoms 输出的 mixed batch
        - fused_point_feat: torch.Tensor, (sumN_real + sumP, C_point), point backbone 输出特征
        - point_output_dict: dict[str, Any], point backbone 原始输出字典
        - layout: PseudoAtomLayout, (B,), mixed layout 计数

    输出:
        - real_batch: dict[str, Any], real-only batch 视图
        - real_fused_point_feat: torch.Tensor, (sumN_real, C_point), 真实原子点特征
        - real_point_state: dict[str, Any], real-only point_state
        - real_point_output_dict: dict[str, Any], real-only point backbone 输出字典
    """
    # torch.Tensor, (sumN_all,), bool, True 表示真实原子
    real_mask = build_real_mask(layout, device=fused_point_feat.device)
    # dict[str, Any], real-only batch 视图
    real_batch = remove_pseudo_atoms(mixed_batch, layout)
    # torch.Tensor, (sumN_real, C_point), 真实原子点特征
    real_fused_point_feat = fused_point_feat[real_mask]
    # dict[str, Any], 真实原子的 point_state
    real_point_state = filter_point_state_with_mask(
        point_state=point_output_dict["point_state"],
        keep_mask=real_mask,
        counts=layout.real_counts,
    )

    # dict[str, Any], 对外返回的 real-only point output
    real_point_output_dict = {**point_output_dict}
    real_point_output_dict["point_feat"] = real_fused_point_feat
    real_point_output_dict["point_state"] = real_point_state
    if point_output_dict.get("point_recycle_out") is not None:
        real_point_output_dict["point_recycle_out"] = point_output_dict["point_recycle_out"][real_mask]

    # dict[str, torch.Tensor], point backbone 导出的命名特征; 第一维等于 mixed 点数的张量同步裁剪
    feature_dict = point_output_dict.get("point_feature_dict")
    if isinstance(feature_dict, dict):
        trimmed_feature_dict: dict[str, Any] = {}
        for feature_name, feature_value in feature_dict.items():
            if torch.is_tensor(feature_value) and feature_value.ndim >= 1 and feature_value.shape[0] == real_mask.shape[0]:
                trimmed_feature_dict[feature_name] = feature_value[real_mask]
            else:
                trimmed_feature_dict[feature_name] = feature_value
        real_point_output_dict["point_feature_dict"] = trimmed_feature_dict
    return real_batch, real_fused_point_feat, real_point_state, real_point_output_dict
