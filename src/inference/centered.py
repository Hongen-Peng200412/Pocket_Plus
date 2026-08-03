"""从完整 Stage1 forward 生成并发布三类 centered BOX 归档。

本模块保留两段可直接阅读的结构：分隔线之前是 NumPy/torch 转换、空间包络、entry 组装和单个 forward batch 拆分等冷读工具；
分隔线之后是完整 forward、F1、CLG、Selected 和 NPZ 发布的业务演进顺序。Dataset、Collator、wrapper 和 centered_batch_size 都在主流程的真实调用位置出现，不通过额外 callback 隐藏。

主要产物字段保持 centered/BOX-level 契约：共同身份与几何字段、权威 voxel 的 voxel_index_local_zyx/centered_probability/voxel_final、hardmask auxiliary 字段，以及 Find 专属的 P/A 表和 CLG/Selected 专属状态与 membership。P/A 的 L4 特征不落盘。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from src.artifacts.io import atomic_savez_compressed, pack_centered_entries, validate_centered_archive
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import mark_role_complete
from src.component_lineage.structures import CLG, ComponentNode
from src.stage1_producers import FIND_MODEL_NAMES

from .probability import logits_to_probability, postprocess_ligand_probability


@dataclass(frozen=True)
class CenteredGeometry:
    """保存完整图和固定 80³ BOX 的离散、物理几何关系。

    输入字段:

    - `full_shape_zyx`: tuple[int, int, int]，完整图的 `(D, H, W)` ZYX shape。
    - `origin_xyz`: float32 `(3,)`，完整图 voxel-grid 原点的世界 XYZ 坐标，单位 Å。
    - `voxel_size_xyz`: float32 `(3,)`，世界 XYZ 方向的 voxel 尺寸，单位 Å/voxel。
    - `box_shape_zyx`: tuple[int, int, int]，centered BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
    """
    full_shape_zyx: tuple[int, int, int]
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    box_shape_zyx: tuple[int, int, int] = (80, 80, 80)
    def __post_init__(self) -> None:
        object.__setattr__(self, "full_shape_zyx", tuple(int(value) for value in self.full_shape_zyx))
        object.__setattr__(self, "box_shape_zyx", tuple(int(value) for value in self.box_shape_zyx))
        object.__setattr__(self, "origin_xyz", np.asarray(self.origin_xyz, dtype=np.float32))
        object.__setattr__(self, "voxel_size_xyz", np.asarray(self.voxel_size_xyz, dtype=np.float32))


@dataclass(frozen=True)
class CenteredRequest:
    """描述一次 centered BOX 裁剪所需的 Dataset 请求。

    字段语义:

    - stage1_model_name: str，当前 checkpoint 的正式 producer 名。
    - split: str，当前数据划分；正式值为 calibration、validation 或 train。
    - pdb_id: str，当前完整图和 receptor 表共用的小写 PDB identity。
    - centered_role: str，F1_centered、CLG_centered 或 Selected_Refined_Centered。
    - centered_box_index: int，同一 PDB/role 归档中的连续 entry 行号。
    - box_start_zyx: tuple[int, int, int]，完整图 ZYX voxel-grid 中的 BOX 起点。
    - source_tree_id: int，来源 component 所属 forest tree 编号。
    - source_node_id: int，来源 component 在 forest tree 内的节点编号。
    - source_threshold_grid_index: int，来源 component 的冻结阈值网格编号。
    """
    stage1_model_name: str
    split: str
    pdb_id: str
    centered_role: str
    centered_box_index: int
    box_start_zyx: tuple[int, int, int]
    source_tree_id: int
    source_node_id: int
    source_threshold_grid_index: int


CenteredBatchBuilder = Callable[[Sequence[CenteredRequest]], Mapping[str, Any]]
StartResolver = Callable[[np.ndarray, tuple[int, int, int]], Sequence[int]]


# ============================================================================
# 冷读工具：只保留有独立数据语义、被多个主流程使用或需要单独测试的函数。
def _to_numpy(value: Any) -> np.ndarray:
    """把 NumPy 或 torch.Tensor 搬到 CPU；bfloat16 在 NumPy 边界提升为 float32。"""

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.numpy()
    return np.asarray(value)

def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    """把输入转换为 CPU NumPy 数组，并压到指定 dtype。"""
    return np.asarray(_to_numpy(value), dtype=dtype)

def _sigmoid_rows(value: Any) -> np.ndarray:
    """对逐行 logits 计算不会在负值分支溢出的 float32 sigmoid。"""
    logits = _as_numpy(value, np.float32).reshape(-1)
    probability = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probability[~positive] = exp_logits / (1.0 + exp_logits)
    return probability

def _points_within_blob_envelope(
    points_local_xyz: np.ndarray,
    blob_global_linear_index: np.ndarray,
    box_start_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
    voxel_size_xyz: np.ndarray,
    distance_angstrom: float,
) -> np.ndarray:
    """返回 BOX 内各点是否处于来源 blob voxel center 的距离包络内。

    输入字段:

    - `points_local_xyz`: float array `(N_point, 3)`，BOX-local XYZ voxel 坐标。
    - `blob_global_linear_index`: int64 `(K_blob,)`，完整图 ZYX 网格的 C-order 线性索引。
    - `box_start_zyx`: int64 `(3,)`，当前 BOX 在完整图中的 ZYX 起点。
    - `full_shape_zyx`: tuple[int, int, int]，完整图 `(D, H, W)` shape。
    - `voxel_size_xyz`: float32 `(3,)`，世界 XYZ voxel 尺寸。
    - `distance_angstrom`: float，世界坐标距离上限，单位 Å。

    输出:

    - `within`: bool `(N_point,)`，每个点是否至少接近一个来源 blob voxel center。
    """

    points = np.asarray(points_local_xyz, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.bool_)
    global_zyx = np.column_stack(
        np.unravel_index(np.asarray(blob_global_linear_index, dtype=np.int64), full_shape_zyx)
    )
    local_zyx = global_zyx - box_start_zyx[None, :]
    centers_local_xyz = (local_zyx[:, [2, 1, 0]] + 0.5) * voxel_size_xyz[None, :]
    points_local_world = points * voxel_size_xyz[None, :]
    distance, _ = cKDTree(centers_local_xyz).query(
        points_local_world,
        k=1,
        distance_upper_bound=float(distance_angstrom),
    )
    return np.isfinite(distance)

def _base_entry(
    source: ComponentNode,
    centered_role: str,
    centered_box_index: int,
    box_start_zyx: np.ndarray,
    geometry: CenteredGeometry,
) -> dict[str, Any]:
    """对 source 构造三个 centered role 共用纯身份(source_tree_id / source_node_id / source_threshold_grid_index / source_threshold_value)与 BOX 几何字段。"""

    box_start_xyz = box_start_zyx[[2, 1, 0]]
    box_origin_world = geometry.origin_xyz + box_start_xyz * geometry.voxel_size_xyz
    return {
        "centered_role": centered_role,
        "centered_box_index": int(centered_box_index),
        "box_start_zyx": box_start_zyx.astype(np.int32),
        "box_shape_zyx": np.asarray(geometry.box_shape_zyx, dtype=np.uint8),
        "box_origin_world": box_origin_world.astype(np.float32),
        "voxel_size_world": geometry.voxel_size_xyz.astype(np.float32),
        "source_tree_id": int(source.tree_id),
        "source_node_id": int(source.node_id),
        "source_threshold_grid_index": int(source.threshold_grid_index),
        "source_threshold_value": np.float32(source.threshold_value),
    }

def iter_stage1_centered_batch_payloads(
    forward_output: Mapping[str, Any],
    batch: Mapping[str, Any],
    stage1_model_name: str,
) -> Iterator[dict[str, np.ndarray]]:
    """把一个完整 forward batch 拆成按 BOX 对齐的 V 或 V/P/A payload。

    输入:
        - forward_output: Mapping[str, Any]，完整 wrapper 的具名输出；必须包含 ligand/aux logits、voxel_features["voxel_final"]，Find 还必须包含 A/P logits、坐标、归属和 L1-L3 特征。
        - batch: Mapping[str, Any]，训练同源 Collator 生成的目标设备 batch；使用 hardmask、box_shape_zyx、voxel_size_world，以及与原始受体行对齐的 atom_global_indices 和 atom_feat。
        - stage1_model_name: str，决定只拆 V/aux，还是同时拆 Find 的 ragged A/P 表。

    每个 BOX payload 的字段:
        - ligand_probability: float32 (80, 80, 80)，BOX-local ZYX ligand sigmoid 概率，尚未应用 Find hardmask。
        - hardmask: bool (80, 80, 80)，BOX-local ZYX receptor home-voxel 掩码。
        - voxel_aux_probability_grid: float32 (80, 80, 80)，与 hardmask 同一坐标系的 auxiliary receptor 概率。
        - voxel_final_grid: numeric (C_voxel, 80, 80, 80)，channel-first 最终 V 网格，C_voxel 取 checkpoint 实际通道数。
        - A_global_index: int64 (N_A,)，A 原子在当前 PDB receptor 表中的全局行号。
        - A_coord_local_xyz: float32 (N_A, 3)，A 原子的 BOX-local XYZ voxel 坐标。
        - A_coord_centered_world: float32 (N_A, 3)，A 原子相对 BOX 中心的世界 XYZ 坐标，单位 Å。
        - A_probability: float32 (N_A,)，与 A 行对齐的 sigmoid 概率。
        - A_feat_L0: float32 (N_A, 49)，按 A_global_index 从当前输入 batch 对齐的原始受体特征。
        - A_feat_L1: numeric (N_A, C_A1)，与 A_global_index 逐行对齐的 A L1 特征。
        - A_feat_L2: numeric (N_A, C_A2)，与 A_global_index 逐行对齐的 A L2 特征。
        - A_feat_L3: numeric (N_A, C_A3)，与 A_global_index 逐行对齐的 A L3 特征。
        - P_coord_local_xyz: float32 (N_P, 3)，P anchor 的 BOX-local XYZ voxel 坐标。
        - P_probability: float32 (N_P,)，与 P 行对齐的 sigmoid 概率。
        - P_feat_L2: numeric (N_P, C_P2)，与 P_coord_local_xyz 逐行对齐的 P L2 特征。
        - P_feat_L3: numeric (N_P, C_P3)，与 P_coord_local_xyz 逐行对齐的 P L3 特征。

    dense 字段按 batch 第一维切分；A 表按 forward 后的 atom_counts 连续段切分，并在每个 BOX 内用 A_global_index 把输入 atom_feat 重排到 forward 输出顺序；P 表按 anchor_batch_index 筛选。每个 yield 保留模型输出中的实体顺序。
    """
    ligand_probability = logits_to_probability(forward_output["voxel_logits_ligand"])
    aux_probability = logits_to_probability(forward_output["voxel_logits_aux"])
    hardmask = _as_numpy(batch["hardmask"], np.bool_)
    if hardmask.ndim == 5:
        hardmask = hardmask[:, 0]
    voxel_features = forward_output["voxel_features"]

    if stage1_model_name not in FIND_MODEL_NAMES:
        for batch_index in range(ligand_probability.shape[0]):
            yield {
                "ligand_probability": ligand_probability[batch_index],
                "hardmask": hardmask[batch_index],
                "voxel_aux_probability_grid": aux_probability[batch_index],
                "voxel_final_grid": _to_numpy(voxel_features["voxel_final"][batch_index]),
            }
        return

    atom_counts = _as_numpy(forward_output["atom_counts"], np.int64).reshape(-1)
    atom_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(atom_counts)))
    atom_local_xyz = _as_numpy(forward_output["atom_coord_local_voxel"], np.float32)
    input_atom_counts = _as_numpy(batch["atom_counts"], np.int64).reshape(-1)
    input_atom_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(input_atom_counts))
    )
    input_atom_global_index = _as_numpy(
        batch["atom_global_indices"], np.int64
    ).reshape(-1)
    input_atom_feat_l0 = _as_numpy(batch["atom_feat"], np.float32)
    if input_atom_feat_l0.ndim != 2 or input_atom_feat_l0.shape[1] != 49:
        raise ValueError("batch['atom_feat'] 必须为 [N_A_input,49]")
    atom_values = {
        "A_global_index": _as_numpy(forward_output["atom_global_indices"], np.int64).reshape(-1),
        "A_probability": _sigmoid_rows(forward_output["atom_logits"]),
        "A_feat_L1": _to_numpy(forward_output["A_feat_L1"]),
        "A_feat_L2": _to_numpy(forward_output["A_feat_L2"]),
        "A_feat_L3": _to_numpy(forward_output["A_feat_L3"]),
    }
    pseudo_batch_index = _as_numpy(forward_output["anchor_batch_index"], np.int64).reshape(-1)
    pseudo_values = {
        "P_coord_local_xyz": _as_numpy(forward_output["anchor_coord_local_voxel"], np.float32),
        "P_probability": _sigmoid_rows(forward_output["pseudo_logits"]),
        "P_feat_L2": _to_numpy(forward_output["P_feat_L2"]),
        "P_feat_L3": _to_numpy(forward_output["P_feat_L3"]),
    }
    box_shape_xyz = _as_numpy(batch["box_shape_zyx"], np.float32)[:, [2, 1, 0]]
    voxel_size_xyz = _as_numpy(batch["voxel_size_world"], np.float32)

    for batch_index in range(ligand_probability.shape[0]):
        atom_slice = slice(int(atom_offsets[batch_index]), int(atom_offsets[batch_index + 1]))
        input_atom_slice = slice(
            int(input_atom_offsets[batch_index]),
            int(input_atom_offsets[batch_index + 1]),
        )
        payload = {
            "ligand_probability": ligand_probability[batch_index],
            "hardmask": hardmask[batch_index],
            "voxel_aux_probability_grid": aux_probability[batch_index],
            "voxel_final_grid": _to_numpy(voxel_features["voxel_final"][batch_index]),
        }
        payload.update({field: value[atom_slice] for field, value in atom_values.items()})
        input_rows_by_global_index = {
            int(global_index): row
            for row, global_index in enumerate(input_atom_global_index[input_atom_slice])
        }
        output_global_indices = payload["A_global_index"]
        if len(input_rows_by_global_index) != int(input_atom_counts[batch_index]):
            raise ValueError("同一 BOX 的输入 atom_global_indices 必须唯一")
        try:
            # !! 反向根据裁剪后的 global index 定位出裁剪前batch的"box内部的原子" !!
            l0_rows = np.asarray(
                [input_rows_by_global_index[int(value)] for value in output_global_indices],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "forward 输出的 A_global_index 在当前 BOX 输入原子表中不存在"
            ) from error
        payload["A_feat_L0"] = input_atom_feat_l0[input_atom_slice][l0_rows]
        payload["A_coord_local_xyz"] = atom_local_xyz[atom_slice]
        payload["A_coord_centered_world"] = (
            (payload["A_coord_local_xyz"] - box_shape_xyz[batch_index][None, :] / np.float32(2.0)) * voxel_size_xyz[batch_index][None, :]
        ).astype(np.float32, copy=False)
        pseudo_rows = pseudo_batch_index == batch_index
        payload.update({field: value[pseudo_rows] for field, value in pseudo_values.items()})
        yield payload

def _payload_from_forward(
    base: Mapping[str, Any],
    authority_global_linear_index: np.ndarray,
    probability: np.ndarray,
    forward_payload: Mapping[str, Any],
    geometry: CenteredGeometry,
    stage1_model_name: str,
) -> dict[str, Any]:
    """在 base 中加入一揽子字段(如forward产生的 forward_payload)。

    输入参数:
    - base: `_base_entry` 生成的当前 entry 。
        - centered_role: str，当前 centered 产物角色，取 F1、CLG 或 Selected 角色名。
        - centered_box_index: int，同一 PDB/角色归档中的连续 entry 行号。
        - box_start_zyx: int32 `(3,)`，完整图 ZYX voxel 网格中的离散 BOX 起点。
        - box_shape_zyx: uint8 `(3,)`，当前 BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
        - box_origin_world: float32 `(3,)`，BOX 角点的世界 XYZ 坐标，单位 Å。
        - voxel_size_world: float32 `(3,)`，世界 XYZ 方向的 voxel spacing，单位 Å/voxel。
        - source_tree_id: int 标量，来源 component 所属 forest tree 编号。
        - source_node_id: int 标量，来源 component 在 forest tree 内的节点编号。
        - source_threshold_grid_index: int 标量，来源 component 的冻结阈值网格编号。
        - source_threshold_value: float32 标量，来源 component 对应的原始阈值数值。
    - authority_global_linear_index: 整数数组，转换为 int64 `(K_source,)`；`K_source` 是来源 component 的权威 voxel 数，即该数组第一维长度；每个值是完整图 `(D, H, W)` ZYX 网格按 C-order 展平（X/W 轴最快）的全局线性索引；数组顺序是来源 component 的权威 voxel 顺序，输出 V 与概率字段必须保持该顺序。
    - probability: float32 `(80, 80, 80)`，当前 BOX-local ZYX 网格上的 ligand 概率；调用方已经完成 sigmoid 和模型专属后处理，Find 的 receptor hardmask 位置已经清零，unet_c1 则保留原 ligand 概率。
    - forward_payload 是当前 BOX 的 `iter_stage1_centered_batch_payloads` 结果，字段如下；P 字段逐行对齐，A 字段逐行对齐，本函数会把最终 P/A 数值压到归档 dtype。
        - ligand_probability: float32 `(80, 80, 80)`，BOX-local ZYX ligand 概率；本函数不直接读取，调用方已将其后处理结果传入 `probability`。
        - hardmask: bool `(80, 80, 80)`，BOX-local ZYX receptor home-voxel mask，True 表示 auxiliary 表中的 voxel；`L_aux` 是 True voxel 的数量。
        - voxel_aux_probability_grid: float32 `(80, 80, 80)`，与 `hardmask` 同坐标系的 auxiliary receptor 概率网格。
        - voxel_final_grid: numeric `(C_voxel, 80, 80, 80)`，channel-first 的 V 特征网格，最后三维轴序为 BOX-local ZYX；`C_voxel` 是 checkpoint 实际输出的 V 通道数。
        - P_coord_local_xyz: float32 `(N_P, 3)`，P 点的 BOX-local 连续 voxel 坐标，坐标轴序为 XYZ；`N_P` 是当前 BOX 的 P 点数。
        - P_probability: float32 `(N_P,)`，与 P 行对齐的 sigmoid 概率。
        - P_feat_L2: numeric `(N_P, C_P2)`，与 P 行对齐的 L2 特征；`C_P2` 是 P L2 特征宽度。
        - P_feat_L3: numeric `(N_P, C_P3)`，与 P 行对齐的 L3 特征；`C_P3` 是 P L3 特征宽度。
        - A_global_index: int64 `(N_A,)`，A 原子在原始 receptor 表中的全局行号；`N_A` 是当前 BOX forward payload 中的 A 原子数。
        - A_coord_local_xyz: float32 `(N_A, 3)`，A 原子的 BOX-local 连续 XYZ voxel 坐标。
        - A_coord_centered_world: float32 `(N_A, 3)`，A 原子相对 BOX 中心的世界 XYZ 坐标，单位 Å。
        - A_probability: float32 `(N_A,)`，与 A 行对齐的 sigmoid 概率。
        - A_feat_L0: float32 `(N_A, 49)`，与 A 行对齐、送入 Stage1-Find 点侧嵌入层之前的原始受体特征。
        - A_feat_L1: numeric `(N_A, C_A1)`，与 A 行对齐的 L1 特征；`C_A1` 是 A L1 特征宽度。
        - A_feat_L2: numeric `(N_A, C_A2)`，与 A 行对齐的 L2 特征；`C_A2` 是 A L2 特征宽度。
        - A_feat_L3: numeric `(N_A, C_A3)`，与 A 行对齐的 L3 特征；`C_A3` 是 A L3 特征宽度。
    - geometry 是 `CenteredGeometry`，字段如下：
        - full_shape_zyx: tuple[int, int, int]，完整图 `(D, H, W)` shape，轴序为 ZYX。
        - origin_xyz: float32 `(3,)`，完整图 voxel-grid 原点的世界 XYZ 坐标，单位 Å。
        - voxel_size_xyz: float32 `(3,)`，世界 XYZ 方向的 voxel 尺寸，单位 Å/voxel。
        - box_shape_zyx: tuple[int, int, int]，centered BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
    - stage1_model_name: str，当前 Stage-1 checkpoint 的模型身份；属于 `FIND_MODEL_NAMES` 时输出完整 P 表和筛选后的 A 表，否则只输出通用 V/aux 字段，不创建任何 P/A 键。

    输出:
    - entry: dict[str, Any]，在 base 基础上生成的可归档 centered entry；始终保留 base 的身份与几何字段，始终生成通用 V/aux 字段；Find 分支完整保留 BOX P 表，并仅保留满足 `0 <= x,y,z < box_shape_xyz` 且距至少一个来源 blob voxel center 不超过 10 Å 的 A 表；`N_A_selected` 是通过该条件保留的 A 原子数；unet_c1 分支不创建任何 P/A 字段。
        - base 继承字段：entry 直接保留 base 中的身份与几何字段。
            - centered_role: str，当前 centered 产物角色，正式值为 `F1_centered`、`CLG_centered` 或 `Selected_Refined_Centered`。
            - centered_box_index: int，同一 PDB/角色归档中的连续 entry 行号。
            - box_start_zyx: int32 `(3,)`，完整图 ZYX voxel 网格中的离散 BOX 起点。
            - box_shape_zyx: uint8 `(3,)`，当前 BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
            - box_origin_world: float32 `(3,)`，BOX 角点的世界 XYZ 坐标，单位 Å。
            - voxel_size_world: float32 `(3,)`，世界 XYZ 方向的 voxel spacing，单位 Å/voxel。
            - source_tree_id: int 标量，来源 component 所属 forest tree 编号。
            - source_node_id: int 标量，来源 component 在 forest tree 内的节点编号。
            - source_threshold_grid_index: int 标量，来源 component 的冻结阈值网格编号。
            - source_threshold_value: float32 标量，来源 component 对应的原始阈值数值。
        - V 与 auxiliary 字段：由 authority_global_linear_index、probability、forward_payload 和 geometry 共同整理，V/概率行保持权威 voxel 顺序。
            - voxel_index_local_zyx: int16 `(K_source, 3)`，把 authority_global_linear_index 按 geometry.full_shape_zyx 解码为完整图 ZYX 后减去 base.box_start_zyx 得到的 BOX-local ZYX 离散坐标；行顺序与 authority_global_linear_index 相同。
            - centered_probability: float32 `(K_source,)`，在 probability 网格按 voxel_index_local_zyx 取值的 ligand 概率；与 voxel_index_local_zyx 逐行对齐。
            - voxel_final: float16 `(K_source, C_voxel)`，从 forward_payload.voxel_final_grid 按 voxel_index_local_zyx 抽取并转置为逐 voxel 行的 V 特征；与 voxel_index_local_zyx 和 centered_probability 逐行对齐。
            - voxel_aux_index_local_zyx: int16 `(L_aux, 3)`，forward_payload.hardmask 为 True 的 BOX-local ZYX 坐标，按 np.argwhere 的 C-order 行顺序排列。
            - voxel_aux_probability: float32 `(L_aux,)`，取 forward_payload.voxel_aux_probability_grid 在 voxel_aux_index_local_zyx 处的值；与该坐标表逐行对齐。
        - Find P/A 字段：仅 Find 分支生成；P 表完整保留 BOX 内实体，A 表仅保留 core BOX 与来源 blob 10 Å 包络的交集。
            - P_coord_local_xyz: float32 `(N_P, 3)`，完整 BOX P 表中的连续 XYZ voxel 坐标。
            - P_probability: float32 `(N_P,)`，P 点 sigmoid 概率，与 P 坐标逐行对齐。
            - P_feat_L2: float16 `(N_P, C_P2)`，P 点 L2 特征，与 P 坐标逐行对齐。
            - P_feat_L3: float16 `(N_P, C_P3)`，P 点 L3 特征，与 P 坐标逐行对齐。
            - A_global_index: int64 `(N_A_selected,)`，筛选后 A 原子在原始 receptor 表中的全局行号。
            - A_coord_local_xyz: float32 `(N_A_selected, 3)`，筛选后 A 原子的 BOX-local 连续 XYZ voxel 坐标。
            - A_coord_centered_world: float32 `(N_A_selected, 3)`，筛选后 A 原子相对 BOX 中心的世界 XYZ 坐标，单位 Å。
            - A_probability: float32 `(N_A_selected,)`，筛选后 A 原子 sigmoid 概率。
            - A_feat_L0: float32 `(N_A_selected, 49)`，筛选后 A 原子的原始受体特征。
            - A_feat_L1: float16 `(N_A_selected, C_A1)`，筛选后 A 原子 L1 特征。
            - A_feat_L2: float16 `(N_A_selected, C_A2)`，筛选后 A 原子 L2 特征。
            - A_feat_L3: float16 `(N_A_selected, C_A3)`，筛选后 A 原子 L3 特征；所有 A 字段共享同一个筛选掩码并逐行对齐，顺序仍是模型 A payload 的原始顺序。
    """
    entry = dict(base)
    box_start = np.asarray(base["box_start_zyx"], dtype=np.int64)
    global_zyx = np.column_stack(
        np.unravel_index(np.asarray(authority_global_linear_index, dtype=np.int64), geometry.full_shape_zyx)
    )
    local_zyx = global_zyx - box_start[None, :]
    entry["voxel_index_local_zyx"] = local_zyx.astype(np.int16)
    entry["centered_probability"] = probability[tuple(local_zyx.T)].astype(np.float32)

    voxel_final_grid = _to_numpy(forward_payload["voxel_final_grid"])
    entry["voxel_final"] = voxel_final_grid[
        :, local_zyx[:, 0], local_zyx[:, 1], local_zyx[:, 2]
    ].T.astype(np.float16)

    hardmask = _as_numpy(forward_payload["hardmask"], np.bool_)
    aux_probability = _as_numpy(forward_payload["voxel_aux_probability_grid"], np.float32)
    aux_index = np.argwhere(hardmask).astype(np.int16)
    entry["voxel_aux_index_local_zyx"] = aux_index
    entry["voxel_aux_probability"] = aux_probability[tuple(aux_index.T)].astype(np.float32)   # 把 (L_aux, 3) 转置拆成3个索引数组, 用高级索引

    if stage1_model_name not in FIND_MODEL_NAMES:
        return entry

    p_fields = ("P_coord_local_xyz", "P_probability", "P_feat_L2", "P_feat_L3")
    p_dtypes = {
        "P_coord_local_xyz": np.float32,
        "P_probability": np.float32,
        "P_feat_L2": np.float16,
        "P_feat_L3": np.float16,
    }
    for field in p_fields:
        entry[field] = _as_numpy(forward_payload[field], p_dtypes[field])

    a_fields = (
        "A_global_index",
        "A_coord_local_xyz",
        "A_coord_centered_world",
        "A_probability",
        "A_feat_L0",
        "A_feat_L1",
        "A_feat_L2",
        "A_feat_L3",
    )
    a_dtypes = {
        "A_global_index": np.int64,
        "A_coord_local_xyz": np.float32,
        "A_coord_centered_world": np.float32,
        "A_probability": np.float32,
        "A_feat_L0": np.float32,
        "A_feat_L1": np.float16,
        "A_feat_L2": np.float16,
        "A_feat_L3": np.float16,
    }
    a_values = {field: _as_numpy(forward_payload[field], a_dtypes[field]) for field in a_fields}
    coords = a_values["A_coord_local_xyz"]
    box_shape_xyz = np.asarray(geometry.box_shape_zyx, dtype=np.float32)[[2, 1, 0]]
    core = np.all((coords >= 0.0) & (coords < box_shape_xyz), axis=1)
    envelope = _points_within_blob_envelope(
        points_local_xyz=coords,
        blob_global_linear_index=authority_global_linear_index,
        box_start_zyx=box_start,
        full_shape_zyx=geometry.full_shape_zyx,
        voxel_size_xyz=geometry.voxel_size_xyz,
        distance_angstrom=10.0,
    )
    selected = core & envelope
    for field, value in a_values.items():
        entry[field] = value[selected]
    return entry






# ================================================================================================================================================
# 主流程：按 forward → F1 → CLG → Selected → 发布的业务顺序排列。
def iter_model_centered_payloads(
    requests: Sequence[CenteredRequest],
    wrapper: Any,
    batch_builder: CenteredBatchBuilder,
    centered_batch_size: int,
) -> Iterator[dict[str, np.ndarray]]:
    """按显式 centered batch size 调用完整 wrapper，并保持请求顺序逐 BOX 返回结果。

    输入:
        - requests: Sequence[CenteredRequest]，同一 producer、split、PDB 和 role 的有序请求。
        - wrapper: 完整 Stage1 wrapper；输入是 Dataset/Collator 生成的目标设备 batch。
        - batch_builder: 接收连续请求切片，返回训练同源 Collator 生成的 dense/ragged batch。
        - centered_batch_size: int，每次 GPU forward 的 BOX 数，正式默认值为 10，尾批可以更短。

    输出:
        - payloads: Iterator[dict[str, np.ndarray]]，与 requests 严格同序；每项是一个已去除 batch 维的 BOX payload。

    Dataset、Collator 和 wrapper 依赖在这里明确出现，不通过 callback 或 output adapter 隐藏。批量拆分只改变 forward 次数，不改变 entry 顺序或字段契约。
    """
    with torch.inference_mode():
        for offset in range(0, len(requests), int(centered_batch_size)):
            request_batch = requests[offset : offset + int(centered_batch_size)]
            batch = batch_builder(request_batch)
            forward_output = wrapper(batch)
            yield from iter_stage1_centered_batch_payloads(
                forward_output=forward_output,
                batch=batch,
                stage1_model_name=request_batch[0].stage1_model_name,
            )


def produce_f1_centered_entries(
    nodes: Sequence[ComponentNode],
    f1_threshold_grid_index: int,
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    wrapper: Any,
    batch_builder: CenteredBatchBuilder,
    centered_batch_size: int,
) -> list[dict[str, Any]]:
    """为 t_F1 层 eligible component 生成 F1_centered entries。
    处理顺序是筛选并排序来源 component、构造 CenteredRequest、按显式 batch size 运行完整 forward、对每个权威 voxel 抽取 V 概率/特征，并在 Find 模型下追加完整  P 表和 core BOX 与来源 blob 10 Å 包络交集内的 A 表。

    每个返回 entry 的字段(详见 def _payload_from_forward ):
        - centered_role、centered_box_index：固定 role 和连续 entry 编号。
        - box_start_zyx、box_shape_zyx、box_origin_world、voxel_size_world：BOX 离散/物理几何。
        - source_tree_id、source_node_id、source_threshold_grid_index、source_threshold_value：来源身份。
        - voxel_index_local_zyx、centered_probability、voxel_final：权威 voxel 的坐标、概率和 V 特征。
        - voxel_aux_index_local_zyx、voxel_aux_probability：hardmask auxiliary 表。
        - Find 额外包含 P_coord_local_xyz、P_probability、P_feat_L2、P_feat_L3，以及筛选后的 A_global_index、A_coord_local_xyz、A_coord_centered_world、A_probability、A_feat_L0、A_feat_L1、A_feat_L2、A_feat_L3。
        - unet_c1 不返回 P/A 字段，局部 forward 的其他 component 不进入当前 entry。
    """
    sources = [
        node
        for node in nodes
        if node.candidate_eligible
        and node.threshold_grid_index == int(f1_threshold_grid_index)
    ]
    sources.sort(key=lambda node: (-node.probability_mean, node.tree_id, node.node_id))

    bases: list[dict[str, Any]] = []
    requests: list[CenteredRequest] = []
    for centered_box_index, source in enumerate(sources):
        base = _base_entry(
            source,
            "F1_centered",
            centered_box_index,
            np.asarray(resolve_box_start(source.centroid_zyx, geometry.full_shape_zyx), dtype=np.int64),
            geometry,
        )
        bases.append(base)
        requests.append(
            CenteredRequest(
                stage1_model_name=str(stage1_model_name),
                split=str(split),
                pdb_id=str(pdb_id),
                centered_role="F1_centered",
                centered_box_index=int(base["centered_box_index"]),
                box_start_zyx=tuple(int(value) for value in base["box_start_zyx"]),
                source_tree_id=int(base["source_tree_id"]),
                source_node_id=int(base["source_node_id"]),
                source_threshold_grid_index=int(base["source_threshold_grid_index"]),
            )
        )

    entries: list[dict[str, Any]] = []
    for source, base, forward_payload in zip(
        sources,
        bases,
        iter_model_centered_payloads(requests, wrapper, batch_builder, centered_batch_size),
    ):
        probability = postprocess_ligand_probability(
            _as_numpy(forward_payload["ligand_probability"], np.float32),
            stage1_model_name,
            forward_payload.get("hardmask"),
        )
        entries.append(
            _payload_from_forward(
                base,
                source.voxel_global_linear_index,
                probability,
                forward_payload,
                geometry,
                stage1_model_name,
            )
        )
    return entries


def produce_clg_centered_entries(
    clgs: Sequence[CLG],
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    wrapper: Any,
    batch_builder: CenteredBatchBuilder,
    centered_batch_size: int,
) -> list[dict[str, Any]]:
    """以每个 CLG 的 oldest component 为权威 voxel 集生成 CLG_centered entries。

    输入参数:
    - clgs: Sequence[CLG]，按 CLG 正式来源顺序排列的成功 Candidate Lineage Group；每个 CLG 的 oldest_node 提供当前 entry 的权威 voxel 集，candidate_nodes 提供候选节点的稳定顺序和 membership 对象。
        - CLG_id: int，当前 PDB 内按正式枚举顺序分配的 CLG 本地编号。
        - tree_id: int，CLG 全部候选所属的 component tree 编号。
        - seed_node: ComponentNode，本次工作树扫描选中的 active seed，必须属于 candidate_nodes。
        - oldest_node: ComponentNode，覆盖全部 candidate voxel mask 的最低阈值候选；其 voxel_global_linear_index 是当前 entry 的权威 voxel 顺序。
        - candidate_nodes: Sequence[ComponentNode]，同一 tree 内按确定性枚举顺序排列的候选节点；长度记为 N_candidate。
    - geometry: CenteredGeometry，提供完整图和 centered BOX 的离散/物理几何。
        - full_shape_zyx: tuple[int, int, int]，完整图 `(D, H, W)` shape，轴序为 ZYX。
        - origin_xyz: float32 `(3,)`，完整图 voxel-grid 原点的世界 XYZ 坐标，单位 Å。
        - voxel_size_xyz: float32 `(3,)`，世界 XYZ 方向的 voxel 尺寸，单位 Å/voxel。
        - box_shape_zyx: tuple[int, int, int]，centered BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
    - stage1_model_name: str，当前 Stage-1 checkpoint 的模型身份；属于 `FIND_MODEL_NAMES` 时生成 P/A 表和 candidate_A_membership，否则不创建这些字段。
    - split: str，当前数据划分，写入每个 CenteredRequest；正式值为 calibration、validation 或 train。
    - pdb_id: str，当前完整图和 receptor 表共用的小写 PDB identity，写入每个 CenteredRequest。
    - resolve_box_start: StartResolver，输入 source.centroid_zyx（float32 `(3,)`，完整图连续 ZYX voxel-index 质心）和 geometry.full_shape_zyx，返回当前 80³ BOX 的 int ZYX 起点。
    - wrapper: Any，接收 batch_builder 生成的设备 batch 并返回完整 Stage-1 forward output 的模型 wrapper；本函数不改变 wrapper 的输出字段。
    - batch_builder: CenteredBatchBuilder，输入当前批次的 CenteredRequest 序列，返回训练同源 Collator 使用的模型 batch；批次顺序必须与 requests 同序。
    - centered_batch_size: int，单次模型 forward 的 BOX 数；显式控制 GPU batch，不能改变 CLG/entry 的逻辑顺序。

    输出:
    - entries: list[dict[str, Any]]，与 clgs 严格同序、长度相同的 CLG_centered entries；每个 entry 分为三组字段。
        - 基础身份与几何字段：
            - centered_role: str，固定为 `CLG_centered`。
            - centered_box_index: int，同一 PDB/角色归档中的连续 entry 行号。
            - box_start_zyx: int32 `(3,)`，完整图 ZYX voxel 网格中的离散 BOX 起点。
            - box_shape_zyx: uint8 `(3,)`，当前 BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
            - box_origin_world: float32 `(3,)`，BOX 角点的世界 XYZ 坐标，单位 Å。
            - voxel_size_world: float32 `(3,)`，世界 XYZ 方向的 voxel spacing，单位 Å/voxel。
            - source_tree_id: int 标量，oldest_node 所属 forest tree 编号。
            - source_node_id: int 标量，oldest_node 在 forest tree 内的节点编号。
            - source_threshold_grid_index: int 标量，oldest_node 的冻结阈值网格编号。
            - source_threshold_value: float32 标量，oldest_node 的阈值数值。
        - V/aux/P/A 模型字段：
            - voxel_index_local_zyx: int16 `(K_source, 3)`，oldest_node 权威 voxel 的 BOX-local ZYX 坐标；K_source 是 oldest_node.voxel_global_linear_index 的长度。
            - centered_probability: float32 `(K_source,)`，权威 voxel 对应的 ligand 概率，与 voxel_index_local_zyx 逐行对齐。
            - voxel_final: float16 `(K_source, C_voxel)`，权威 voxel 的逐 voxel V 特征，与 voxel_index_local_zyx 和 centered_probability 逐行对齐；C_voxel 是 checkpoint 实际输出的 V 通道数。
            - voxel_aux_index_local_zyx: int16 `(L_aux, 3)`，hardmask True 的 BOX-local ZYX 坐标，按 C-order 排列；L_aux 是 True voxel 数。
            - voxel_aux_probability: float32 `(L_aux,)`，与 voxel_aux_index_local_zyx 逐行对齐的 auxiliary 概率。
            - P_coord_local_xyz: float32 `(N_P, 3)`，Find 分支完整 BOX P 表的连续 XYZ voxel 坐标；N_P 是当前 BOX 的 P 点数。
            - P_probability: float32 `(N_P,)`，与 P 坐标逐行对齐的 sigmoid 概率。
            - P_feat_L2: float16 `(N_P, C_P2)`，与 P 坐标逐行对齐的 P L2 特征。
            - P_feat_L3: float16 `(N_P, C_P3)`，与 P 坐标逐行对齐的 P L3 特征。
            - A_global_index: int64 `(N_A_selected,)`，Find 分支筛选后 A 原子在原始 receptor 表中的全局行号；N_A_selected 是筛选后 A 原子数。
            - A_coord_local_xyz: float32 `(N_A_selected, 3)`，Find 分支筛选后 A 原子的 BOX-local 连续 XYZ voxel 坐标。
            - A_coord_centered_world: float32 `(N_A_selected, 3)`，Find 分支筛选后 A 原子相对 BOX 中心的世界 XYZ 坐标，单位 Å。
            - A_probability: float32 `(N_A_selected,)`，Find 分支筛选后 A 原子 sigmoid 概率。
            - A_feat_L0: float32 `(N_A_selected, 49)`，Find 分支筛选后 A 原子的原始受体特征。
            - A_feat_L1: float16 `(N_A_selected, C_A1)`，Find 分支筛选后 A 原子 L1 特征。
            - A_feat_L2: float16 `(N_A_selected, C_A2)`，Find 分支筛选后 A 原子 L2 特征。
            - A_feat_L3: float16 `(N_A_selected, C_A3)`，Find 分支筛选后 A 原子 L3 特征；所有 A 字段按同一 core BOX 与来源 blob 10 Å 包络掩码逐行对齐。unet_c1 不创建任何 P/A 字段，P 表也不建立 candidate P membership。
        - CLG 谱系字段：
            - CLG_id: int，当前 CLG 身份编号。
            - CLG_seed_node_id: int，当前 CLG seed 节点编号。
            - CLG_oldest_node_id: int，当前 CLG oldest 节点编号。
            - candidate_node_id: int32 `(N_candidate,)`，按 candidate_nodes 顺序排列的候选 node_id。
            - candidate_threshold_grid_index: int32 `(N_candidate,)`，按 candidate_nodes 顺序排列的候选阈值网格编号。
            - candidate_voxel_membership: list[np.ndarray]，长度为 N_candidate；第 i 项为 int32 `(K_candidate_i,)`，数值是第 i 个候选 voxel 在 oldest 权威 voxel 表第一维中的局部行号。
            - candidate_A_membership: Find 分支才存在的 list[np.ndarray]，长度为 N_candidate；第 i 项为 int32 `(L_A_i,)`，数值是第 i 个候选 blob 包络内 A 点在当前 A 表第一维中的局部行号；unet_c1 不包含该字段。

    candidate_voxel_membership 使用每个候选节点的完整图 voxel 线性索引映射到 oldest 权威 voxel 行号；candidate_A_membership 使用当前 entry 的 A 坐标判断候选 blob 的 10 Å 包络，均不改变候选节点原始顺序。
    """
    bases: list[dict[str, Any]] = []
    requests: list[CenteredRequest] = []
    for centered_box_index, clg in enumerate(clgs):
        source = clg.oldest_node
        base = _base_entry(
            source,
            "CLG_centered",
            centered_box_index,
            np.asarray(resolve_box_start(source.centroid_zyx, geometry.full_shape_zyx), dtype=np.int64),
            geometry,
        )
        bases.append(base)
        requests.append(
            CenteredRequest(
                stage1_model_name=str(stage1_model_name),
                split=str(split),
                pdb_id=str(pdb_id),
                centered_role="CLG_centered",
                centered_box_index=int(base["centered_box_index"]),
                box_start_zyx=tuple(int(value) for value in base["box_start_zyx"]),
                source_tree_id=int(base["source_tree_id"]),
                source_node_id=int(base["source_node_id"]),
                source_threshold_grid_index=int(base["source_threshold_grid_index"]),
            )
        )

    entries: list[dict[str, Any]] = []
    for clg, base, forward_payload in zip(
        clgs,
        bases,
        iter_model_centered_payloads(requests, wrapper, batch_builder, centered_batch_size),
    ):
        source = clg.oldest_node
        probability = postprocess_ligand_probability(
            _as_numpy(forward_payload["ligand_probability"], np.float32),
            stage1_model_name,
            forward_payload.get("hardmask"),
        )
        entry = _payload_from_forward(
            base,
            source.voxel_global_linear_index,
            probability,
            forward_payload,
            geometry,
            stage1_model_name,
        )
        entry.update(
            {
                "CLG_id": int(clg.CLG_id),
                "CLG_seed_node_id": int(clg.seed_node.node_id),
                "CLG_oldest_node_id": int(clg.oldest_node.node_id),
                "candidate_node_id": np.asarray([node.node_id for node in clg.candidate_nodes], dtype=np.int32),
                "candidate_threshold_grid_index": np.asarray([node.threshold_grid_index for node in clg.candidate_nodes], dtype=np.int32),
            }
        )
        oldest_global = np.asarray(source.voxel_global_linear_index, dtype=np.int64)
        entry["candidate_voxel_membership"] = [
            np.searchsorted(
                oldest_global,
                np.asarray(node.voxel_global_linear_index, dtype=np.int64),
            ).astype(np.int32)
            for node in clg.candidate_nodes
        ]
        if stage1_model_name in FIND_MODEL_NAMES:
            a_coords = np.asarray(entry["A_coord_local_xyz"], dtype=np.float32)
            entry["candidate_A_membership"] = [
                np.flatnonzero(
                    _points_within_blob_envelope(
                        points_local_xyz=a_coords,
                        blob_global_linear_index=node.voxel_global_linear_index,
                        box_start_zyx=np.asarray(entry["box_start_zyx"], dtype=np.int64),
                        full_shape_zyx=geometry.full_shape_zyx,
                        voxel_size_xyz=geometry.voxel_size_xyz,
                        distance_angstrom=10.0,
                    )
                ).astype(np.int32)
                for node in clg.candidate_nodes
            ]
        entries.append(entry)
    return entries


def produce_selected_refined_entries(
    selected_nodes: Sequence[ComponentNode],
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    wrapper: Any,
    batch_builder: CenteredBatchBuilder,
    centered_batch_size: int,
) -> list[dict[str, Any]]:
    """重跑每个 Selector 来源并按 IoU 最大局部组件生成 Selected refined entries。

    输入参数:
    - selected_nodes: Sequence[ComponentNode]，按 Selector 选中来源顺序排列的 component node；每个 node 的完整图 voxel 集作为 source mask，threshold_value 决定当前 BOX 的二值化阈值。
        - tree_id: int，来源 component 所属 forest tree 编号。
        - node_id: int，来源 component 在 forest tree 内的稳定节点编号。
        - threshold_grid_index: int，来源 component 的冻结阈值网格编号。
        - threshold_value: float，当前来源 component 的概率阈值，直接用于 `probability >= threshold_value`。
        - voxel_global_linear_index: int64 `(K_source,)`，来源 component 在完整图 ZYX 网格中的 C-order 全局线性索引；K_source 是 source mask 的 voxel 数。
        - centroid_zyx: float32 `(3,)`，来源 component voxel index 的连续 ZYX 质心，交给 resolve_box_start 计算 BOX 起点。
    - geometry: CenteredGeometry，提供完整图和 centered BOX 的离散/物理几何。
        - full_shape_zyx: tuple[int, int, int]，完整图 `(D, H, W)` shape，轴序为 ZYX。
        - origin_xyz: float32 `(3,)`，完整图 voxel-grid 原点的世界 XYZ 坐标，单位 Å。
        - voxel_size_xyz: float32 `(3,)`，世界 XYZ 方向的 voxel 尺寸，单位 Å/voxel。
        - box_shape_zyx: tuple[int, int, int]，centered BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
    - stage1_model_name: str，当前 Stage-1 checkpoint 的模型身份；属于 `FIND_MODEL_NAMES` 时 success entry 生成 P/A 表，失败 entry 在存在 success 模板时同步生成 P/A 空表。
    - split: str，当前数据划分，写入每个 CenteredRequest；正式值为 calibration、validation 或 train。
    - pdb_id: str，当前完整图和 receptor 表共用的小写 PDB identity，写入每个 CenteredRequest。
    - resolve_box_start: StartResolver，输入 source.centroid_zyx 和 geometry.full_shape_zyx，返回当前 80³ BOX 的 int ZYX 起点。
    - wrapper: Any，接收 batch_builder 生成的设备 batch 并返回完整 Stage-1 forward output 的模型 wrapper；本函数不改变 wrapper 的输出字段。
    - batch_builder: CenteredBatchBuilder，输入当前批次的 CenteredRequest 序列，返回训练同源 Collator 使用的模型 batch；批次顺序必须与 requests 同序。
    - centered_batch_size: int，单次模型 forward 的 BOX 数；显式控制 GPU batch，不改变 selected_nodes/entry 的逻辑顺序。

    输出:
    - entries: list[dict[str, Any]]，与 selected_nodes 严格同序、长度相同的 Selected_Refined_Centered entries；每个 entry 分为三组字段。
        - 基础身份与几何字段：
            - centered_role: str，固定为 `Selected_Refined_Centered`。
            - centered_box_index: int，同一 PDB/角色归档中的连续 entry 行号。
            - box_start_zyx: int32 `(3,)`，完整图 ZYX voxel 网格中的离散 BOX 起点。
            - box_shape_zyx: uint8 `(3,)`，当前 BOX 的 ZYX shape，正式值为 `(80, 80, 80)`。
            - box_origin_world: float32 `(3,)`，BOX 角点的世界 XYZ 坐标，单位 Å。
            - voxel_size_world: float32 `(3,)`，世界 XYZ 方向的 voxel spacing，单位 Å/voxel。
            - source_tree_id: int 标量，selected source 所属 forest tree 编号。
            - source_node_id: int 标量，selected source 在 forest tree 内的节点编号。
            - source_threshold_grid_index: int 标量，selected source 的冻结阈值网格编号。
            - source_threshold_value: float32 标量，selected source 的阈值数值。
        - refine 状态与 voxel/aux 字段：
            - refine_status: str，正式值为 `success`、`empty` 或 `no_overlap`；`success` 表示存在与 source mask 相交的局部预测组件并选择 IoU 最大者，`empty` 表示阈值二值化后没有 26-连通组件，`no_overlap` 表示存在局部组件但全部不与 source mask 相交。
            - voxel_index_local_zyx: success 时为 int16 `(K_refined, 3)`，选中局部组件的 BOX-local ZYX 坐标，按 `np.argwhere(labels == selected_label)` 的 C-order 行顺序排列；`K_refined` 是选中局部组件的 voxel 数；empty/no_overlap 时为 int16 `(0, 3)`。
            - centered_probability: success 时为 float32 `(K_refined,)`，在 postprocess 后 probability 网格按 refined voxel 坐标取出的 ligand 概率；与 voxel_index_local_zyx 逐行对齐；empty/no_overlap 时为 float32 `(0,)`。
            - voxel_final: success 时为 float16 `(K_refined, C_voxel)`，与 refined voxel 逐行对齐的 V 特征；`C_voxel` 是成功 entry 的 checkpoint V 通道数；empty/no_overlap 初始为 float16 `(0, 0)`，同一 PDB 存在 success entry 后统一改为 `(0, C_voxel)`，不凭空猜测通道宽度。
            - voxel_aux_index_local_zyx: int16 `(L_aux, 3)`，当前 BOX hardmask True 的 auxiliary voxel 坐标；`L_aux` 是 True voxel 数；失败 entry 也保留空表 `(0, 3)`。
            - voxel_aux_probability: float32 `(L_aux,)`，与 voxel_aux_index_local_zyx 逐行对齐的 auxiliary 概率；失败 entry 为 `(0,)`。
        - Find P/A 字段：仅 Find 模型的 success entry 直接携带完整 P 表和 core BOX 与 source blob 10 Å 包络交集内的 A 表；若同一 PDB 至少有一个 success entry，失败 entry 会按照 success entry 的实际特征宽度生成对应 P/A 空表；若全部 entry 失败，则不创建未知宽度的 P/A 字段，unet_c1 始终不创建 P/A 字段。
            - P_coord_local_xyz: float32 `(N_P, 3)`，success entry 的完整 BOX P 点连续 XYZ voxel 坐标；`N_P` 是 success entry 的 P 点数；失败 entry 在有 success 模板时为 `(0, 3)`。
            - P_probability: float32 `(N_P,)`，与 P 坐标逐行对齐的 sigmoid 概率；失败 entry 在有 success 模板时为 `(0,)`。
            - P_feat_L2: float16 `(N_P, C_P2)`，与 P 坐标逐行对齐的 P L2 特征；失败 entry 在有 success 模板时为 `(0, C_P2)`。
            - P_feat_L3: float16 `(N_P, C_P3)`，与 P 坐标逐行对齐的 P L3 特征；失败 entry 在有 success 模板时为 `(0, C_P3)`。
            - A_global_index: int64 `(N_A_selected,)`，success entry 筛选后 A 原子在原始 receptor 表中的全局行号；`N_A_selected` 是 success entry 筛选后 A 原子数；失败 entry 在有 success 模板时为 `(0,)`。
            - A_coord_local_xyz: float32 `(N_A_selected, 3)`，success entry 筛选后 A 原子的 BOX-local 连续 XYZ voxel 坐标；失败 entry 在有 success 模板时为 `(0, 3)`。
            - A_coord_centered_world: float32 `(N_A_selected, 3)`，success entry 筛选后 A 原子相对 BOX 中心的世界 XYZ 坐标，单位 Å；失败 entry 在有 success 模板时为 `(0, 3)`。
            - A_probability: float32 `(N_A_selected,)`，success entry 筛选后 A 原子 sigmoid 概率；失败 entry 在有 success 模板时为 `(0,)`。
            - A_feat_L0: float32 `(N_A_selected, 49)`，success entry 筛选后 A 原子的原始受体特征；失败 entry 在有 success 模板时为 `(0, 49)`。
            - A_feat_L1: float16 `(N_A_selected, C_A1)`，success entry 筛选后 A 原子 L1 特征；失败 entry 在有 success 模板时为 `(0, C_A1)`。
            - A_feat_L2: float16 `(N_A_selected, C_A2)`，success entry 筛选后 A 原子 L2 特征；失败 entry 在有 success 模板时为 `(0, C_A2)`。
            - A_feat_L3: float16 `(N_A_selected, C_A3)`，success entry 筛选后 A 原子 L3 特征；失败 entry 在有 success 模板时为 `(0, C_A3)`；所有 A 字段沿用同一筛选掩码并逐行对齐。

    success 的局部组件通过 26-connectivity 标记；对每个组件计算其与 source mask 的交集和并集，IoU 为 intersection / union，使用最大 IoU 的组件，平局时保留 `np.argmax` 的首个组件。success 的 refined_global voxel 顺序由选中标签的 `np.argwhere` 结果按 C-order 展平回完整图索引；empty/no_overlap 不伪造未知的 P/A 或 V 通道宽度。
    """

    # list[dict[str, Any]] / list[CenteredRequest]，长度均为 N_selected；两表按 selected_nodes 顺序一一对齐，后续 zip 不改变该顺序。
    bases: list[dict[str, Any]] = []
    requests: list[CenteredRequest] = []
    for centered_box_index, source in enumerate(selected_nodes):
        # source.centroid_zyx 为完整图连续 ZYX voxel-index 质心；resolver 返回当前 80³ BOX 的离散 ZYX 起点。
        base = _base_entry(
            source,
            "Selected_Refined_Centered",
            centered_box_index,
            np.asarray(resolve_box_start(source.centroid_zyx, geometry.full_shape_zyx), dtype=np.int64),
            geometry,
        )
        bases.append(base)
        # CenteredRequest 携带 Dataset/Collator 所需的身份和 BOX 几何；centered_box_index 与 base 完全一致。
        requests.append(
            CenteredRequest(
                stage1_model_name=str(stage1_model_name),
                split=str(split),
                pdb_id=str(pdb_id),
                centered_role="Selected_Refined_Centered",
                centered_box_index=int(base["centered_box_index"]),
                box_start_zyx=tuple(int(value) for value in base["box_start_zyx"]),
                source_tree_id=int(base["source_tree_id"]),
                source_node_id=int(base["source_node_id"]),
                source_threshold_grid_index=int(base["source_threshold_grid_index"]),
            )
        )

    # list[dict[str, Any]]，长度最终等于 N_selected；每个元素对应一个 source，可能是 success、empty 或 no_overlap。
    entries: list[dict[str, Any]] = []
    for source, base, forward_payload in zip(
        selected_nodes,
        bases,
        iter_model_centered_payloads(requests, wrapper, batch_builder, centered_batch_size),
    ):
        # float32，(80, 80, 80)，BOX-local ZYX ligand 概率；Find 分支在 receptor hardmask 位置清零，unet_c1 保留原概率。
        probability = postprocess_ligand_probability(
            _as_numpy(forward_payload["ligand_probability"], np.float32),
            stage1_model_name,
            forward_payload.get("hardmask"),
        )
        # bool，(80, 80, 80)，按当前 source.threshold_value 二值化 ligand 概率；True voxel 进入局部 26-连通分量标记。
        binary = probability >= float(source.threshold_value)
        # labels 为 BOX-local ZYX 整数标签网格；label 0 是背景，component_count 是非背景 26-连通组件数。
        labels, component_count = ndimage.label(
            binary,
            structure=ndimage.generate_binary_structure(3, 3),
        )
        if int(component_count) == 0:
            # 当前 source 没有任何局部预测组件；保留 base 身份，V/aux 使用明确的空表，不伪造 P/A 或 V 通道宽度。
            entry = dict(base)
            entry.update(
                {
                    "refine_status": "empty",
                    "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int16),
                    "centered_probability": np.empty(0, dtype=np.float32),
                    "voxel_final": np.empty((0, 0), dtype=np.float16),
                    "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
                    "voxel_aux_probability": np.empty(0, dtype=np.float32),
                }
            )
            entries.append(entry)
            continue

        # int64，(K_source, 3)，把 source 的完整图 C-order 线性索引解码为 ZYX，再减去 BOX 起点得到局部坐标。
        box_start = np.asarray(base["box_start_zyx"], dtype=np.int64)
        source_global = np.asarray(source.voxel_global_linear_index, dtype=np.int64)   # source节点的完整图 C-order 线性索引
        source_local = np.column_stack(np.unravel_index(source_global, geometry.full_shape_zyx)) - box_start[None, :]  # source节点的在box内的 C-order 线性索引
        # bool，(80, 80, 80)，source mask 的唯一权威成员集合；它与 labels 使用同一个 BOX-local ZYX 坐标系。
        source_mask = np.zeros(geometry.box_shape_zyx, dtype=np.bool_)
        source_mask[tuple(source_local.T)] = True
        # int64，(N_component,)，逐个统计 label 1..N_component 与 source_mask 的 voxel 交集；背景 label 0 不参与 IoU。
        intersection_counts = np.asarray(
            [
                np.logical_and(labels == label_id, source_mask).sum()
                for label_id in range(1, int(component_count) + 1)
            ],
            dtype=np.int64,
        )
        if int(intersection_counts.max(initial=0)) == 0:
            # 所有局部组件都与 source mask 不相交；该 entry 不进入 forward payload 抽取，直接发布 no_overlap 空表。
            entry = dict(base)
            entry.update(
                {
                    "refine_status": "no_overlap",
                    "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int16),
                    "centered_probability": np.empty(0, dtype=np.float32),
                    "voxel_final": np.empty((0, 0), dtype=np.float16),
                    "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
                    "voxel_aux_probability": np.empty(0, dtype=np.float32),
                }
            )
            entries.append(entry)
            continue

        # int64，(N_component,)，排除背景后的各局部组件 voxel 数；与 intersection_counts 逐元素对齐。
        component_sizes = np.bincount( labels.reshape(-1), minlength=int(component_count) + 1)[1:]
        # int64，(N_component,)，每个组件与 source mask 的并集大小；union = component_size + source_size - intersection。
        unions = component_sizes + int(source_mask.sum()) - intersection_counts
        # float64，(N_component,)，按组件顺序保存 IoU；np.argmax 平局时返回最先出现的最大值。
        iou = intersection_counts / unions.astype(np.float64)
        selected_label = int(np.argmax(iou)) + 1
        # int16，(K_refined, 3)，选中组件的 BOX-local ZYX 坐标；np.argwhere 按 C-order 返回，定义 success voxel 顺序。
        refined_local = np.argwhere(labels == selected_label).astype(np.int16)
        # int64，(K_refined,)，把局部 ZYX 坐标加回完整图起点后按完整图 shape 展平，恢复归档使用的全局线性索引。
        refined_global = np.ravel_multi_index(
            (refined_local.astype(np.int64) + box_start[None, :]).T,
            geometry.full_shape_zyx,
        ).astype(np.int64)
        # 以 refined_global 作为 success entry 的权威 voxel 集，抽取 V/概率/aux，并按 Find 契约筛选 P/A。
        entry = _payload_from_forward(
            base,
            refined_global,
            probability,
            forward_payload,
            geometry,
            stage1_model_name,
        )
        entry["refine_status"] = "success"
        entries.append(entry)

    # 从本批 entries 中寻找第一个 success，作为所有失败 entry 空变长字段的 trailing-shape 模板。
    successful = next(
        (entry for entry in entries if entry["refine_status"] == "success"),
        None,
    )
    if successful is not None:
        # voxel_template 第一维是某个成功组件的 K_refined，后续只复用其特征宽度 C_voxel。
        voxel_template = np.asarray(successful["voxel_final"])
        # 这些字段只在 success entry 实际存在时为失败 entry 建立空表；字段名和 trailing shape 与 success 完全一致。
        p_fields = ("P_coord_local_xyz", "P_probability", "P_feat_L2", "P_feat_L3")
        a_fields = (
            "A_global_index",
            "A_coord_local_xyz",
            "A_coord_centered_world",
            "A_probability",
            "A_feat_L0",
            "A_feat_L1",
            "A_feat_L2",
            "A_feat_L3",
        )
        for entry in entries:
            if entry["refine_status"] == "success":
                continue
            # 失败 entry 的 voxel_final 保留 success 的 C_voxel 宽度，第一维置 0，便于 archive packer 聚合。
            entry["voxel_final"] = np.empty(
                (0, *voxel_template.shape[1:]),
                dtype=voxel_template.dtype,
            )
            for field in (*p_fields, *a_fields):
                if field not in successful:
                    continue
                # 按 success 字段的 dtype 和所有非行维度创建空表；第一维 0 与同 entry 的 offsets 对齐。
                template = np.asarray(successful[field])
                entry[field] = np.empty((0, *template.shape[1:]), dtype=template.dtype)
    return entries


def publish_centered_entries(
    paths: Stage1ArtifactPaths,
    centered_role: str,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """聚合、校验并原子发布一个 PDB/role 的 centered NPZ，再写 _COMPLETE。

    输入字段:

    - paths：当前 model/split/PDB 的正式 artifact 路径，并提供 archive validator 所需的 model name。
    - centered_role：F1_centered、CLG_centered 或 Selected_Refined_Centered。
    - entries：同一 PDB/role 的有序 entry；entry 第一维决定归档中的 centered_box_index 和 offsets 切分。

    发布顺序固定为 pack、NPZ 原子写入、读取校验、最后写 role _COMPLETE。
    """
    arrays = pack_centered_entries(
        entries,
        centered_role,
        stage1_model_name=paths.stage1_model_name,
    )
    atomic_savez_compressed(
        paths.centered_npz(centered_role),
        arrays,
        validator=lambda value: validate_centered_archive(
            value,
            centered_role,
            stage1_model_name=paths.stage1_model_name,
        ),
    )
    mark_role_complete(paths, centered_role)


__all__ = [
    "CenteredBatchBuilder",
    "CenteredGeometry",
    "CenteredRequest",
    "iter_model_centered_payloads",
    "iter_stage1_centered_batch_payloads",
    "produce_clg_centered_entries",
    "produce_f1_centered_entries",
    "produce_selected_refined_entries",
    "publish_centered_entries",
]
