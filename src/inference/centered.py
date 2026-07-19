"""F1、CLG 与 Selected 三类居中 BOX 的 callback 驱动生产。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.artifacts.io import (
    FIXED_VOXEL_GRID_FIELDS,
    atomic_savez_compressed,
    pack_centered_entries,
    validate_centered_archive,
)
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import mark_role_complete
from src.component_lineage.structures import CLG, ComponentNode
from src.datasets.stage1_requests import centered_start_from_centroid_zyx

from .probability import (
    FIND_MODEL_NAMES,
    logits_to_probability,
    postprocess_ligand_probability,
)


@dataclass(frozen=True)
class CenteredGeometry:
    """
    描述一张完整图及其固定 80³ 居中 BOX 的物理几何。

    输入参数:
        - full_shape_zyx: tuple[int,int,int], 完整图数组 shape
        - origin_xyz: np.ndarray, (3,), float32，完整网格物理下角点
        - voxel_size_xyz: np.ndarray, (3,), float32，XYZ Å/voxel
        - box_shape_zyx: tuple[int,int,int], 正式值为 (80,80,80)
    """

    full_shape_zyx: tuple[int, int, int]
    origin_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    box_shape_zyx: tuple[int, int, int] = (80, 80, 80)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "full_shape_zyx", tuple(int(value) for value in self.full_shape_zyx)
        )
        object.__setattr__(
            self, "box_shape_zyx", tuple(int(value) for value in self.box_shape_zyx)
        )
        object.__setattr__(self, "origin_xyz", np.asarray(self.origin_xyz, dtype=np.float32))
        object.__setattr__(
            self, "voxel_size_xyz", np.asarray(self.voxel_size_xyz, dtype=np.float32)
        )
        if len(self.full_shape_zyx) != 3 or len(self.box_shape_zyx) != 3:
            raise ValueError("full_shape_zyx 与 box_shape_zyx 必须是长度 3 的 ZYX")
        if self.origin_xyz.shape != (3,) or self.voxel_size_xyz.shape != (3,):
            raise ValueError("origin_xyz 与 voxel_size_xyz 必须是长度 3 的 XYZ")
        if bool(np.any(self.voxel_size_xyz <= 0)):
            raise ValueError("voxel_size_xyz 必须为正")


@dataclass(frozen=True)
class CenteredRequest:
    """
    传给统一 Dataset/materializer 和完整 forward callback 的居中请求。

    输入参数:
        - stage1_model_name: str, 当前 producer 正式名
        - split: str, 当前数据划分
        - pdb_id: str, 当前完整图身份
        - centered_role: str, 三类正式 centered role 之一
        - centered_box_index: int, 当前聚合文件内 entry 顺序
        - box_start_zyx: tuple[int,int,int], 已由统一 resolver 解析的合法起点
        - source_tree_id: int, 来源 component tree
        - source_node_id: int, F1 source、CLG oldest 或 Selected source node
        - source_threshold_grid_index: int, 来源整数阈值 j
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


FullForwardCallback = Callable[[CenteredRequest], Mapping[str, Any]]
StartResolver = Callable[[np.ndarray, tuple[int, int, int]], Sequence[int]]


def resolve_component_centered_start(
    centroid_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    把组件 voxel-index centroid 转成与训练请求相同的唯一 80³ 起点。

    输入参数:
        - centroid_zyx: np.ndarray, (3,), 组件 mask 的 voxel-index 均值
        - full_shape_zyx: tuple[int,int,int], 当前完整图 shape

    输出:
        - resolved_start_zyx: tuple[int,int,int], 由训练、forest 与 centered 共用的
          `centered_start_from_centroid_zyx` 统一完成 `+0.5`、取整与边界 clamp
    """
    return centered_start_from_centroid_zyx(
        centroid_zyx=centroid_zyx,
        full_shape_zyx=full_shape_zyx,
    )


def make_model_centered_callback(
    wrapper: Any,
    batch_builder: Callable[[CenteredRequest], Mapping[str, Any]],
    output_adapter: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> FullForwardCallback:
    """
    把正式完整 wrapper、统一 batch builder 与具名 feature adapter 组装为 callback。

    输入参数:
        - wrapper: Any, 已恢复并 eval 的完整 Stage1 wrapper；居中推理走普通 `forward`
        - batch_builder: Callable, 从 CenteredRequest 构造统一 Dataset/Collator batch
        - output_adapter: Callable | None，可选自定义 `(forward_output,batch)` 适配器；
          为 None 时使用正式 `adapt_stage1_centered_output`，读取模型的具名 V/P/A 出口

    输出:
        - callback: FullForwardCallback, 可注入三类 centered 生产函数
    """
    def callback(request: CenteredRequest) -> Mapping[str, Any]:
        batch = batch_builder(request)
        try:
            import torch

            with torch.no_grad():
                output = wrapper(batch)
        except ImportError:
            output = wrapper(batch)
        if output_adapter is not None:
            return output_adapter(output, batch)
        return adapt_stage1_centered_output(
            forward_output=output,
            batch=batch,
            stage1_model_name=request.stage1_model_name,
        )

    return callback


def adapt_stage1_centered_output(
    forward_output: Mapping[str, Any],
    batch: Mapping[str, Any],
    stage1_model_name: str,
) -> dict[str, np.ndarray]:
    """
    把单个 80³ BOX 的正式完整 forward 转为 centered 归档使用的具名 payload。

    输入参数:
        - forward_output: Mapping[str,Any]，Stage1 wrapper 的完整 forward 输出；必须含
          ligand/aux logits 和五张具名 voxel features，Find 还含 A_feat_L1..L4、
          P_feat_L2..L4、A/P logits 与身份坐标
        - batch: Mapping[str,Any]，单请求 Dataset/Collator batch；提供 hardmask、BOX shape
          与 voxel size
        - stage1_model_name: str，`Find_0`、`Find_1` 或 `unet_c1`

    输出:
        - payload: dict[str,np.ndarray]，概率为 float32，学习特征仍保留原值并在后续
          归档边界转 float16；所有 batch-first 网格已取唯一 batch row

    本入口故意只接受 batch size 1。F1/CLG/Selected 每个来源 BOX 独立 forward，避免
    在此重新实现跨 BOX ragged 拆分；生产层可在更外层调度多个请求。
    """
    if stage1_model_name not in (*FIND_MODEL_NAMES, "unet_c1"):
        raise ValueError(f"未知 stage1_model_name={stage1_model_name!r}")
    ligand_probability = logits_to_probability(
        forward_output["voxel_logits_ligand"]
    )
    aux_probability = logits_to_probability(forward_output["voxel_logits_aux"])
    if ligand_probability.shape[0] != 1 or aux_probability.shape[0] != 1:
        raise ValueError("centered 完整 forward adapter 只接受 batch size 1")

    hardmask = _single_grid_from_batch(batch["hardmask"], "hardmask", np.bool_)
    voxel_features = forward_output.get("voxel_features")
    if voxel_features is None:
        voxel_outputs = forward_output.get("voxel_outputs")
        if not isinstance(voxel_outputs, Mapping):
            raise KeyError("完整 forward 缺少具名 voxel_features")
        voxel_features = voxel_outputs.get("voxel_features")
    if not isinstance(voxel_features, Mapping):
        raise TypeError("voxel_features 必须是具名 tensor mapping")

    payload: dict[str, np.ndarray] = {
        "ligand_probability": ligand_probability[0],
        "hardmask": hardmask,
        "voxel_aux_probability_grid": aux_probability[0],
        "voxel_final_grid": _single_feature_grid(voxel_features, "voxel_final"),
        "voxel_ds_2": _single_feature_grid(voxel_features, "voxel_ds_2"),
        "voxel_ds_3": _single_feature_grid(voxel_features, "voxel_ds_3"),
        "voxel_ds_4": _single_feature_grid(voxel_features, "voxel_ds_4"),
        "voxel_c4": _single_feature_grid(voxel_features, "voxel_c4"),
    }
    if stage1_model_name not in FIND_MODEL_NAMES:
        return payload

    atom_local_xyz = _as_numpy(
        forward_output["atom_coord_local_voxel"], np.float32
    )
    atom_global_index = _as_numpy(
        forward_output["atom_global_indices"], np.int64
    ).reshape(-1)
    atom_probability = _sigmoid_rows(
        forward_output["atom_logits"], "atom_logits"
    )
    atom_count = int(atom_local_xyz.shape[0])
    if atom_local_xyz.shape != (atom_count, 3):
        raise ValueError("atom_coord_local_voxel 必须为 [N_A,3] XYZ")
    if atom_global_index.shape != (atom_count,) or atom_probability.shape != (atom_count,):
        raise ValueError("A identity/probability 必须逐行对齐")
    box_shape_xyz = _single_vector_from_batch(
        batch["box_shape_zyx"], "box_shape_zyx", np.float32
    )[[2, 1, 0]]
    voxel_size_xyz = _single_vector_from_batch(
        batch["voxel_size_world"], "voxel_size_world", np.float32
    )
    atom_centered_world = (
        atom_local_xyz - box_shape_xyz[None, :] / np.float32(2.0)
    ) * voxel_size_xyz[None, :]
    payload.update(
        {
            "A_global_index": atom_global_index,
            "A_coord_local_xyz": atom_local_xyz,
            "A_coord_centered_world": atom_centered_world.astype(np.float32, copy=False),
            "A_probability": atom_probability,
            "A_feat_L1": _aligned_rows(forward_output, "A_feat_L1", atom_count),
            "A_feat_L2": _aligned_rows(forward_output, "A_feat_L2", atom_count),
            "A_feat_L3": _aligned_rows(forward_output, "A_feat_L3", atom_count),
            "A_feat_L4": _aligned_rows(forward_output, "A_feat_L4", atom_count),
        }
    )

    pseudo_local_xyz = _as_numpy(
        forward_output["anchor_coord_local_voxel"], np.float32
    )
    pseudo_probability = _sigmoid_rows(
        forward_output["pseudo_logits"], "pseudo_logits"
    )
    pseudo_count = int(pseudo_local_xyz.shape[0])
    if pseudo_local_xyz.shape != (pseudo_count, 3) or pseudo_probability.shape != (
        pseudo_count,
    ):
        raise ValueError("P coordinates/probability 必须逐行对齐")
    payload.update(
        {
            "P_coord_local_xyz": pseudo_local_xyz,
            "P_probability": pseudo_probability,
            "P_feat_L2": _aligned_rows(forward_output, "P_feat_L2", pseudo_count),
            "P_feat_L3": _aligned_rows(forward_output, "P_feat_L3", pseudo_count),
            "P_feat_L4": _aligned_rows(forward_output, "P_feat_L4", pseudo_count),
        }
    )
    return payload


def _single_grid_from_batch(value: Any, field: str, dtype: Any) -> np.ndarray:
    """从 `[1,D,H,W]`、`[1,1,D,H,W]` 或无 batch 的三维字段取唯一网格。"""
    array = _as_numpy(value, dtype)
    if array.ndim == 5 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{field} 必须能解析为单张三维网格")
    return array


def _single_feature_grid(features: Mapping[str, Any], field: str) -> np.ndarray:
    """读取 `[1,C,D,H,W]` 具名特征并去掉唯一 batch 维。"""
    if field not in features:
        raise KeyError(f"完整 forward 的 voxel_features 缺少 {field}")
    array = _to_numpy(features[field])
    if array.ndim != 5 or int(array.shape[0]) != 1:
        raise ValueError(f"{field} 必须为 batch size 1 的 [1,C,D,H,W]")
    return array[0]


def _single_vector_from_batch(value: Any, field: str, dtype: Any) -> np.ndarray:
    """从 `[1,3]` 或 `[3]` batch 字段读取唯一三向量。"""
    array = _as_numpy(value, dtype)
    if array.shape == (1, 3):
        array = array[0]
    if array.shape != (3,):
        raise ValueError(f"{field} 必须为 [3] 或 [1,3]")
    return array


def _sigmoid_rows(value: Any, field: str) -> np.ndarray:
    """把 `[N]` 或单通道 `[N,1]` logits 稳定转为 float32 概率行。"""
    logits = _as_numpy(value, np.float32)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1:
        raise ValueError(f"{field} 必须为 [N] 或 [N,1]")
    probability = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probability[~positive] = exp_logits / (1.0 + exp_logits)
    return probability


def _aligned_rows(
    forward_output: Mapping[str, Any],
    field: str,
    expected_rows: int,
) -> np.ndarray:
    """读取一个具名二维学习特征并校验其实体行数。"""
    value = _to_numpy(forward_output[field])
    if value.ndim != 2 or int(value.shape[0]) != int(expected_rows):
        raise ValueError(f"{field} 必须为 [{expected_rows},C]")
    return value


def produce_f1_centered_entries(
    nodes: Sequence[ComponentNode],
    f1_threshold_grid_index: int,
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    full_forward: FullForwardCallback,
) -> list[dict[str, Any]]:
    """
    为 `t_F1` 层全部 eligible component 生成 F1_centered entries。

    输入参数:
        - nodes: Sequence[ComponentNode], 当前 PDB forest nodes
        - f1_threshold_grid_index: int, `t_F1` 的整数 j
        - geometry: CenteredGeometry, 当前 PDB 几何
        - stage1_model_name: str, 当前 producer
        - split: str, 当前数据划分
        - pdb_id: str, 当前完整图身份
        - resolve_box_start: StartResolver, 与训练/forest 共用的唯一 resolver
        - full_forward: FullForwardCallback, 正常完整 forward 的具名输出 callback

    输出:
        - entries: list[dict[str,Any]], 每项权威 voxel 恰为来源全图 component mask；
          局部额外组件不进入身份或 payload
    """
    sources = [
        node
        for node in nodes
        if node.candidate_eligible
        and node.threshold_grid_index == int(f1_threshold_grid_index)
    ]
    sources.sort(key=lambda node: (-node.probability_mean, node.tree_id, node.node_id))
    return [
        _produce_success_entry(
            source=node,
            authority_global_linear_index=node.voxel_global_linear_index,
            centered_role="F1_centered",
            centered_box_index=index,
            geometry=geometry,
            stage1_model_name=stage1_model_name,
            split=split,
            pdb_id=pdb_id,
            resolve_box_start=resolve_box_start,
            full_forward=full_forward,
        )
        for index, node in enumerate(sources)
    ]


def produce_clg_centered_entries(
    clgs: Sequence[CLG],
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    full_forward: FullForwardCallback,
) -> list[dict[str, Any]]:
    """
    为每个成功 CLG 以其唯一 oldest mask 生成一个 CLG_centered entry。

    输入参数:
        - clgs: Sequence[CLG], 按 `CLG_id` 来源顺序排列的成功 CLG
        - geometry: CenteredGeometry, 当前 PDB 几何
        - stage1_model_name: str, 当前 producer
        - split: str, 当前数据划分
        - pdb_id: str, 当前完整图身份
        - resolve_box_start: StartResolver, 统一 80³ 起点解析器
        - full_forward: FullForwardCallback, 正常完整 forward callback

    输出:
        - entries: list[dict[str,Any]], oldest 共享 voxel 表，candidate 用局部
          offsets+indices 引用；Find A membership 单独保存，P 不建立 membership
    """
    entries: list[dict[str, Any]] = []
    for centered_box_index, clg in enumerate(clgs):
        source = clg.oldest_node
        entry = _produce_success_entry(
            source=source,
            authority_global_linear_index=source.voxel_global_linear_index,
            centered_role="CLG_centered",
            centered_box_index=centered_box_index,
            geometry=geometry,
            stage1_model_name=stage1_model_name,
            split=split,
            pdb_id=pdb_id,
            resolve_box_start=resolve_box_start,
            full_forward=full_forward,
        )
        entry.update(
            {
                "CLG_id": int(clg.CLG_id),
                "CLG_seed_node_id": int(clg.seed_node.node_id),
                "CLG_oldest_node_id": int(clg.oldest_node.node_id),
                "candidate_node_id": np.asarray(
                    [node.node_id for node in clg.candidate_nodes], dtype=np.int32
                ),
                "candidate_threshold_grid_index": np.asarray(
                    [node.threshold_grid_index for node in clg.candidate_nodes],
                    dtype=np.int32,
                ),
            }
        )
        oldest_global = source.voxel_global_linear_index
        entry["candidate_voxel_membership"] = [
            _membership_rows(oldest_global, node.voxel_global_linear_index)
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
    full_forward: FullForwardCallback,
) -> list[dict[str, Any]]:
    """
    对每个唯一 selected source 重跑居中概率并按 source 阈值生成 refined blob。

    输入参数:
        - selected_nodes: Sequence[ComponentNode], selector 最终选择的原 forest nodes；
          同一 `(tree_id,node_id)` 最多出现一次
        - geometry: CenteredGeometry, 当前 PDB 几何
        - stage1_model_name: str, 当前 producer
        - split: str, 当前数据划分
        - pdb_id: str, 当前完整图身份
        - resolve_box_start: StartResolver, 统一 80³ 起点解析器
        - full_forward: FullForwardCallback, 正常完整 forward callback

    输出:
        - entries: list[dict[str,Any]], `refine_status` 固定为:
            - `success`(0): 至少一个局部组件与 source mask 正交集，取 IoU 最大者
            - `empty`(1): 按 source 阈值二值化后无局部组件
            - `no_overlap`(2): 有组件但与投影 source mask 的交集全为 0
            - `failed`(3): forward、组件构造或必要校验失败，保留身份且无权威 payload
          多组件 IoU 并列时沿 scipy 26-CCL 的自然 label 顺序取第一个
    """
    identities = [(node.tree_id, node.node_id) for node in selected_nodes]
    if len(set(identities)) != len(identities):
        raise ValueError("正式 Selected 目录中同一 source node 最多出现一次")
    entries: list[dict[str, Any]] = []
    for centered_box_index, source in enumerate(selected_nodes):
        box_start = _resolve_source_start(source, geometry, resolve_box_start)
        base = _base_entry(
            source=source,
            centered_role="Selected_Refined_Centered",
            centered_box_index=centered_box_index,
            box_start_zyx=box_start,
            geometry=geometry,
        )
        try:
            request = _request_from_entry(
                base,
                "Selected_Refined_Centered",
                stage1_model_name,
                split,
                pdb_id,
            )
            forward_payload = dict(full_forward(request))
            probability = _processed_probability(
                forward_payload, stage1_model_name, geometry.box_shape_zyx
            )
            binary = probability >= float(source.threshold_value)
            labels, component_count = ndimage.label(
                binary, structure=ndimage.generate_binary_structure(3, 3)
            )
            if int(component_count) == 0:
                entries.append(_empty_selected_entry(base, "empty"))
                continue
            source_local = _global_to_local_zyx(
                source.voxel_global_linear_index,
                box_start,
                geometry.full_shape_zyx,
                geometry.box_shape_zyx,
            )
            source_mask = np.zeros(geometry.box_shape_zyx, dtype=np.bool_)
            source_mask[tuple(source_local.T)] = True
            intersection_counts = np.asarray(
                [
                    np.logical_and(labels == label_id, source_mask).sum()
                    for label_id in range(1, int(component_count) + 1)
                ],
                dtype=np.int64,
            )
            if int(intersection_counts.max(initial=0)) == 0:
                entries.append(_empty_selected_entry(base, "no_overlap"))
                continue
            component_sizes = np.bincount(labels.reshape(-1), minlength=int(component_count) + 1)[1:]
            source_size = int(source_mask.sum())
            unions = component_sizes + source_size - intersection_counts
            iou = intersection_counts / unions.astype(np.float64)
            selected_label = int(np.argmax(iou)) + 1
            refined_local = np.argwhere(labels == selected_label).astype(np.int16)
            refined_global = np.ravel_multi_index(
                (refined_local.astype(np.int64) + box_start[None, :]).T,
                geometry.full_shape_zyx,
            ).astype(np.int64)
            entry = _payload_from_forward(
                base=base,
                authority_global_linear_index=refined_global,
                probability=probability,
                forward_payload=forward_payload,
                geometry=geometry,
                stage1_model_name=stage1_model_name,
            )
            entry["refine_status"] = "success"
            entries.append(entry)
        except Exception:
            entries.append(_empty_selected_entry(base, "failed"))
    return _normalize_selected_payloads(entries, stage1_model_name)


def publish_centered_entries(
    paths: Stage1ArtifactPaths,
    centered_role: str,
    entries: Sequence[Mapping[str, Any]],
) -> None:
    """
    把一个 PDB 的 centered entries 聚合、校验、原子发布后写 role `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 路径
        - centered_role: str, 三类正式 centered role
        - entries: Sequence[Mapping[str,Any]], 当前 PDB 的全部同类 BOX

    输出:
        - None, 最终只有一个 `{centered_role}.npz`，完成标记最后写入
    """
    arrays = pack_centered_entries(entries, centered_role)
    atomic_savez_compressed(
        paths.centered_npz(centered_role),
        arrays,
        validator=lambda value: validate_centered_archive(value, centered_role),
    )
    mark_role_complete(paths, centered_role)


def _produce_success_entry(
    source: ComponentNode,
    authority_global_linear_index: np.ndarray,
    centered_role: str,
    centered_box_index: int,
    geometry: CenteredGeometry,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
    resolve_box_start: StartResolver,
    full_forward: FullForwardCallback,
) -> dict[str, Any]:
    """执行一次普通居中 full forward，并按权威 voxel 集稀疏抽取 payload。"""
    box_start = _resolve_source_start(source, geometry, resolve_box_start)
    base = _base_entry(
        source, centered_role, centered_box_index, box_start, geometry
    )
    request = _request_from_entry(
        base, centered_role, stage1_model_name, split, pdb_id
    )
    forward_payload = dict(full_forward(request))
    probability = _processed_probability(
        forward_payload, stage1_model_name, geometry.box_shape_zyx
    )
    return _payload_from_forward(
        base,
        authority_global_linear_index,
        probability,
        forward_payload,
        geometry,
        stage1_model_name,
    )


def _base_entry(
    source: ComponentNode,
    centered_role: str,
    centered_box_index: int,
    box_start_zyx: np.ndarray,
    geometry: CenteredGeometry,
) -> dict[str, Any]:
    """构造不依赖 forward 成功与否的 entry 身份与几何字段。"""
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


def _request_from_entry(
    entry: Mapping[str, Any],
    centered_role: str,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
) -> CenteredRequest:
    """由共同 entry 字段构造 callback 请求。"""
    return CenteredRequest(
        stage1_model_name=str(stage1_model_name),
        split=str(split),
        pdb_id=str(pdb_id),
        centered_role=centered_role,
        centered_box_index=int(entry["centered_box_index"]),
        box_start_zyx=tuple(int(value) for value in entry["box_start_zyx"]),
        source_tree_id=int(entry["source_tree_id"]),
        source_node_id=int(entry["source_node_id"]),
        source_threshold_grid_index=int(entry["source_threshold_grid_index"]),
    )


def _resolve_source_start(
    source: ComponentNode,
    geometry: CenteredGeometry,
    resolve_box_start: StartResolver,
) -> np.ndarray:
    """调用统一 resolver 并验证来源 bbox 完整落入真实 80³ BOX。"""
    start = np.asarray(
        resolve_box_start(source.centroid_zyx, geometry.full_shape_zyx), dtype=np.int64
    )
    box_shape = np.asarray(geometry.box_shape_zyx, dtype=np.int64)
    if start.shape != (3,) or bool(np.any(start < 0)):
        raise ValueError("resolve_box_start 返回不合法 ZYX 起点")
    if bool(np.any(start + box_shape > np.asarray(geometry.full_shape_zyx))):
        raise ValueError("resolve_box_start 产生越过完整图的 BOX")
    if bool(np.any(source.bbox_min_zyx < start)) or bool(
        np.any(source.bbox_max_zyx > start + box_shape - 1)
    ):
        raise ValueError("来源 component 的完整 bbox 无法装入 resolved 80³ BOX")
    return start


def _processed_probability(
    forward_payload: Mapping[str, Any],
    stage1_model_name: str,
    box_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """读取 full-forward sigmoid 概率，并对两个 Find 应用当前 BOX hardmask。"""
    probability = _as_numpy(forward_payload["ligand_probability"], np.float32)
    if probability.shape == (1, *box_shape_zyx):
        probability = probability[0]
    if probability.shape != box_shape_zyx:
        raise ValueError("ligand_probability 必须为 [80,80,80] 或 [1,80,80,80]")
    hardmask = forward_payload.get("hardmask")
    return postprocess_ligand_probability(probability, stage1_model_name, hardmask)


def _payload_from_forward(
    base: Mapping[str, Any],
    authority_global_linear_index: np.ndarray,
    probability: np.ndarray,
    forward_payload: Mapping[str, Any],
    geometry: CenteredGeometry,
    stage1_model_name: str,
) -> dict[str, Any]:
    """从具名 full-forward 输出抽取与权威 voxel/实体对齐的正式 payload。"""
    entry = dict(base)
    box_start = np.asarray(base["box_start_zyx"], dtype=np.int64)
    local_zyx = _global_to_local_zyx(
        authority_global_linear_index,
        box_start,
        geometry.full_shape_zyx,
        geometry.box_shape_zyx,
    )
    entry["voxel_index_local_zyx"] = local_zyx.astype(np.int16)
    entry["centered_probability"] = probability[tuple(local_zyx.T)].astype(np.float32)
    voxel_final_grid = _to_numpy(forward_payload["voxel_final_grid"])
    if voxel_final_grid.shape != (48, *geometry.box_shape_zyx):
        raise ValueError("voxel_final_grid 必须是 channel-first [48,80,80,80]")
    entry["voxel_final"] = voxel_final_grid[
        :, local_zyx[:, 0], local_zyx[:, 1], local_zyx[:, 2]
    ].T.astype(np.float16)
    for field in FIXED_VOXEL_GRID_FIELDS:
        entry[field] = _as_numpy(forward_payload[field], np.float16)

    hardmask = _as_numpy(forward_payload["hardmask"], np.bool_)
    aux_probability = _as_numpy(forward_payload["voxel_aux_probability_grid"], np.float32)
    if hardmask.shape != geometry.box_shape_zyx or aux_probability.shape != geometry.box_shape_zyx:
        raise ValueError("hardmask 与 voxel_aux_probability_grid 必须为 [80,80,80]")
    aux_index = np.argwhere(hardmask).astype(np.int16)
    entry["voxel_aux_index_local_zyx"] = aux_index
    entry["voxel_aux_probability"] = aux_probability[tuple(aux_index.T)].astype(np.float32)

    if stage1_model_name in FIND_MODEL_NAMES:
        _copy_find_tables(entry, forward_payload, authority_global_linear_index, geometry)
    return entry


def _copy_find_tables(
    entry: dict[str, Any],
    forward_payload: Mapping[str, Any],
    authority_global_linear_index: np.ndarray,
    geometry: CenteredGeometry,
) -> None:
    """复制完整 P 表，并按来源 blob 10 Å 包络筛出 BOX 内 A 表。"""
    p_fields = (
        "P_coord_local_xyz",
        "P_probability",
        "P_feat_L2",
        "P_feat_L3",
        "P_feat_L4",
    )
    for field in p_fields:
        dtype = np.float32 if field in ("P_coord_local_xyz", "P_probability") else np.float16
        entry[field] = _as_numpy(forward_payload[field], dtype)
    p_length = int(entry["P_coord_local_xyz"].shape[0])
    if any(int(np.asarray(entry[field]).shape[0]) != p_length for field in p_fields):
        raise ValueError("全部 P 字段必须逐行对齐")

    a_fields = (
        "A_global_index",
        "A_coord_local_xyz",
        "A_coord_centered_world",
        "A_probability",
        "A_feat_L1",
        "A_feat_L2",
        "A_feat_L3",
        "A_feat_L4",
    )
    a_dtypes = {
        "A_global_index": np.int64,
        "A_coord_local_xyz": np.float32,
        "A_coord_centered_world": np.float32,
        "A_probability": np.float32,
        "A_feat_L1": np.float16,
        "A_feat_L2": np.float16,
        "A_feat_L3": np.float16,
        "A_feat_L4": np.float16,
    }
    a_values = {field: _as_numpy(forward_payload[field], a_dtypes[field]) for field in a_fields}
    a_length = int(a_values["A_global_index"].shape[0])
    if any(int(value.shape[0]) != a_length for value in a_values.values()):
        raise ValueError("全部 A 字段必须逐行对齐")
    coords = a_values["A_coord_local_xyz"]
    core = np.all((coords >= 0.0) & (coords < np.asarray(geometry.box_shape_zyx)[[2, 1, 0]]), axis=1)
    envelope = _points_within_blob_envelope(
        points_local_xyz=coords,
        blob_global_linear_index=authority_global_linear_index,
        box_start_zyx=np.asarray(entry["box_start_zyx"], dtype=np.int64),
        full_shape_zyx=geometry.full_shape_zyx,
        voxel_size_xyz=geometry.voxel_size_xyz,
        distance_angstrom=10.0,
    )
    selected = core & envelope
    for field, value in a_values.items():
        entry[field] = value[selected]


def _points_within_blob_envelope(
    points_local_xyz: np.ndarray,
    blob_global_linear_index: np.ndarray,
    box_start_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
    voxel_size_xyz: np.ndarray,
    distance_angstrom: float,
) -> np.ndarray:
    """返回 BOX 内点是否位于 blob voxel center 的指定 Å 欧氏包络内。"""
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
        points_local_world, k=1, distance_upper_bound=float(distance_angstrom)
    )
    return np.isfinite(distance)


def _membership_rows(authority_sorted: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """把 candidate 全图 voxel indices 恢复为所属 entry voxel 段局部行。"""
    authority = np.asarray(authority_sorted, dtype=np.int64)
    subset_values = np.asarray(subset, dtype=np.int64)
    rows = np.searchsorted(authority, subset_values)
    if rows.size and (
        int(rows.max()) >= authority.size
        or not np.array_equal(authority[rows], subset_values)
    ):
        raise ValueError("candidate mask 不是 CLG_oldest_node 权威 voxel 集的子集")
    return rows.astype(np.int32)


def _global_to_local_zyx(
    global_linear_index: np.ndarray,
    box_start_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
    box_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """把全图 C-order 线性 index 转为当前 BOX 内唯一 ZYX index。"""
    global_indices = np.asarray(global_linear_index, dtype=np.int64)
    if np.unique(global_indices).size != global_indices.size:
        raise ValueError("权威 voxel 集中的全图 index 必须唯一")
    global_zyx = np.column_stack(np.unravel_index(global_indices, full_shape_zyx))
    local = global_zyx - box_start_zyx[None, :]
    if local.size and (
        bool(np.any(local < 0))
        or bool(np.any(local >= np.asarray(box_shape_zyx, dtype=np.int64)))
    ):
        raise ValueError("权威 global_component_mask 未完整落入当前 BOX")
    return local.astype(np.int64, copy=False)


def _empty_selected_entry(base: Mapping[str, Any], refine_status: str) -> dict[str, Any]:
    """构造保留来源身份但没有权威变长 payload 的 Selected 记录。"""
    entry = dict(base)
    entry.update(
        {
            "refine_status": refine_status,
            "voxel_index_local_zyx": np.empty((0, 3), dtype=np.int16),
            "centered_probability": np.empty(0, dtype=np.float32),
            "voxel_final": np.empty((0, 48), dtype=np.float16),
            "voxel_aux_index_local_zyx": np.empty((0, 3), dtype=np.int16),
            "voxel_aux_probability": np.empty(0, dtype=np.float32),
        }
    )
    return entry


def _normalize_selected_payloads(
    entries: list[dict[str, Any]],
    stage1_model_name: str,
) -> list[dict[str, Any]]:
    """让失败 Selected entry 具有可聚合的空 ragged 表与固定网格占位。"""
    successful = next((entry for entry in entries if entry["refine_status"] == "success"), None)
    fixed_shapes = {
        "voxel_ds_2": (256, 20, 20, 20),
        "voxel_ds_3": (256, 10, 10, 10),
        "voxel_ds_4": (256, 5, 5, 5),
        "voxel_c4": (256, 5, 5, 5),
    }
    for entry in entries:
        if entry["refine_status"] == "success":
            continue
        for field, shape in fixed_shapes.items():
            entry[field] = np.zeros(shape, dtype=np.float16)
        if successful is not None:
            for group_fields in (
                ("P_coord_local_xyz", "P_probability", "P_feat_L2", "P_feat_L3", "P_feat_L4"),
                ("A_global_index", "A_coord_local_xyz", "A_coord_centered_world", "A_probability", "A_feat_L1", "A_feat_L2", "A_feat_L3", "A_feat_L4"),
            ):
                for field in group_fields:
                    if field in successful:
                        template = np.asarray(successful[field])
                        entry[field] = np.empty((0, *template.shape[1:]), dtype=template.dtype)
        elif stage1_model_name in FIND_MODEL_NAMES:
            # 全部 source 都失败时没有真实通道宽度可恢复，因此不伪造 P/A 模态。
            pass
    return entries


def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    """把 NumPy 或 torch.Tensor 输出搬到 CPU 并转换为契约 dtype。"""
    return np.asarray(_to_numpy(value), dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    """把 NumPy 或 torch.Tensor 搬到 CPU；BF16 以 float32 作为 NumPy 桥接类型。"""
    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            # NumPy 没有原生 bfloat16 dtype；只在桥接时提升，正式 learning feature
            # 仍由调用方按契约压成 float16，不改变模型内部 BF16 计算。
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(dtype=torch.float32)
            return tensor.numpy()
    except ImportError:
        pass
    return np.asarray(value)
