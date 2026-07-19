"""由多阈值 26-连通组件构造只读 ComponentForest。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from src.artifacts.io import atomic_savez_compressed, atomic_write_json, validate_offsets
from src.artifacts.paths import Stage1ArtifactPaths
from src.artifacts.states import mark_role_complete

from .clg import CLGEnumerationResult
from .structures import ComponentForest, ComponentNode, ComponentTree


INELIGIBLE_REASON: dict[int, str] = {
    0: "eligible",
    1: "below_min_voxels",
    2: "above_max_voxels",
    3: "bbox_not_contained_by_resolved_box",
}


@dataclass(eq=False)
class _RawNode:
    """构树期间使用的临时节点；发布前会转换成只读 ComponentNode。"""

    threshold_grid_index: int
    voxel_global_linear_index: np.ndarray
    bbox_min_zyx: np.ndarray
    bbox_max_zyx: np.ndarray
    centroid_zyx: np.ndarray
    probability_mean: float
    probability_max: float
    candidate_eligible: bool
    ineligible_reason_code: int
    parent: "_RawNode | None" = None
    children: list["_RawNode"] = field(default_factory=list)


def build_component_forest(
    probability_map: np.ndarray,
    threshold_grid_indices: Sequence[int],
    denominator: int,
    min_voxels: int,
    max_voxels: int,
    resolve_box_start: Callable[[np.ndarray, tuple[int, int, int]], Sequence[int]],
    box_shape_zyx: tuple[int, int, int] = (80, 80, 80),
) -> tuple[ComponentForest, dict[str, object]]:
    """
    在实际去重阈值层上构造 26-连通组件森林。

    输入参数:
        - probability_map: np.ndarray, (D,H,W), float32，producer 后处理后的完整图概率
        - threshold_grid_indices: Sequence[int], 实际整数阈值 j；函数会去重并降序
        - denominator: int, 阈值分母，正式值为 32768
        - min_voxels: int, candidate 最小体素数，正式值为 32
        - max_voxels: int, 已由 GT occurrence Q95×1.5 冻结的最大体素数
        - resolve_box_start: Callable, 接收 `(centroid_zyx, full_shape_zyx)`，返回合法
          80³ ZYX 起点；它必须与训练/居中请求共用同一实现
        - box_shape_zyx: tuple[int,int,int], 居中 BOX 形状，正式值为 (80,80,80)

    输出:
        - forest: ComponentForest, parent 指向相邻更低阈值包含组件的只读森林
        - summary: dict[str, object], 含阈值、逐层 node/eligible 数和 reason code 说明
    """
    probability = np.asarray(probability_map, dtype=np.float32)
    if probability.ndim != 3 or not bool(np.all(np.isfinite(probability))):
        raise ValueError("probability_map 必须是有限值三维数组")
    if denominator <= 0 or min_voxels <= 0 or max_voxels < min_voxels:
        raise ValueError("denominator/min_voxels/max_voxels 配置不合法")
    thresholds = sorted({int(value) for value in threshold_grid_indices}, reverse=True)
    if not thresholds or thresholds[0] > denominator or thresholds[-1] < 0:
        raise ValueError("threshold_grid_indices 必须是 [0,denominator] 内的非空集合")
    full_shape = tuple(int(value) for value in probability.shape)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if box_shape.shape != (3,) or bool(np.any(np.asarray(full_shape) < box_shape)):
        raise ValueError("完整图三轴必须不小于 centered BOX")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    layers: list[list[_RawNode]] = []
    label_layers: list[np.ndarray] = []
    layer_stats: list[dict[str, int]] = []
    for threshold_grid_index in thresholds:
        threshold = float(threshold_grid_index) / float(denominator)
        labels, component_count = ndimage.label(probability >= threshold, structure=structure)
        objects = ndimage.find_objects(labels, max_label=int(component_count))
        nodes: list[_RawNode] = []
        reason_counts = {reason: 0 for reason in INELIGIBLE_REASON.values()}
        for label_id, object_slices in enumerate(objects, start=1):
            if object_slices is None:
                continue
            local_mask = labels[object_slices] == label_id
            local_zyx = np.argwhere(local_mask)
            start_zyx = np.asarray([axis_slice.start for axis_slice in object_slices], dtype=np.int64)
            voxel_zyx = local_zyx + start_zyx[None, :]
            linear_index = np.ravel_multi_index(voxel_zyx.T, full_shape).astype(
                np.int64, copy=False
            )
            linear_index.sort()
            bbox_min = voxel_zyx.min(axis=0).astype(np.int32)
            bbox_max = voxel_zyx.max(axis=0).astype(np.int32)
            centroid = voxel_zyx.mean(axis=0, dtype=np.float64).astype(np.float32)
            values = probability.reshape(-1)[linear_index]
            reason_code = _candidate_reason_code(
                voxel_count=int(linear_index.size),
                bbox_min_zyx=bbox_min,
                bbox_max_zyx=bbox_max,
                centroid_zyx=centroid,
                full_shape_zyx=full_shape,
                box_shape_zyx=box_shape,
                min_voxels=int(min_voxels),
                max_voxels=int(max_voxels),
                resolve_box_start=resolve_box_start,
            )
            reason_counts[INELIGIBLE_REASON[reason_code]] += 1
            nodes.append(
                _RawNode(
                    threshold_grid_index=threshold_grid_index,
                    voxel_global_linear_index=linear_index,
                    bbox_min_zyx=bbox_min,
                    bbox_max_zyx=bbox_max,
                    centroid_zyx=centroid,
                    probability_mean=float(values.mean(dtype=np.float64)),
                    probability_max=float(values.max()),
                    candidate_eligible=reason_code == 0,
                    ineligible_reason_code=reason_code,
                )
            )
        layers.append(nodes)
        label_layers.append(labels)
        layer_stats.append(
            {
                "threshold_grid_index": int(threshold_grid_index),
                "n_nodes": len(nodes),
                "n_eligible": int(sum(node.candidate_eligible for node in nodes)),
                **{f"n_{name}": int(count) for name, count in reason_counts.items()},
            }
        )

    # 高阈值组件只能落入相邻低阈值层的一个组件；这条边定义 direct parent。
    for high_layer_index in range(len(layers) - 1):
        high_nodes = layers[high_layer_index]
        low_nodes = layers[high_layer_index + 1]
        low_labels = label_layers[high_layer_index + 1]
        low_by_label = {label_id: node for label_id, node in enumerate(low_nodes, start=1)}
        low_flat = low_labels.reshape(-1)
        for child in high_nodes:
            containing_labels = np.unique(low_flat[child.voxel_global_linear_index])
            containing_labels = containing_labels[containing_labels > 0]
            if containing_labels.size != 1:
                raise RuntimeError("阈值单调性被破坏：高阈值组件没有唯一相邻低阈值 parent")
            parent = low_by_label[int(containing_labels[0])]
            child.parent = parent
            parent.children.append(child)

    roots = [node for layer in layers for node in layer if node.parent is None]
    roots.sort(key=_node_sort_key)
    trees: list[ComponentTree] = []
    for tree_id, root in enumerate(roots):
        raw_order = _deterministic_depth_first(root)
        converted = {
            raw_node: ComponentNode(
                tree_id=tree_id,
                node_id=node_id,
                threshold_grid_index=raw_node.threshold_grid_index,
                threshold_value=float(raw_node.threshold_grid_index) / float(denominator),
                voxel_global_linear_index=raw_node.voxel_global_linear_index,
                bbox_min_zyx=raw_node.bbox_min_zyx,
                bbox_max_zyx=raw_node.bbox_max_zyx,
                centroid_zyx=raw_node.centroid_zyx,
                probability_mean=raw_node.probability_mean,
                probability_max=raw_node.probability_max,
                candidate_eligible=raw_node.candidate_eligible,
                ineligible_reason_code=raw_node.ineligible_reason_code,
            )
            for node_id, raw_node in enumerate(raw_order)
        }
        for raw_node, node in converted.items():
            node.parent = None if raw_node.parent is None else converted[raw_node.parent]
            node.children.extend(converted[child] for child in sorted(raw_node.children, key=_node_sort_key))
        trees.append(ComponentTree(tree_id=tree_id, nodes=tuple(converted.values())))

    forest = ComponentForest(trees=trees)
    summary: dict[str, object] = {
        "denominator": int(denominator),
        "threshold_grid_indices_descending": thresholds,
        "connectivity": 26,
        "min_voxels": int(min_voxels),
        "max_voxels": int(max_voxels),
        "n_trees": len(trees),
        "n_nodes": len(forest.nodes),
        "ineligible_reason_code": {str(code): name for code, name in INELIGIBLE_REASON.items()},
        "layers": layer_stats,
    }
    return forest, summary


def _candidate_reason_code(
    voxel_count: int,
    bbox_min_zyx: np.ndarray,
    bbox_max_zyx: np.ndarray,
    centroid_zyx: np.ndarray,
    full_shape_zyx: tuple[int, int, int],
    box_shape_zyx: np.ndarray,
    min_voxels: int,
    max_voxels: int,
    resolve_box_start: Callable[[np.ndarray, tuple[int, int, int]], Sequence[int]],
) -> int:
    """按体积优先、bbox 其次返回稳定的 candidate ineligible reason code。"""
    if voxel_count < min_voxels:
        return 1
    if voxel_count > max_voxels:
        return 2
    resolved_start = np.asarray(
        resolve_box_start(centroid_zyx, full_shape_zyx), dtype=np.int64
    )
    if resolved_start.shape != (3,):
        raise ValueError("resolve_box_start 必须返回长度 3 的 ZYX 起点")
    max_start = np.asarray(full_shape_zyx, dtype=np.int64) - box_shape_zyx
    if bool(np.any(resolved_start < 0)) or bool(np.any(resolved_start > max_start)):
        raise ValueError("resolve_box_start 返回了不合法的真实 crop 起点")
    box_max = resolved_start + box_shape_zyx - 1
    if bool(np.any(bbox_min_zyx < resolved_start)) or bool(np.any(bbox_max_zyx > box_max)):
        return 3
    return 0


def _node_sort_key(node: _RawNode) -> tuple[int, float, int]:
    """为 root/children 提供与平台无关的稳定排序键。"""
    first_voxel = (
        int(node.voxel_global_linear_index[0])
        if node.voxel_global_linear_index.size
        else -1
    )
    return (-int(node.threshold_grid_index), -float(node.probability_mean), first_voxel)


def _deterministic_depth_first(root: _RawNode) -> list[_RawNode]:
    """以 parent-before-children 的确定性顺序展开一棵临时树。"""
    result: list[_RawNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(sorted(node.children, key=_node_sort_key)))
    return result


def count_f1_eligible(
    forest: ComponentForest,
    f1_threshold_grid_index: int,
) -> int:
    """
    统计 `t_F1` 层正式 candidate-eligible component 数。

    输入参数:
        - forest: ComponentForest, 当前 PDB 的内存森林
        - f1_threshold_grid_index: int, `t_F1` 对应整数 j

    输出:
        - count: int, 用于 `N_F1_eligible>200` 异常终态判定
    """
    return sum(
        node.candidate_eligible
        and node.threshold_grid_index == int(f1_threshold_grid_index)
        for node in forest.nodes
    )


def publish_component_artifacts(
    paths: Stage1ArtifactPaths,
    forest: ComponentForest,
    clg_result: CLGEnumerationResult,
    overlap_arrays: Mapping[str, np.ndarray],
    forest_summary: Mapping[str, object],
) -> None:
    """
    原子发布 forest、CLG、overlap、summary，最后写 components `_COMPLETE`。

    输入参数:
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 路径
        - forest: ComponentForest, 原始只读森林
        - clg_result: CLGEnumerationResult, 成功 CLG 与 cap 统计
        - overlap_arrays: Mapping[str,np.ndarray], `candidate_occurrence_offsets` 同时切分
          `overlap_occurrence_index` 与 `intersection_voxel_count` 的基础事实
        - forest_summary: Mapping[str,object], 阈值层与 eligibility 统计

    输出:
        - None, 四个 payload 都可冷读后才发布 role 完成标记
    """
    forest_arrays = forest.to_arrays()
    atomic_savez_compressed(
        paths.forest_npz,
        forest_arrays,
        validator=lambda arrays: ComponentForest.from_arrays(arrays),
    )
    atomic_savez_compressed(
        paths.clg_npz,
        clg_result.arrays,
        validator=lambda arrays: _validate_clg_arrays(arrays, forest),
    )
    atomic_savez_compressed(
        paths.overlap_npz,
        overlap_arrays,
        validator=_validate_overlap_arrays,
    )
    summary = dict(forest_summary)
    summary.update(clg_result.summary)
    atomic_write_json(paths.component_summary_json, summary)
    mark_role_complete(paths, "components")


def _validate_clg_arrays(
    arrays: Mapping[str, np.ndarray],
    forest: ComponentForest,
) -> None:
    """通过完整对象重建校验 CLG 数值 ragged 与 forest 回指。"""
    from .structures import clgs_from_arrays

    clgs_from_arrays(arrays, forest)


def _validate_overlap_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """校验 overlap 局部 occurrence index 与两张同步 value 表。"""
    required = (
        "candidate_occurrence_offsets",
        "overlap_occurrence_index",
        "intersection_voxel_count",
        "occurrence_id",
        "occurrence_voxel_count",
    )
    missing = [field for field in required if field not in arrays]
    if missing:
        raise KeyError(f"overlap.npz 缺少字段: {missing}")
    occurrence_index = np.asarray(arrays["overlap_occurrence_index"])
    intersections = np.asarray(arrays["intersection_voxel_count"])
    if occurrence_index.shape != intersections.shape:
        raise ValueError("overlap_occurrence_index 与 intersection_voxel_count 必须对齐")
    validate_offsets(
        np.asarray(arrays["candidate_occurrence_offsets"]),
        int(occurrence_index.size),
        "candidate_occurrence_offsets",
    )
    occurrence_id = np.asarray(arrays["occurrence_id"])
    occurrence_count = np.asarray(arrays["occurrence_voxel_count"])
    if occurrence_id.shape != occurrence_count.shape:
        raise ValueError("occurrence_id 与 occurrence_voxel_count 必须对齐")
    if occurrence_index.size and (
        int(occurrence_index.min()) < 0
        or int(occurrence_index.max()) >= int(occurrence_id.size)
    ):
        raise ValueError("overlap_occurrence_index 越过 occurrence_id 本地表")
    if bool(np.any(intersections <= 0)):
        raise ValueError("overlap.npz 只保存正交集")
