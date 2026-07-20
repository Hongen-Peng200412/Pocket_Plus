"""从完整图概率的多个冻结阈值层构造并发布只读 26-连通组件森林。

主要入口:
    - `build_component_forest`: 每个实际阈值独立标记 26-连通组件，再按相邻阈值层的 voxel 包含关系连接直接 parent/children。
    - `count_f1_eligible`: 统计冻结 `t_F1` 层可进入后续居中生产的组件数量。
    - `publish_component_artifacts`: 原子发布 `forest.npz`、`clg.npz`、`overlap.npz`、`summary.json`，最后发布 components 完成标记。

候选资格只决定组件能否进入 CLG 或 centered 产物，不会从完整 forest 删除组件。谱系方向为低阈值 parent、较高阈值 child；输入和几何数组的体素轴顺序均为 ZYX。
"""

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


# uint8 原因码到稳定名称的映射；体积检查优先于 80³ BOX 包含检查，因此每个节点只记录一个原因。
INELIGIBLE_REASON: dict[int, str] = {
    0: "eligible",
    1: "below_min_voxels",
    2: "above_max_voxels",
    3: "bbox_not_contained_by_resolved_box",
}


# =========================================================== 给定概率图+阈值, 产生forest ===========================================================
@dataclass(eq=False)
class _RawNode:
    """
    保存构树期间可连接 parent/children 的临时组件节点。

    输入参数:
        - threshold_grid_index: int, 当前节点所属整数阈值 j
        - voxel_global_linear_index: np.ndarray, (K_node,), int64，完整图 ZYX voxel grid 的 C-order 离散线性索引
        - bbox_min_zyx: np.ndarray, (3,), int32，完整图离散 ZYX voxel-index 闭区间 bbox 最小角
        - bbox_max_zyx: np.ndarray, (3,), int32，完整图离散 ZYX voxel-index 闭区间 bbox 最大角
        - centroid_zyx: np.ndarray, (3,), float32，由完整图离散 ZYX voxel indices 求均值得到的连续 voxel-index 质心
        - probability_mean: float, 节点 mask 内完整图概率均值
        - probability_max: float, 节点 mask 内完整图概率最大值
        - candidate_eligible: bool, 节点是否满足候选体积与 bbox 约束
        - ineligible_reason_code: int, 0 表示 eligible，其余值由 `INELIGIBLE_REASON` 解释
        - parent: _RawNode | None, 相邻低阈值层的 direct parent
        - children: list[_RawNode], 相邻高阈值层的 direct children
    """
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


#### 核心函数 ####
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
        - probability_map: float32, (D, H, W), producer 后处理后的完整图概率，三维轴顺序 ZYX，所有数值必须有限。
        - threshold_grid_indices: Sequence[int], 要实际构树的阈值网格整数 j；本函数去重并按 j 从高到低排列————去重后就是树的层数 N_layer 。
        - denominator: int, 概率阈值分母 D_threshold；节点阈值为 `j/D_threshold`，正式值为 32768。
        - min_voxels: int, candidate 最小组件 voxel 数，正式值为 32。
        - max_voxels: int, candidate 最大组件 voxel 数，由 GT occurrence 体积 Q95×3.0 冻结。
        - resolve_box_start: Callable[[np.ndarray, tuple[int, int, int]], Sequence[int]], 接收连续 ZYX voxel-index 质心和完整图 ZYX 形状，返回合法的离散 ZYX BOX 起点；必须与训练及 centered 请求共用同一实现。
        - box_shape_zyx: tuple[int, int, int], centered BOX 的 ZYX voxel 形状，正式值为 `(80, 80, 80)`。

    输出:
        - forest: ComponentForest, parent 指向相邻更低实际阈值包含组件的只读森林；tree_id 和 node_id 由稳定排序确定。
        - summary: dict[str, object], 当前 PDB 的 forest 基础统计；发布时与 CLG 统计合并后写入 `summary.json`，包含以下字段。
            - `denominator`: int, 概率阈值网格分母 D_threshold；实际阈值由 `threshold_grid_index / denominator` 得到。
            - `threshold_grid_indices_descending`: list[int], 长度 N_layer；去重后按高阈值到低阈值排列的实际构树阈值网格整数 j。
            - `connectivity`: int, 固定为 26，表示三维组件连接中心 voxel 及其 26 个邻居。
            - `min_voxels`: int, candidate 允许的最小组件 voxel 数，少于该值的节点记为 `below_min_voxels`。
            - `max_voxels`: int, candidate 允许的最大组件 voxel 数，多于该值的节点记为 `above_max_voxels`。
            - `n_trees`: int, 最低实际阈值层根节点形成的 component tree 总数。
            - `n_nodes`: int, 全部实际阈值层中 ComponentNode 的总数。
            - `ineligible_reason_code`: dict[str, str], 字符串化原因码 `"0"` 至 `"3"` 到 `eligible`、`below_min_voxels`、`above_max_voxels` 和 `bbox_not_contained_by_resolved_box` 的稳定映射。
            - `layers`: list[dict[str, int]], 长度 N_layer；与 `threshold_grid_indices_descending` 逐层对齐，每层的 `n_eligible` 与三个 ineligible 原因计数之和等于 `n_nodes`。
                - `threshold_grid_index`: int, 当前层的阈值网格整数 j。
                - `n_nodes`: int, 当前阈值层的 26-连通组件总数。
                - `n_eligible`: int, 当前层满足体积与 centered BOX 包含约束的 candidate 节点数。
                - `n_below_min_voxels`: int, 当前层因组件 voxel 数少于 `min_voxels` 而不可作为 candidate 的节点数。
                - `n_above_max_voxels`: int, 当前层因组件 voxel 数多于 `max_voxels` 而不可作为 candidate 的节点数。
                - `n_bbox_not_contained_by_resolved_box`: int, 当前层体积合格但组件包围盒无法被解析出的 centered BOX 完整容纳的节点数。
    """
    # float32, (D, H, W), producer 后处理后的完整图概率；三维轴顺序 ZYX。
    probability = np.asarray(probability_map, dtype=np.float32)
    if probability.ndim != 3 or not bool(np.all(np.isfinite(probability))):
        raise ValueError("probability_map 必须是有限值三维数组")
    if denominator <= 0 or min_voxels <= 0 or max_voxels < min_voxels:
        raise ValueError("denominator/min_voxels/max_voxels 配置不合法")
    # list[int], 长度 N_layer；去重后从高到低的阈值网格整数 j，重复阈值不制造重复 forest 层。
    thresholds = sorted({int(value) for value in threshold_grid_indices}, reverse=True)
    if not thresholds or thresholds[0] > denominator or thresholds[-1] < 0:
        raise ValueError("threshold_grid_indices 必须是 [0,denominator] 内的非空集合")
    full_shape = tuple(int(value) for value in probability.shape)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if box_shape.shape != (3,) or bool(np.any(np.asarray(full_shape) < box_shape)):
        raise ValueError("完整图三轴必须不小于 centered BOX")

    # bool, (3, 3, 3), 三维中心 voxel 及其 26 个邻居的连通结构。
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    # list[list[_RawNode]], 长度 N_layer；`layers[i]` 保存阈值层 i 的全部临时组件节点。
    layers: list[list[_RawNode]] = []
    # list[int array], 长度 N_layer；`label_layers[i]` 为 `(D, H, W)` 的 1-based 组件标签图(0 表示背景, 从1开始每个数字代表一个连通分支)，与 `layers[i]` 按标签编号对齐。
    label_layers: list[np.ndarray] = []
    # list[dict[str, int]], 长度 N_layer；每项记录当前阈值、组件总数、eligible 数和各原因码计数。
    layer_stats: list[dict[str, int]] = []
    for threshold_grid_index in thresholds:
        threshold = float(threshold_grid_index) / float(denominator)
        # labels: int array, (D, H, W), 当前阈值二值图的 1-based 26-连通组件标签；0 表示背景, 从1开始每个数字代表一个连通分支。
        # component_count: int, 当前阈值层的 26-连通组件数。
        labels, component_count = ndimage.label(probability >= threshold, structure=structure)
        # list[tuple[slice, slice, slice] | None], 长度 `component_count`；第 `label_id - 1` 项给出标签 `label_id` 在完整 `(D, H, W)` ZYX 标签图中的最小包围盒切片 `(z_slice, y_slice, x_slice)`，缺失标签时为 None，后续用它裁剪该组件的局部 mask。
        # “最小包围盒切片”是用三个切片范围把一个组件的全部 voxel 包住，并且每个轴的范围尽可能小。
        # 例：若组件 voxel 范围为 z: 2 到 4、y: 10 到 12、x: 5 到 8，则 `(z_slice, y_slice, x_slice) = (slice(2, 5), slice(10, 13), slice(5, 9))`。
        # Python 切片的结束位置不包含，因此 `slice(2, 5)` 实际包含索引 2、3、4。
        objects = ndimage.find_objects(labels, max_label=int(component_count))
        nodes: list[_RawNode] = []
        reason_counts = {reason: 0 for reason in INELIGIBLE_REASON.values()}
        for label_id, object_slices in enumerate(objects, start=1):
            if object_slices is None:
                continue
            # bool, (d, h, w), 当前组件包围盒内的局部 mask；`object_slices` 给出该局部网格在完整图中的 ZYX 区间。
            local_mask = labels[object_slices] == label_id
            # int64, (K_node, 3), 当前组件 K_node 个 voxel 在局部包围盒中的离散 ZYX 索引。
            local_zyx = np.argwhere(local_mask)
            start_zyx = np.asarray([axis_slice.start for axis_slice in object_slices], dtype=np.int64)
            # int64, (K_node, 3), 加上包围盒起点后得到的完整图离散 ZYX voxel index。
            voxel_zyx = local_zyx + start_zyx[None, :]
            # int64, (K_node,), 当前组件 voxel 在完整 ZYX 网格中的 C-order 离散线性索引，升序且唯一。
            linear_index = np.ravel_multi_index(voxel_zyx.T, full_shape).astype(np.int64, copy=False)
            linear_index.sort()
            bbox_min = voxel_zyx.min(axis=0).astype(np.int32)
            bbox_max = voxel_zyx.max(axis=0).astype(np.int32)
            centroid = voxel_zyx.mean(axis=0, dtype=np.float64).astype(np.float32)
            # float32, (K_node,), 当前组件每个 voxel 的完整图融合概率，与 `linear_index` 逐 voxel 对齐。
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
        # dict[int, _RawNode], 相邻低阈值层的 1-based label id 到临时节点的映射。
        low_by_label = {label_id: node for label_id, node in enumerate(low_nodes, start=1)}
        low_flat = low_labels.reshape(-1)
        for child in high_nodes:
            # int array, (K_label,), 高阈值 child 的全部 voxel 在相邻低阈值 label 图中命中的唯一标签集合；0 表示背景，合法单调关系必须恰好命中一个正 parent 标签。
            containing_labels = np.unique(low_flat[child.voxel_global_linear_index])
            if containing_labels.size != 1 or int(containing_labels[0]) <= 0:
                raise RuntimeError("阈值单调性被破坏：高阈值组件没有唯一相邻低阈值 parent")
            parent = low_by_label[int(containing_labels[0])]
            child.parent = parent
            parent.children.append(child)

    # list[_RawNode], 没有相邻更低阈值 parent 的根节点；每个 root 形成一棵独立 tree。
    roots = [node for layer in layers for node in layer if node.parent is None]
    roots.sort(key=_node_sort_key)
    trees: list[ComponentTree] = []
    for tree_id, root in enumerate(roots):
        # list[_RawNode], parent-before-children 的稳定深度优先顺序；列表位置同时成为当前 tree 的连续 node_id。
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
    # dict[str, object], 可直接写入 `summary.json` 的 forest 基础契约；CLG 统计在发布时并入同一顶层映射。
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
    """
    按体积优先、bbox 其次返回稳定的 candidate ineligible reason code。

    输入参数:
        - voxel_count: int, 当前组件的完整图 voxel 数
        - bbox_min_zyx: np.ndarray, (3,), 完整图离散 ZYX voxel-index 闭区间 bbox 最小角
        - bbox_max_zyx: np.ndarray, (3,), 完整图离散 ZYX voxel-index 闭区间 bbox 最大角
        - centroid_zyx: np.ndarray, (3,), 完整图连续 ZYX voxel-index 质心
        - full_shape_zyx: tuple[int,int,int], (3,), 完整图 ZYX voxel-grid shape
        - box_shape_zyx: np.ndarray, (3,), 输出 BOX 的 ZYX voxel-grid shape
        - min_voxels: int, candidate 最小 voxel 数
        - max_voxels: int, candidate 最大 voxel 数
        - resolve_box_start: Callable, 把完整图 voxel-index 质心解析为离散 ZYX voxel-index BOX corner 起点

    输出:
        - reason_code: int, 0=eligible、1=过小、2=过大、3=bbox 不能被解析后的 BOX 包含
    """
    if voxel_count < min_voxels:
        return 1
    if voxel_count > max_voxels:
        return 2
    resolved_start = np.asarray(resolve_box_start(centroid_zyx, full_shape_zyx), dtype=np.int64)
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
    """
    为 root/children 提供与平台无关的稳定排序键: (-int(node.threshold_grid_index), -float(node.probability_mean), first_voxel) 。

    输入参数:
        - node: _RawNode, 要排序的临时节点

    输出:
        - key: tuple[int,float,int], 依次按高阈值、高平均概率、最小完整图线性 voxel index 排序
    """
    first_voxel = (
        int(node.voxel_global_linear_index[0])
        if node.voxel_global_linear_index.size
        else -1
    )
    return (-int(node.threshold_grid_index), -float(node.probability_mean), first_voxel)

def _deterministic_depth_first(root: _RawNode) -> list[_RawNode]:
    """
    以 parent-before-children 的确定性顺序展开一棵临时树。

    输入参数:
        - root: _RawNode, 当前临时树的唯一 root

    输出:
        - nodes: list[_RawNode], parent-before-children 的稳定 DFS 顺序
    """
    result: list[_RawNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(sorted(node.children, key=_node_sort_key)))
    return result





# =========================================================== 其余小工具函数: 发布/校验 ===========================================================
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
        - paths: Stage1ArtifactPaths, 当前 producer/split/PDB 的正式 forest、CLG、overlap、summary 与状态路径。
        - forest: ComponentForest, 要编码为 `forest.npz` 的原始只读组件森林。
        - clg_result: CLGEnumerationResult, 要编码为 `clg.npz` 的成功 CLG，以及并入 summary 的枚举计数。
        - overlap_arrays: Mapping[str, np.ndarray], `overlap.npz` 字段；`candidate_occurrence_offsets` 同步切分 occurrence 局部行号与正交集 voxel 计数。
        - forest_summary: Mapping[str, object], `build_component_forest` 返回的阈值层、eligibility 与原因码统计。

    输出:
        - None, 依次原子发布 `forest.npz`、`clg.npz`、`overlap.npz` 和 `summary.json`，前三个 NPZ 均重读校验；全部成功后才发布 components `_COMPLETE`。
    """
    # dict[str, np.ndarray], `forest.npz` 的节点主表、children ragged 表和 voxel membership ragged 表。
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
    # dict[str, object], forest 基础统计与 CLG 枚举统计的单一 `summary.json` 顶层映射。
    summary = dict(forest_summary)
    summary.update(clg_result.summary)
    atomic_write_json(paths.component_summary_json, summary)
    mark_role_complete(paths, "components")

def _validate_clg_arrays(
    arrays: Mapping[str, np.ndarray],
    forest: ComponentForest,
) -> None:
    """
    通过完整对象重建校验 CLG 数值 ragged 与 forest 回指。

    输入参数:
        - arrays: Mapping[str,np.ndarray], `clg.npz` 的全部数值字段
        - forest: ComponentForest, seed/oldest/candidate identity 的权威来源

    输出:
        - None: `clgs_from_arrays` 可完整恢复且阈值回指一致时返回
    """
    from .structures import clgs_from_arrays

    clgs_from_arrays(arrays, forest)

def _validate_overlap_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """
    校验 overlap 局部 occurrence index 与两张同步 value 表。

    输入参数:
        - arrays: Mapping[str, np.ndarray], `overlap.npz` 的 `candidate_occurrence_offsets/overlap_occurrence_index/intersection_voxel_count/occurrence_id/occurrence_voxel_count` 字段。

    输出:
        - None: candidate offsets、occurrence 局部行号、正交集计数与 occurrence 主表全部对齐时返回
    """
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
