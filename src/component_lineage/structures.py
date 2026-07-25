"""组件森林、树节点、CLG 与只复制拓扑的工作树对象。

``ComponentNode/Tree/Forest`` 表示完整、只读的正式谱系；``WorkingTree`` 只复制
parent/children identity 和 active 状态，供一次 CLG 扫描删除节点而不改正式 forest。
``to_arrays/from_arrays`` 则在对象树与无 pickle ragged NPZ 之间转换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.artifacts.io import validate_offsets


@dataclass(eq=False)
class ComponentNode:
    """
    表示完整图组件森林中的一个只读组件节点。

    输入参数:
        - tree_id: int, 当前 PDB 内树身份
        - node_id: int, 当前树内节点身份
        - threshold_grid_index: int, 阈值整数 j，物理阈值为 j/32768
        - threshold_value: float, 便于冷读的物理阈值
        - voxel_global_linear_index: np.ndarray, (K,), int64，完整图 ZYX voxel grid 的 C-order 离散线性索引
        - bbox_min_zyx: np.ndarray, (3,), int32，完整图离散 ZYX voxel-index 闭区间 bbox 最小角
        - bbox_max_zyx: np.ndarray, (3,), int32，完整图离散 ZYX voxel-index 闭区间 bbox 最大角
        - centroid_zyx: np.ndarray, (3,), float32，由完整图离散 ZYX voxel indices 求均值得到的连续 voxel-index 质心
        - probability_mean: float, mask 内完整图概率均值
        - probability_max: float, mask 内完整图概率最大值
        - candidate_eligible: bool, 是否满足体积与 80³ bbox 规则
        - ineligible_reason_code: int, 0 表示 eligible，其余值由 summary 解释

    属性:
        - parent: ComponentNode | None, 唯一 direct parent，root 为 None
        - children: list[ComponentNode], 全部 direct children
    """

    tree_id: int
    node_id: int
    threshold_grid_index: int
    threshold_value: float
    voxel_global_linear_index: np.ndarray
    bbox_min_zyx: np.ndarray
    bbox_max_zyx: np.ndarray
    centroid_zyx: np.ndarray
    probability_mean: float
    probability_max: float
    candidate_eligible: bool
    ineligible_reason_code: int
    parent: "ComponentNode | None" = field(default=None, repr=False)
    children: list["ComponentNode"] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """
        规范化节点数组 dtype，并把几何与 voxel payload 标记为只读。

        输出:
            - None: 原地规范化四个 NumPy 数组；不建立 parent/children 拓扑
        """
        self.voxel_global_linear_index = np.asarray(
            self.voxel_global_linear_index, dtype=np.int64
        )
        self.bbox_min_zyx = np.asarray(self.bbox_min_zyx, dtype=np.int32)
        self.bbox_max_zyx = np.asarray(self.bbox_max_zyx, dtype=np.int32)
        self.centroid_zyx = np.asarray(self.centroid_zyx, dtype=np.float32)
        for array in (
            self.voxel_global_linear_index,
            self.bbox_min_zyx,
            self.bbox_max_zyx,
            self.centroid_zyx,
        ):
            array.setflags(write=False)

    @property
    def voxel_count(self) -> int:
        """
        返回当前组件 mask 的 voxel 数。

        输出:
            - voxel_count: int, `voxel_global_linear_index` 的长度 K
        """
        return int(self.voxel_global_linear_index.size)

    def ancestors(self, include_self: bool = True) -> list["ComponentNode"]:
        """
        返回从当前节点向低阈值方向排列的祖先链。

        输入参数:
            - include_self: bool, 是否把当前节点放在首项

        输出:
            - nodes: list[ComponentNode], 当前节点到 root 的有序链
        """
        nodes: list[ComponentNode] = [self] if include_self else []
        cursor = self.parent
        while cursor is not None:
            nodes.append(cursor)
            cursor = cursor.parent
        return nodes

    def subtree(self, include_self: bool = True) -> list["ComponentNode"]:
        """
        返回当前节点向高阈值方向的完整子树，顺序为确定性的深度优先序。

        输入参数:
            - include_self: bool, 是否把当前节点作为首项

        输出:
            - nodes: list[ComponentNode], 子树中的组件节点
        """
        nodes: list[ComponentNode] = []
        stack = [self]
        while stack:
            node = stack.pop()
            if include_self or node is not self:
                nodes.append(node)
            stack.extend(reversed(node.children))
        return nodes

    def sisters(self) -> list["ComponentNode"]:
        """
        返回与当前节点共享 direct parent 的其它节点。

        输出:
            - sisters: list[ComponentNode], 按 parent.children 顺序排列的同层其它节点；root 返回空列表
        """
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child is not self]

    def is_ancestor_of(self, other: "ComponentNode") -> bool:
        """
        判断当前节点是否是 `other` 的祖先，节点自身也算祖先。

        输入参数:
            - other: ComponentNode, 要检查的另一节点

        输出:
            - is_ancestor: bool, 同树且当前节点出现在 `other` 到 root 的 parent 链上时为 True
        """
        if self.tree_id != other.tree_id:
            return False
        cursor: ComponentNode | None = other
        while cursor is not None:
            if cursor is self:
                return True
            cursor = cursor.parent
        return False


@dataclass
class ComponentTree:
    """
    表示一棵具有唯一 root 的组件树。

    输入参数:
        - tree_id: int, 当前 PDB 内树身份
        - nodes: Sequence[ComponentNode], 当前树全部节点
    """

    tree_id: int
    nodes: Sequence[ComponentNode]

    def __post_init__(self) -> None:
        """
        建立 node_id 索引，定位唯一 root 并校验整棵树拓扑。

        输出:
            - None: 原地把 `nodes` 固定为 tuple，并建立 `_by_id/root`
        """
        self.nodes = tuple(self.nodes)
        self._by_id = {node.node_id: node for node in self.nodes}
        if len(self._by_id) != len(self.nodes):
            raise ValueError(f"tree_id={self.tree_id} 内 node_id 不唯一")
        if any(node.tree_id != self.tree_id for node in self.nodes):
            raise ValueError("ComponentTree 不能混入其它 tree_id 的节点")
        roots = [node for node in self.nodes if node.parent is None]
        if len(roots) != 1:
            raise ValueError(f"tree_id={self.tree_id} 必须恰有一个 root，实际 {len(roots)}")
        self.root = roots[0]
        self._validate_topology()

    def _validate_topology(self) -> None:
        """
        校验 parent/children 双向一致、无环且全部节点从 root 可达。

        输出:
            - None: 拓扑合法时返回；发现跨树 child、双向关系不一致、环或不可达节点时直接报错
        """
        for node in self.nodes:
            for child in node.children:
                if child.parent is not node:
                    raise ValueError("parent/children 双向关系不一致")
                if child.tree_id != self.tree_id:
                    raise ValueError("children 不得跨 tree")
        visited: set[int] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.node_id in visited:
                raise ValueError("组件树包含环或重复 child")
            visited.add(node.node_id)
            stack.extend(node.children)
        if visited != set(self._by_id):
            raise ValueError("组件树存在 root 不可达节点")

    def node(self, node_id: int) -> ComponentNode:
        """
        按当前树内 node_id 返回节点。

        输入参数:
            - node_id: int, 当前 tree 内唯一节点 identity

        输出:
            - node: ComponentNode, 对应的权威节点对象
        """
        return self._by_id[int(node_id)]

    def lca(self, left: ComponentNode, right: ComponentNode) -> ComponentNode:
        """
        返回两个同树节点的最低公共祖先。

        输入参数:
            - left: ComponentNode, 第一个节点
            - right: ComponentNode, 第二个节点

        输出:
            - lca_node: ComponentNode, 离两个节点最近的共同低阈值祖先
        """
        if left.tree_id != self.tree_id or right.tree_id != self.tree_id:
            raise ValueError("LCA 的两个节点必须属于当前树")
        left_ancestors = {node.node_id for node in left.ancestors()}
        cursor: ComponentNode | None = right
        while cursor is not None:
            if cursor.node_id in left_ancestors:
                return cursor
            cursor = cursor.parent
        raise RuntimeError("合法树中的同树节点必有公共 root")


@dataclass
class ComponentForest:
    """
    表示一个 PDB 的多阈值组件森林。

    输入参数:
        - trees: Sequence[ComponentTree], 当前 PDB 的全部互不相交树
    """

    trees: Sequence[ComponentTree]

    def __post_init__(self) -> None:
        """
        固定 forest 的 tree 顺序并建立 tree_id 索引。

        输出:
            - None: 原地把 `trees` 固定为 tuple，并建立 `_by_tree_id`
        """
        self.trees = tuple(self.trees)
        self._by_tree_id = {tree.tree_id: tree for tree in self.trees}
        if len(self._by_tree_id) != len(self.trees):
            raise ValueError("ComponentForest 的 tree_id 不唯一")

    @property
    def nodes(self) -> tuple[ComponentNode, ...]:
        """
        按 tree_id/node_id 排序返回森林全部节点。

        输出:
            - nodes: tuple[ComponentNode,...], 长度 N_node，按 `(tree_id,node_id)` 稳定排序
        """
        return tuple(
            node
            for tree in sorted(self.trees, key=lambda value: value.tree_id)
            for node in sorted(tree.nodes, key=lambda value: value.node_id)
        )

    def tree(self, tree_id: int) -> ComponentTree:
        """
        按 tree_id 返回组件树。

        输入参数:
            - tree_id: int, 当前 PDB forest 内唯一 tree identity

        输出:
            - tree: ComponentTree, 对应的权威组件树
        """
        return self._by_tree_id[int(tree_id)]

    def node(self, tree_id: int, node_id: int) -> ComponentNode:
        """
        按 `(tree_id,node_id)` 返回唯一组件节点。

        输入参数:
            - tree_id: int, 当前 PDB forest 内唯一 tree identity
            - node_id: int, 当前 tree 内唯一 node identity

        输出:
            - node: ComponentNode, 对应的权威组件节点
        """
        return self.tree(tree_id).node(node_id)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """
        编码为 BOX 契约规定的无 pickle 数值 ragged 字段。

        输出:
            - arrays: dict[str, np.ndarray], `forest.npz` 的全部字段；
              `children_offsets` 切分 `children_node_id`，`node_voxel_offsets`
              切分 `node_voxel_global_linear_index`
        """
        # tuple[ComponentNode,...], 长度 N_node，按 (tree_id,node_id) 固定所有主表行序。
        nodes = self.nodes
        # list[np.ndarray[int32]], 长度 N_node，每项是同树 direct children 的 node_id。
        children_rows = [
            np.asarray([child.node_id for child in node.children], dtype=np.int32)
            for node in nodes
        ]
        # list[np.ndarray[int64]], 长度 N_node，每项是节点完整图 C-order voxel 索引。
        voxel_rows = [node.voxel_global_linear_index for node in nodes]
        # 两个 (N_node+1,) offsets 分别切分 children_node_id 与 node_voxel_global_linear_index。
        children_offsets = _offsets_from_rows(children_rows)
        voxel_offsets = _offsets_from_rows(voxel_rows)
        return {
            "tree_id": np.asarray([node.tree_id for node in nodes], dtype=np.int32),
            "node_id": np.asarray([node.node_id for node in nodes], dtype=np.int32),
            "threshold_grid_index": np.asarray(
                [node.threshold_grid_index for node in nodes], dtype=np.int32
            ),
            "threshold_value": np.asarray(
                [node.threshold_value for node in nodes], dtype=np.float32
            ),
            "parent_node_id": np.asarray(
                [-1 if node.parent is None else node.parent.node_id for node in nodes],
                dtype=np.int32,
            ),
            "children_offsets": children_offsets,
            "children_node_id": _concatenate_rows(children_rows, np.dtype(np.int32)),
            "node_voxel_offsets": voxel_offsets,
            "node_voxel_global_linear_index": _concatenate_rows(
                voxel_rows, np.dtype(np.int64)
            ),
            "voxel_count": np.asarray([node.voxel_count for node in nodes], dtype=np.int32),
            "bbox_min_zyx": np.asarray(
                [node.bbox_min_zyx for node in nodes], dtype=np.int32
            ).reshape(len(nodes), 3),
            "bbox_max_zyx": np.asarray(
                [node.bbox_max_zyx for node in nodes], dtype=np.int32
            ).reshape(len(nodes), 3),
            "centroid_zyx": np.asarray(
                [node.centroid_zyx for node in nodes], dtype=np.float32
            ).reshape(len(nodes), 3),
            "probability_mean": np.asarray(
                [node.probability_mean for node in nodes], dtype=np.float32
            ),
            "probability_max": np.asarray(
                [node.probability_max for node in nodes], dtype=np.float32
            ),
            "candidate_eligible": np.asarray(
                [node.candidate_eligible for node in nodes], dtype=np.bool_
            ),
            "ineligible_reason_code": np.asarray(
                [node.ineligible_reason_code for node in nodes], dtype=np.uint8
            ),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "ComponentForest":
        """
        从 `forest.npz` 数值字段重建专属树对象。

        输入参数:
            - arrays: Mapping[str, np.ndarray], 完整 forest 归档字段

        输出:
            - forest: ComponentForest, parent/children 已直接连接的只读对象森林
        """
        required = (
            "tree_id",
            "node_id",
            "threshold_grid_index",
            "threshold_value",
            "parent_node_id",
            "children_offsets",
            "children_node_id",
            "node_voxel_offsets",
            "node_voxel_global_linear_index",
            "voxel_count",
            "bbox_min_zyx",
            "bbox_max_zyx",
            "centroid_zyx",
            "probability_mean",
            "probability_max",
            "candidate_eligible",
            "ineligible_reason_code",
        )
        missing = [field for field in required if field not in arrays]
        if missing:
            raise KeyError(f"forest.npz 缺少字段: {missing}")
        # np.ndarray[int32], (N_node,), forest 主表的 tree identity。
        tree_ids = np.asarray(arrays["tree_id"], dtype=np.int32)
        # np.ndarray[int32], (N_node,), 在各 tree 内唯一的 node identity。
        node_ids = np.asarray(arrays["node_id"], dtype=np.int32)
        n_node = int(tree_ids.size)
        if node_ids.shape != (n_node,):
            raise ValueError("tree_id 与 node_id 长度不一致")
        for field in ("bbox_min_zyx", "bbox_max_zyx", "centroid_zyx"):
            if np.asarray(arrays[field]).shape != (n_node, 3):
                raise ValueError(f"{field} 必须为 [N_node,3]")
        child_offsets = np.asarray(arrays["children_offsets"])
        # child_offsets 切 children_node_id；voxel_offsets 切 node_voxel_global_linear_index。
        child_values = np.asarray(arrays["children_node_id"])
        voxel_offsets = np.asarray(arrays["node_voxel_offsets"])
        voxel_values = np.asarray(arrays["node_voxel_global_linear_index"])
        validate_offsets(child_offsets, int(child_values.size), "children_offsets")
        validate_offsets(voxel_offsets, int(voxel_values.size), "node_voxel_offsets")
        if child_offsets.shape != (n_node + 1,) or voxel_offsets.shape != (n_node + 1,):
            raise ValueError("forest offsets 长度必须为 N_node+1")

        nodes_by_key: dict[tuple[int, int], ComponentNode] = {}
        ordered_nodes: list[ComponentNode] = []
        for row in range(n_node):
            node = ComponentNode(
                tree_id=int(tree_ids[row]),
                node_id=int(node_ids[row]),
                threshold_grid_index=int(np.asarray(arrays["threshold_grid_index"])[row]),
                threshold_value=float(np.asarray(arrays["threshold_value"])[row]),
                voxel_global_linear_index=voxel_values[
                    int(voxel_offsets[row]) : int(voxel_offsets[row + 1])
                ],
                bbox_min_zyx=np.asarray(arrays["bbox_min_zyx"])[row],
                bbox_max_zyx=np.asarray(arrays["bbox_max_zyx"])[row],
                centroid_zyx=np.asarray(arrays["centroid_zyx"])[row],
                probability_mean=float(np.asarray(arrays["probability_mean"])[row]),
                probability_max=float(np.asarray(arrays["probability_max"])[row]),
                candidate_eligible=bool(np.asarray(arrays["candidate_eligible"])[row]),
                ineligible_reason_code=int(np.asarray(arrays["ineligible_reason_code"])[row]),
            )
            key = (node.tree_id, node.node_id)
            if key in nodes_by_key:
                raise ValueError(f"重复 forest node identity={key}")
            nodes_by_key[key] = node
            ordered_nodes.append(node)

        parent_ids = np.asarray(arrays["parent_node_id"], dtype=np.int32)
        for row, node in enumerate(ordered_nodes):
            parent_id = int(parent_ids[row])
            if parent_id >= 0:
                node.parent = nodes_by_key[(node.tree_id, parent_id)]
            expected_children = [
                nodes_by_key[(node.tree_id, int(child_id))]
                for child_id in child_values[
                    int(child_offsets[row]) : int(child_offsets[row + 1])
                ]
            ]
            node.children.extend(expected_children)
        trees = [
            ComponentTree(tree_id=int(tree_id), nodes=[n for n in ordered_nodes if n.tree_id == tree_id])
            for tree_id in sorted(set(int(value) for value in tree_ids.tolist()))
        ]
        forest = cls(trees=trees)
        encoded = forest.to_arrays()
        if not np.array_equal(encoded["voxel_count"], np.asarray(arrays["voxel_count"])):
            raise ValueError("voxel_count 与 node voxel ragged 长度不一致")
        return forest


@dataclass(frozen=True)
class CLG:
    """
    表示一个成功的 Candidate Lineage Group。

    输入参数:
        - CLG_id: int, 当前 PDB 内连续本地身份
        - tree_id: int, 所属原始 component tree
        - seed_node: ComponentNode, 本次扫描选中的唯一 active seed
        - oldest_node: ComponentNode, 覆盖全部 candidate masks 的唯一最老候选
        - candidate_nodes: Sequence[ComponentNode], 按确定性枚举顺序保存的候选
    """

    CLG_id: int
    tree_id: int
    seed_node: ComponentNode
    oldest_node: ComponentNode
    candidate_nodes: Sequence[ComponentNode]

    def __post_init__(self) -> None:
        """
        固定 CLG 候选顺序并校验 seed、oldest 与 candidate 的同树覆盖关系。

        输出:
            - None: 原地把 `candidate_nodes` 固定为 tuple；身份或拓扑关系不合法时直接报错
        """
        candidates = tuple(self.candidate_nodes)
        object.__setattr__(self, "candidate_nodes", candidates)
        if len(candidates) == 0:
            raise ValueError("成功 CLG 至少包含一个 candidate")
        if len({node.node_id for node in candidates}) != len(candidates):
            raise ValueError("同一 CLG 的 candidate_node_id 必须唯一")
        if any(node.tree_id != self.tree_id for node in candidates):
            raise ValueError("CLG candidates 必须属于同一 tree")
        if self.seed_node not in candidates or self.oldest_node not in candidates:
            raise ValueError("CLG 的 seed_node 与 oldest_node 都必须是 candidate")
        if any(not self.oldest_node.is_ancestor_of(node) for node in candidates):
            raise ValueError("CLG_oldest_node 必须覆盖/祖先于全部 candidates")


@dataclass
class _WorkingNode:
    """
    仅保存拓扑身份、active 状态和原节点回指的工作节点。

    输入参数:
        - original: ComponentNode, 权威只读节点回指
        - parent_id: int | None, 原 tree 内 direct parent node_id；root 为 None
        - children_ids: tuple[int,...], 原 tree 内 direct children node_id
        - active: bool, 当前工作树是否仍保留该节点
    """

    original: ComponentNode
    parent_id: int | None
    children_ids: tuple[int, ...]
    active: bool = True


class WorkingTree:
    """
    为一次 CLG 枚举复制拓扑和 active 状态，不复制 voxel payload。

    输入参数:
        - original_tree: ComponentTree, 始终保持只读的原组件树
    """

    def __init__(self, original_tree: ComponentTree) -> None:
        """
        从只读正式树复制轻量拓扑与 active 状态。

        输入参数:
            - original_tree: ComponentTree, voxel payload 保持只读且不复制的权威组件树

        输出:
            - None: 原地建立 `node_id -> _WorkingNode` 工作索引，所有节点初始 active
        """
        self.original_tree = original_tree
        # dict[int,_WorkingNode], node_id -> 轻量拓扑副本；voxel/probability 仍回指 original。
        self._nodes = {
            node.node_id: _WorkingNode(
                original=node,
                parent_id=None if node.parent is None else node.parent.node_id,
                children_ids=tuple(child.node_id for child in node.children),
            )
            for node in original_tree.nodes
        }

    def is_active(self, node: ComponentNode | int) -> bool:
        """
        返回原节点或 node_id 在当前工作副本中是否 active。

        输入参数:
            - node: ComponentNode | int, 原节点对象或当前 tree 内 node_id

        输出:
            - is_active: bool, 节点尚未被 `delete_D` 删除时为 True
        """
        node_id = node.node_id if isinstance(node, ComponentNode) else int(node)
        return self._nodes[node_id].active

    def active_ancestors(self, node: ComponentNode) -> list[ComponentNode]:
        """
        返回 `Ancestors_W(node)`，包含 node 且只返回当前 active 节点。

        输入参数:
            - node: ComponentNode, 当前 tree 中的原节点

        输出:
            - ancestors: list[ComponentNode], 从 node 向低阈值 root 方向排列的 active 节点
        """
        result: list[ComponentNode] = []
        cursor_id: int | None = node.node_id
        while cursor_id is not None:
            working_node = self._nodes[cursor_id]
            if working_node.active:
                result.append(working_node.original)
            cursor_id = working_node.parent_id
        return result

    def active_subtree(self, node: ComponentNode) -> list[ComponentNode]:
        """
        返回 `Subtree_W(node)`，包含 node 且只返回当前 active 节点。

        输入参数:
            - node: ComponentNode, 当前 tree 中的原节点

        输出:
            - subtree: list[ComponentNode], 从 node 向高阈值 children 方向按稳定 DFS 排列的 active 节点
        """
        result: list[ComponentNode] = []
        stack = [node.node_id]
        while stack:
            node_id = stack.pop()
            working_node = self._nodes[node_id]
            if working_node.active:
                result.append(working_node.original)
            stack.extend(reversed(working_node.children_ids))
        return result

    def active_children(self, node: ComponentNode) -> list[ComponentNode]:
        """
        返回当前节点仍 active 的 direct children。

        输入参数:
            - node: ComponentNode, 当前 tree 中的原节点

        输出:
            - children: list[ComponentNode], 按原 children 顺序排列的 active direct children
        """
        return [
            self._nodes[node_id].original
            for node_id in self._nodes[node.node_id].children_ids
            if self._nodes[node_id].active
        ]

    def active_parent(self, node: ComponentNode) -> ComponentNode | None:
        """
        返回当前节点仍 active 的 direct parent。

        输入参数:
            - node: ComponentNode, 当前 tree 中的原节点

        输出:
            - parent: ComponentNode | None, active direct parent；root 或 parent 已删除时为 None
        """
        parent_id = self._nodes[node.node_id].parent_id
        if parent_id is None or not self._nodes[parent_id].active:
            return None
        return self._nodes[parent_id].original

    def active_sisters(self, node: ComponentNode) -> list[ComponentNode]:
        """
        返回共享原 direct parent 且当前 active 的其它 children。

        输入参数:
            - node: ComponentNode, 当前 tree 中的原节点

        输出:
            - sisters: list[ComponentNode], 按原 children 顺序排列的 active sisters；root 返回空列表
        """
        parent_id = self._nodes[node.node_id].parent_id
        if parent_id is None:
            return []
        return [
            self._nodes[child_id].original
            for child_id in self._nodes[parent_id].children_ids
            if child_id != node.node_id and self._nodes[child_id].active
        ]

    def delete_D(self, seed_node: ComponentNode) -> tuple[ComponentNode, ...]:
        """
        计算并删除 `D_W(g)=Ancestors_W(g)∪Subtree_W(g)`。

        输入参数:
            - seed_node: ComponentNode, 当前 CLG 尝试开始时选中的唯一 seed `g`

        输出:
            - removed: tuple[ComponentNode, ...], 本次从 active 变为 removed 的原节点
        """
        # list[ComponentNode], 祖先链与子树可能都含 seed，下一步按 node_id 去重。
        ordered = self.active_ancestors(seed_node) + self.active_subtree(seed_node)
        # dict[int,ComponentNode], 保持首次出现顺序的 D_W(g) 唯一节点集。
        unique: dict[int, ComponentNode] = {node.node_id: node for node in ordered}
        for node_id in unique:
            self._nodes[node_id].active = False
        return tuple(unique.values())

    def active_nodes(self) -> tuple[ComponentNode, ...]:
        """
        返回当前工作副本的全部 active 原节点。

        输出:
            - nodes: tuple[ComponentNode,...], 按 original_tree.nodes 顺序排列的 active 节点
        """
        return tuple(
            working_node.original
            for working_node in self._nodes.values()
            if working_node.active
        )


def clgs_to_arrays(clgs: Sequence[CLG]) -> dict[str, np.ndarray]:
    """
    把成功 CLG 编码为 `clg.npz` 数值 ragged 字段。

    输入参数:
        - clgs: Sequence[CLG], 按正式来源顺序排列的成功 CLG

    输出:
        - arrays: dict[str, np.ndarray], `candidate_offsets` 同时切分
          `candidate_node_id` 与 `candidate_threshold_grid_index`
    """
    # list[np.ndarray[int32]], 长度 N_CLG，每项按枚举顺序保存该 CLG 的 candidate node_id。
    rows = [
        np.asarray([node.node_id for node in clg.candidate_nodes], dtype=np.int32)
        for clg in clgs
    ]
    threshold_rows = [
        np.asarray(
            [node.threshold_grid_index for node in clg.candidate_nodes], dtype=np.int32
        )
        for clg in clgs
    ]
    return {
        "CLG_id": np.asarray([clg.CLG_id for clg in clgs], dtype=np.int32),
        "tree_id": np.asarray([clg.tree_id for clg in clgs], dtype=np.int32),
        "CLG_seed_node_id": np.asarray(
            [clg.seed_node.node_id for clg in clgs], dtype=np.int32
        ),
        "CLG_oldest_node_id": np.asarray(
            [clg.oldest_node.node_id for clg in clgs], dtype=np.int32
        ),
        "candidate_offsets": _offsets_from_rows(rows),
        "candidate_node_id": _concatenate_rows(rows, np.dtype(np.int32)),
        "candidate_threshold_grid_index": _concatenate_rows(
            threshold_rows, np.dtype(np.int32)
        ),
    }


def clgs_from_arrays(
    arrays: Mapping[str, np.ndarray],
    forest: ComponentForest,
) -> tuple[CLG, ...]:
    """
    从 `clg.npz` 和原只读森林恢复 CLG 对象。

    输入参数:
        - arrays: Mapping[str, np.ndarray], `clg.npz` 数值字段
        - forest: ComponentForest, candidate/seed/oldest 的权威节点来源

    输出:
        - clgs: tuple[CLG, ...], 按 `CLG_id` 来源顺序恢复的对象
    """
    clg_ids = np.asarray(arrays["CLG_id"], dtype=np.int32)
    tree_ids = np.asarray(arrays["tree_id"], dtype=np.int32)
    seed_ids = np.asarray(arrays["CLG_seed_node_id"], dtype=np.int32)
    oldest_ids = np.asarray(arrays["CLG_oldest_node_id"], dtype=np.int32)
    # np.ndarray[int64], (N_CLG+1,), 同步切分 candidate_ids 与 candidate_thresholds。
    offsets = np.asarray(arrays["candidate_offsets"])
    candidate_ids = np.asarray(arrays["candidate_node_id"], dtype=np.int32)
    candidate_thresholds = np.asarray(
        arrays["candidate_threshold_grid_index"], dtype=np.int32
    )
    n_clg = int(clg_ids.size)
    validate_offsets(offsets, int(candidate_ids.size), "candidate_offsets")
    if offsets.shape != (n_clg + 1,) or any(
        value.shape != (n_clg,) for value in (tree_ids, seed_ids, oldest_ids)
    ):
        raise ValueError("clg.npz 的 CLG 主表长度不一致")
    if candidate_thresholds.shape != candidate_ids.shape:
        raise ValueError("candidate threshold 与 node ID 长度不一致")
    if not np.array_equal(clg_ids, np.arange(n_clg, dtype=np.int32)):
        raise ValueError("CLG_id 必须是来源顺序中的连续 0..N_CLG-1")

    result: list[CLG] = []
    for row in range(n_clg):
        tree_id = int(tree_ids[row])
        start, end = int(offsets[row]), int(offsets[row + 1])
        candidates = tuple(
            forest.node(tree_id, int(node_id)) for node_id in candidate_ids[start:end]
        )
        if not np.array_equal(
            candidate_thresholds[start:end],
            np.asarray([node.threshold_grid_index for node in candidates], dtype=np.int32),
        ):
            raise ValueError("candidate_threshold_grid_index 与 forest node 不一致")
        result.append(
            CLG(
                CLG_id=int(clg_ids[row]),
                tree_id=tree_id,
                seed_node=forest.node(tree_id, int(seed_ids[row])),
                oldest_node=forest.node(tree_id, int(oldest_ids[row])),
                candidate_nodes=candidates,
            )
        )
    return tuple(result)


def _offsets_from_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    """
    从逐实体数组生成 int64 offsets。

    输入参数:
        - rows: Sequence[np.ndarray], 长度 N_entity，每项的第 0 轴长度定义一个 ragged 段

    输出:
        - offsets: np.ndarray, (N_entity+1,), int64，实体 i 对应拼接 value 表的半开区间 `[offsets[i],offsets[i+1])`
    """
    return np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum([np.asarray(row).shape[0] for row in rows], dtype=np.int64),
        ]
    )


def _concatenate_rows(rows: Sequence[np.ndarray], dtype: np.dtype[object]) -> np.ndarray:
    """
    连接逐实体一维 value 表，空输入返回契约 dtype 空数组。

    输入参数:
        - rows: Sequence[np.ndarray], 长度 N_entity，每项按一维 value 表解释
        - dtype: np.dtype, 输出契约 dtype

    输出:
        - values: np.ndarray, (L_total,), 按实体顺序拼接的一维 value 表
    """
    if len(rows) == 0 or sum(int(np.asarray(row).size) for row in rows) == 0:
        return np.empty(0, dtype=dtype)
    return np.concatenate([np.asarray(row, dtype=dtype).reshape(-1) for row in rows])
