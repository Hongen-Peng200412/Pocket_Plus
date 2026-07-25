"""在树专属工作副本上按 split/merge 事件枚举 CLG。

每次选择一个 active eligible seed，先向高阈值 children 展开 split，再向低阈值
parent 展开 merge 并带入 sisters。无论尝试成功还是因 node cap 放弃，最后都只按
seed 删除 ``D_W(g)``；正式 forest 和其它树不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .structures import CLG, ComponentForest, ComponentNode, WorkingTree, clgs_to_arrays


@dataclass(frozen=True)
class CLGEnumerationConfig:
    """
    定义一次 CLG 枚举的事件与节点上限。

    输入参数:
        - max_split_events: int, 每条向高阈值扩展分支最多接受的多路 split 数
        - max_merge_events: int, 从 seed 向低阈值最多接受的多路 merge 数
        - max_nodes_per_CLG: int, 一个成功 CLG 允许的最大 candidate 数；
          depth1 为 32，depth2 为 64
    """

    max_split_events: int
    max_merge_events: int
    max_nodes_per_CLG: int

    def __post_init__(self) -> None:
        """
        校验 split/merge 预算与 candidate 上限。

        输出:
            - None: 三个上限合法时返回；负事件预算或非正 candidate 上限直接报错
        """
        if self.max_split_events < 0 or self.max_merge_events < 0:
            raise ValueError("split/merge 事件预算不能为负")
        if self.max_nodes_per_CLG <= 0:
            raise ValueError("max_nodes_per_CLG 必须为正数")


@dataclass(frozen=True)
class CLGEnumerationResult:
    """
    保存成功 CLG、可直接落盘的数值字段与 PDB 级统计。

    输入参数:
        - clgs: tuple[CLG,...], 按来源顺序排列的成功 CLG
        - arrays: dict, `clg.npz` 数值 ragged 字段
        - summary: dict, cap、完成数、拒绝数与均值统计
    """

    clgs: tuple[CLG, ...]
    arrays: dict[str, object]
    summary: dict[str, int | float | bool]


class _NodeCapExceeded(RuntimeError):
    """
    表示一次原子加入会使当前 CLG candidate 总数超过上限。

    该异常只终止当前 seed 的 attempt；外层仍执行相同的 `delete_D(seed)`。
    """


class _Attempt:
    """
    维护单次 CLG 尝试的确定性 candidate 顺序与原子加入。

    输入参数:
        - working_tree: WorkingTree, 当前 seed 所属 tree 的 active 工作副本
        - cap: int, 当前 CLG 允许的最大 candidate 数

    状态:
        - `candidates`: list[ComponentNode]，按首次原子加入顺序保存的候选
        - `_candidate_ids`: set[int]，用于阻止同一 node_id 重复加入
    """

    def __init__(self, working_tree: WorkingTree, cap: int) -> None:
        """
        初始化空的单次 CLG attempt。

        输入参数:
            - working_tree: WorkingTree, 当前 seed 所属 tree 的 active 工作副本
            - cap: int, 最大 candidate 数

        输出:
            - None: 原地建立空 candidate 列表与 identity 集合
        """
        self.working_tree = working_tree
        self.cap = int(cap)
        self.candidates: list[ComponentNode] = []
        self._candidate_ids: set[int] = set()

    def add_atomic(self, nodes: Sequence[ComponentNode]) -> bool:
        """
        在全部新节点 active 且 eligible 时原子加入，超过 cap 则抛弃整个尝试。

        输入参数:
            - nodes: Sequence[ComponentNode], 要按给定顺序一起加入的节点集合

        输出:
            - added: bool, 全部新节点成功加入或本来已存在时为 True；任一新节点 inactive/ineligible 时为 False
        """
        # list[ComponentNode], 保持调用顺序且排除本 attempt 已收录的 candidate。
        unique_new = [node for node in nodes if node.node_id not in self._candidate_ids]
        if any(
            (not self.working_tree.is_active(node)) or (not node.candidate_eligible)
            for node in unique_new
        ):
            return False
        if len(self.candidates) + len(unique_new) > self.cap:
            raise _NodeCapExceeded
        self.candidates.extend(unique_new)
        self._candidate_ids.update(node.node_id for node in unique_new)
        return True


def enumerate_clgs(
    forest: ComponentForest,
    f1_threshold_grid_index: int,
    config: CLGEnumerationConfig,
) -> CLGEnumerationResult:
    """
    按 F1 层优先、再向低阈值的固定顺序枚举一个 PDB 的 CLG。

    输入参数:
        - forest: ComponentForest, 原始只读组件森林
        - f1_threshold_grid_index: int, `t_F1` 对应的整数扫描下标 j
        - config: CLGEnumerationConfig, split/merge 与 candidate node 上限

    输出:
        - result: CLGEnumerationResult，其中成功和 node-cap 失败都只对当前 seed
          在所属 WorkingTree 上应用同一 `D_W(g)`；失败尝试不分配 CLG_id
    """
    # dict[int,WorkingTree], 每棵正式 tree 各有一个只改 active 状态的枚举副本。
    working_by_tree = {tree.tree_id: WorkingTree(tree) for tree in forest.trees}
    # list[ComponentNode], F1 层优先，其余从高阈值到低阈值的稳定 seed 候选顺序。
    seed_order = sorted(
        (
            node
            for node in forest.nodes
            if node.candidate_eligible
            and node.threshold_grid_index <= int(f1_threshold_grid_index)
        ),
        key=lambda node: (
            0 if node.threshold_grid_index == int(f1_threshold_grid_index) else 1,
            -int(node.threshold_grid_index),
            -float(node.probability_mean),
            int(node.tree_id),
            int(node.node_id),
        ),
    )
    # int, t_F1 层 eligible seed 数；用于计算本 PDB 的 CLG 总量上限。
    n_f1_seed = sum(
        node.threshold_grid_index == int(f1_threshold_grid_index) for node in seed_order
    )
    n_clg_cap = min(300, 3 * max(2, int(n_f1_seed)))
    clgs: list[CLG] = []
    rejected_by_node_cap = 0

    while len(clgs) < n_clg_cap:
        # ComponentNode | None, seed_order 中第一个仍 active 的节点。
        seed = next(
            (
                node
                for node in seed_order
                if working_by_tree[node.tree_id].is_active(node)
            ),
            None,
        )
        if seed is None:
            break
        working_tree = working_by_tree[seed.tree_id]
        attempt = _Attempt(working_tree, config.max_nodes_per_CLG)
        try:
            if not attempt.add_atomic((seed,)):
                raise RuntimeError("seed_order 中的节点必须同时 active 且 eligible")
            _expand_toward_higher_thresholds(
                current=seed,
                split_events_left=config.max_split_events,
                attempt=attempt,
            )
            _expand_toward_lower_thresholds(
                seed=seed,
                merge_events_left=config.max_merge_events,
                full_split_budget=config.max_split_events,
                attempt=attempt,
            )
            # list[ComponentNode], 同时祖先于本 attempt 全部 candidates 的候选；成功时唯一。
            oldest_candidates = [
                node
                for node in attempt.candidates
                if all(node.is_ancestor_of(other) for other in attempt.candidates)
            ]
            if len(oldest_candidates) != 1:
                raise RuntimeError(
                    "成功 CLG 必须存在唯一、覆盖全部 candidates 的 CLG_oldest_node"
                )
            clgs.append(
                CLG(
                    CLG_id=len(clgs),
                    tree_id=seed.tree_id,
                    seed_node=seed,
                    oldest_node=oldest_candidates[0],
                    candidate_nodes=tuple(attempt.candidates),
                )
            )
        except _NodeCapExceeded:
            rejected_by_node_cap += 1
        finally:
            working_tree.delete_D(seed)

    active_seed_remains = any(
        working_by_tree[node.tree_id].is_active(node) for node in seed_order
    )
    n_completed = len(clgs)
    mean_candidates = (
        float(sum(len(clg.candidate_nodes) for clg in clgs)) / float(n_completed)
        if n_completed
        else 0.0
    )
    summary: dict[str, int | float | bool] = {
        "max_split_events": int(config.max_split_events),
        "max_merge_events": int(config.max_merge_events),
        "max_nodes_per_CLG": int(config.max_nodes_per_CLG),
        "n_f1_eligible_seeds": int(n_f1_seed),
        "n_CLG_cap": int(n_clg_cap),
        "n_CLG_completed": int(n_completed),
        "n_CLG_rejected_by_node_cap": int(rejected_by_node_cap),
        "mean_candidates_per_completed_CLG": float(mean_candidates),
        "CLG_cap_reached": bool(
            n_completed == n_clg_cap and active_seed_remains
        ),
    }
    return CLGEnumerationResult(
        clgs=tuple(clgs),
        arrays=clgs_to_arrays(clgs),
        summary=summary,
    )


def _expand_toward_higher_thresholds(
    current: ComponentNode,
    split_events_left: int,
    attempt: _Attempt,
) -> None:
    """
    沿 direct children 向高阈值扩展，unary 不耗预算，多路 split 原子加入全部 children。

    输入参数:
        - current: ComponentNode, 当前扩展节点
        - split_events_left: int, 当前分支剩余多路 split 预算
        - attempt: _Attempt, 原地累积 candidate 的当前尝试

    输出:
        - None: 原地更新 `attempt.candidates`
    """
    original_children = list(current.children)
    if not original_children:
        return
    is_split = len(original_children) > 1
    if is_split and split_events_left <= 0:
        return
    if not attempt.add_atomic(original_children):
        return
    next_budget = split_events_left - 1 if is_split else split_events_left
    for child in original_children:
        _expand_toward_higher_thresholds(child, next_budget, attempt)


def _expand_toward_lower_thresholds(
    seed: ComponentNode,
    merge_events_left: int,
    full_split_budget: int,
    attempt: _Attempt,
) -> None:
    """
    沿 direct parent 向低阈值扩展，merge 原子加入 parent 与 sisters，并扩展新 sisters。

    输入参数:
        - seed: ComponentNode, 当前 CLG 尝试的 seed
        - merge_events_left: int, 向低阈值方向剩余多路 merge 预算
        - full_split_budget: int, 每个新 sister 向高阈值展开时恢复的完整 split 预算
        - attempt: _Attempt, 原地累积 candidate 的当前尝试

    输出:
        - None: 原地更新 `attempt.candidates`
    """
    current = seed
    remaining = int(merge_events_left)
    while current.parent is not None:
        parent = current.parent
        sisters = [child for child in parent.children if child is not current]
        is_merge = len(sisters) > 0
        if is_merge and remaining <= 0:
            return
        event_nodes = [parent, *sisters]
        if not attempt.add_atomic(event_nodes):
            return
        if is_merge:
            remaining -= 1
            for sister in sisters:
                _expand_toward_higher_thresholds(
                    sister,
                    split_events_left=full_split_budget,
                    attempt=attempt,
                )
        current = parent
