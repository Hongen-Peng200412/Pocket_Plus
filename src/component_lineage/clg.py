"""在树专属 active 工作副本上按 split/merge 事件枚举候选谱系组（CLG）。

主要入口:
    - `CLGEnumerationConfig`: 固定向高阈值 split、向低阈值 merge 和单组候选数量上限。
    - `enumerate_clgs`: 以 `t_F1` 层 eligible 节点优先的稳定顺序选择 seed，构造成功 CLG 及可落盘 `clg.npz` 字段。

一次尝试先从 seed 向高阈值 children 展开 split，再向低阈值 parent 展开 merge 并原子带入 sisters。无论尝试成功还是因候选数量上限放弃，最后都只从所属 `WorkingTree` 删除 `D_W(g)=Ancestors_W(g)∪Subtree_W(g)`；正式 `ComponentForest`、voxel payload 和其它 tree 不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .structures import CLG, ComponentForest, ComponentNode, WorkingTree, clgs_to_arrays

# ================================================================================ 简单配置 & 结果 ================================================================================
@dataclass(frozen=True)
class CLGEnumerationConfig:
    """
    定义一次 CLG 枚举的事件与节点上限。

    输入参数:
        - max_split_events: int, 每条向高阈值扩展分支最多接受的多路 split 事件数；unary child 不消耗预算。
        - max_merge_events: int, 从 seed 向低阈值 parent 链最多接受的多路 merge 事件数；没有 sister 的 unary parent 不消耗预算。
        - max_nodes_per_CLG: int, 一个成功 CLG 允许的最大唯一 candidate 数；depth1 为 32，depth2 为 64。
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
        - clgs: tuple[CLG, ...], 按正式枚举顺序排列的成功 CLG；`CLG_id` 与元组位置一致。
        - arrays: dict[str, object], 可直接写入 `clg.npz` 的 CLG 主表、候选 ragged 字段和阈值回指。
        - summary: dict[str, int | float | bool], 当前 PDB 的事件预算、候选上限、完成数、拒绝数、均值与总量上限状态。
    """
    clgs: tuple[CLG, ...]
    arrays: dict[str, object]
    summary: dict[str, int | float | bool]

class _NodeCapExceeded(RuntimeError):
    """
    表示一次原子加入会使当前 CLG candidate 总数超过上限。

    该异常只终止当前 seed 的 attempt；外层仍执行相同的 `delete_D(seed)`。
    """






# ================================================================================ 核心运算: 给定一个seed node生成CLG ================================================================================
class _Attempt:
    """
    维护单次 CLG 尝试的确定性 candidate 顺序与原子加入。

    输入参数:
        - working_tree: WorkingTree, 当前 seed 所属 tree 的 active 工作副本；只读取其 active 状态。
        - cap: int, 当前尝试允许的最大唯一 candidate 数。

    状态:
        - candidates: list[ComponentNode], 按首次原子加入顺序保存的同 tree 候选。(正常情况下和下面的通常)
        - _candidate_ids: set[int], 已加入 candidate node_id 集合，用于去重且不改变 `candidates` 顺序。
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
        在全部新节点 active 且 eligible 时加入 CLG(不然都不加入); 超过 cap 则抛弃整个尝试。

        输入参数:
            - nodes: Sequence[ComponentNode], 要按给定顺序一起加入的节点集合

        输出:
            - added: bool, 全部新节点成功加入或本来已存在时为 True；任一新节点 inactive/ineligible 时为 False
        """
        # list[ComponentNode], 保持调用顺序且排除本次 attempt 已收录的 node_id；整组后续以原子方式接受或拒绝。
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
        - forest: ComponentForest, 当前 PDB 的原始只读组件森林；每棵 tree 独立建立一个 active 工作副本。
        - f1_threshold_grid_index: int, 冻结 `t_F1` 对应的阈值网格整数 j；该层 eligible 节点在 seed 顺序中优先。
        - config: CLGEnumerationConfig, split/merge 事件预算与单个 CLG 的 candidate 数量上限。

    输出:
        - result: CLGEnumerationResult, 当前 PDB 的完整 CLG 枚举结果；包含以下三个字段。
            - clgs: tuple[CLG, ...], 按正式枚举顺序保存成功 CLG；第 i 个元素的 `CLG_id` 为 i，每个 CLG 还包含所属 `tree_id`、`seed_node`、覆盖全部候选的 `oldest_node` 和按确定性顺序排列的 `candidate_nodes`。
            - arrays: dict[str, np.ndarray], 由 `clgs_to_arrays` 生成、可直接写入 `clg.npz` 的数值字段。
                - `CLG_id`: int32, (N_CLG,), 按枚举顺序保存的连续 CLG 本地编号，并与 `clgs` 逐元素对齐。
                - `tree_id`: int32, (N_CLG,), 每个 CLG 所属的原始 component tree 编号，与 `CLG_id` 逐元素对齐。
                - `CLG_seed_node_id`: int32, (N_CLG,), 每个 CLG 的 seed `node_id`，与 `CLG_id` 逐元素对齐。
                - `CLG_oldest_node_id`: int32, (N_CLG,), 每个 CLG 的最低阈值覆盖节点 `node_id`，与 `CLG_id` 逐元素对齐。
                - `candidate_offsets`: int64, (N_CLG + 1,), 用于切分两个变长 candidate 数组；CLG i 的 candidate 位于 `[candidate_offsets[i], candidate_offsets[i + 1])`。
                - `candidate_node_id`: int32, (N_candidate_total,), 按 CLG 顺序和组内确定性枚举顺序拼接的 candidate `node_id`，由 `candidate_offsets` 与 CLG 主表对齐。
                - `candidate_threshold_grid_index`: int32, (N_candidate_total,), 与 `candidate_node_id` 逐元素对齐的阈值网格整数 j。
            - summary: dict[str, int | float | bool], PDB 级标量统计。
                - `max_split_events`: int, 每个 CLG 尝试允许的最大 split 事件预算。
                - `max_merge_events`: int, 每个 CLG 尝试允许的最大 merge 事件预算。
                - `max_nodes_per_CLG`: int, 单个 CLG 允许的最大唯一 candidate 数量。
                - `n_f1_eligible_seeds`: int, `t_F1` 层 eligible seed 的数量。
                - `n_CLG_cap`: int, PDB 级允许完成的最大 CLG 数量。
                - `n_CLG_completed`: int, 实际成功完成并写入 `clgs` 的 CLG 数量。
                - `n_CLG_rejected_by_node_cap`: int, 因 candidate 数量超过上限而拒绝的 seed 尝试数量。
                - `mean_candidates_per_completed_CLG`: float, 成功 CLG 的平均 candidate 数量。
                - `CLG_cap_reached`: bool, 是否因达到 PDB 级 CLG 数量上限而停止继续枚举。
    约束:
        - 无论尝试成功还是因 node-cap 失败，都只对当前 seed 应用同一 `D_W(g)`；失败尝试不分配 `CLG_id`。
    """
    # dict[int, WorkingTree], key 为 `forest.trees` 中的 tree_id，value 为对应 ComponentTree 的轻量枚举副本；副本只保存 parent/children 拓扑和 active 状态并回指原 ComponentNode，不复制 voxel/probability/geometry 数据；各 tree 独立维护删除状态。
    working_by_tree = {tree.tree_id: WorkingTree(tree) for tree in forest.trees}
    # list[ComponentNode], `t_F1` 层优先，再考察更低阈值的层次: 按高阈值到低阈值排列；概率均值、tree_id 和 node_id 共同打破并列。
    seed_order = sorted(
        (
            node
            for node in forest.nodes
            if node.candidate_eligible and node.threshold_grid_index <= int(f1_threshold_grid_index)
        ),
        key=lambda node: (
            0 if node.threshold_grid_index == int(f1_threshold_grid_index) else 1,  # 越小越优先
            -int(node.threshold_grid_index),
            -float(node.probability_mean),
            int(node.tree_id),
            int(node.node_id),
        ),
    )
    # int, `t_F1` 层 eligible seed 数；用于计算当前 PDB 的 CLG 总量上限 `min(300, 3*max(2, N_F1))`。
    n_f1_seed = sum(node.threshold_grid_index == int(f1_threshold_grid_index) for node in seed_order)
    n_clg_cap = min(300, 3 * max(2, int(n_f1_seed)))
    clgs: list[CLG] = []
    rejected_by_node_cap = 0

    while len(clgs) < n_clg_cap:
        # ComponentNode | None, `seed_order` 中第一个仍 active 的节点；None 表示全部候选 seed 已被先前 `D_W(g)` 覆盖。
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
            # list[ComponentNode], 同时祖先于本次 attempt 全部 candidates 的候选；成功 CLG 必须恰有一个最低阈值 oldest 节点。
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
        finally:                        # (语法) 不管前面成功、失败、抛异常，还是提前 return，离开 try 之前都执行这里。
            working_tree.delete_D(seed)

    active_seed_remains = any(working_by_tree[node.tree_id].is_active(node) for node in seed_order)
    n_completed = len(clgs)
    mean_candidates = (
        float(sum(len(clg.candidate_nodes) for clg in clgs)) / float(n_completed)
        if n_completed
        else 0.0
    )
    # dict[str, int | float | bool], 可并入 `summary.json` 的 PDB 级 CLG 枚举配置、计数和是否因总量上限截断。
    summary: dict[str, int | float | bool] = {
        "max_split_events": int(config.max_split_events),
        "max_merge_events": int(config.max_merge_events),
        "max_nodes_per_CLG": int(config.max_nodes_per_CLG),
        "n_f1_eligible_seeds": int(n_f1_seed),
        "n_CLG_cap": int(n_clg_cap),
        "n_CLG_completed": int(n_completed),
        "n_CLG_rejected_by_node_cap": int(rejected_by_node_cap),
        "mean_candidates_per_completed_CLG": float(mean_candidates),
        "CLG_cap_reached": bool(n_completed == n_clg_cap and active_seed_remains),
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
        - current: ComponentNode, 当前向高阈值方向扩展的原 forest 节点。
        - split_events_left: int, 当前递归分支剩余多路 split 预算；unary child 不消耗预算。
        - attempt: _Attempt, 原地累积 candidate 的当前尝试；任一 child inactive/ineligible 会停止该分支。

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
        - seed: ComponentNode, 当前 CLG 尝试的 seed，也是向低阈值 parent 链扩展的起点。
        - merge_events_left: int, 向低阈值方向剩余多路 merge 预算；没有 sister 的 unary parent 不消耗预算。
        - full_split_budget: int, 每个新加入 sister 向高阈值展开时重新获得的完整 split 预算。
        - attempt: _Attempt, 原地累积 candidate 的当前尝试；parent 与全部 sisters 作为一个原子事件加入。

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
