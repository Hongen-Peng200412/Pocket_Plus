"""在 component tree 上精确计算非空反链 partition 与 MAP。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class CandidateTreeClosure:
    """
    表示一个 CLG candidates 的最小连接闭包。

    输入参数:
        - node_id: tuple[int, ...], 闭包内 forest node ID，父节点总排在子节点之前
        - parent_index: tuple[int, ...], 指向 node_id 的局部父行；闭包根为 -1
        - candidate_index_by_node: tuple[int, ...], 每个闭包节点对应的 CLG candidate 局部下标；非 candidate 为 -1
    输出:
        - node_id: tuple[int,...], 闭包内 forest node ID，父节点总排在子节点之前
        - parent_index: tuple[int,...], 闭包节点的局部父行，闭包根为 -1
        - candidate_index_by_node: tuple[int,...], 闭包节点到 candidate 局部下标的映射，非 candidate 为 -1
    """

    node_id: tuple[int, ...]
    parent_index: tuple[int, ...]
    candidate_index_by_node: tuple[int, ...]


def build_candidate_tree_closure(
    node_id: Sequence[int],
    parent_node_id: Sequence[int],
    candidate_node_id: Sequence[int],
) -> CandidateTreeClosure:
    """
    从一棵原始 component tree 构造 candidates 的最小连接闭包。

    输入参数:
        - node_id: Sequence[int], (N_tree,), 当前 tree 的 node identity
        - parent_node_id: Sequence[int], (N_tree,), direct parent identity；tree root 为 -1
        - candidate_node_id: Sequence[int], (N_candidate,), CLG candidate identity，顺序即 CLG 来源顺序

    输出:
        - closure: CandidateTreeClosure，包含从所有 candidate 到共同 LCA 的最小连接子树
    """
    ids = tuple(int(value) for value in node_id)
    parents = tuple(int(value) for value in parent_node_id)
    candidates = tuple(int(value) for value in candidate_node_id)
    if len(ids) != len(parents):
        raise ValueError("node_id 与 parent_node_id 长度必须一致。")
    if not candidates:
        raise ValueError("CLG 至少需要一个 candidate node。")
    if len(set(ids)) != len(ids) or len(set(candidates)) != len(candidates):
        raise ValueError("tree node_id 与 candidate_node_id 均必须唯一。")

    parent_by_id = dict(zip(ids, parents, strict=True))
    missing = [value for value in candidates if value not in parent_by_id]
    if missing:
        raise KeyError(f"candidate_node_id 不属于当前 tree: {missing}")

    def path_to_root(start: int) -> tuple[int, ...]:
        """
        沿原始 parent identity 从一个 candidate 追溯到 tree root。

        输入参数:
            - start: int, 当前 candidate 的 forest node identity

        输出:
            - path: tuple[int,...], 从 start 到 root 的 node identity 路径
        """
        path: list[int] = []
        seen: set[int] = set()
        current = start
        while current != -1:
            if current in seen:
                raise ValueError("component tree parent 链存在环。")
            if current not in parent_by_id:
                raise KeyError(f"parent_node_id={current} 不在当前 tree。")
            path.append(current)
            seen.add(current)
            current = parent_by_id[current]
        return tuple(path)

    paths = [path_to_root(value) for value in candidates]
    common_ancestors = set(paths[0]).intersection(*(set(path) for path in paths[1:]))
    if not common_ancestors:
        raise ValueError("CLG candidates 不属于同一棵有根树。")
    # paths[0] 从 candidate 指向 root；第一个共同节点就是所有 candidates 的最低共同祖先。
    lca_node_id = next(value for value in paths[0] if value in common_ancestors)

    included: set[int] = {lca_node_id}
    for path in paths:
        for value in path:
            included.add(value)
            if value == lca_node_id:
                break

    depth_by_id: dict[int, int] = {lca_node_id: 0}
    unresolved = included - {lca_node_id}
    while unresolved:
        progressed = False
        for value in tuple(unresolved):
            parent = parent_by_id[value]
            if parent in depth_by_id:
                depth_by_id[value] = depth_by_id[parent] + 1
                unresolved.remove(value)
                progressed = True
        if not progressed:
            raise ValueError("无法在候选最小连接闭包中解析父子拓扑。")

    ordered_ids = tuple(sorted(included, key=lambda value: (depth_by_id[value], value)))
    local_index = {value: index for index, value in enumerate(ordered_ids)}
    candidate_index = {value: index for index, value in enumerate(candidates)}
    local_parent = tuple(
        -1 if value == lca_node_id else local_index[parent_by_id[value]]
        for value in ordered_ids
    )
    candidate_by_node = tuple(candidate_index.get(value, -1) for value in ordered_ids)
    return CandidateTreeClosure(
        node_id=ordered_ids,
        parent_index=local_parent,
        candidate_index_by_node=candidate_by_node,
    )


def _children_from_parent(parent_index: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """
    把局部 parent 表转换为 children 表并校验其为森林。

    输入参数:
        - parent_index: Sequence[int], (N_node,), 局部父行；root 为 -1

    输出:
        - children: tuple[tuple[int, ...], ...], 每个节点的 direct child 局部行
    """
    children: list[list[int]] = [[] for _ in parent_index]
    for child, parent in enumerate(parent_index):
        parent_value = int(parent)
        if parent_value == -1:
            continue
        if parent_value < 0 or parent_value >= len(parent_index) or parent_value == child:
            raise ValueError("parent_index 含越界或自环。")
        children[parent_value].append(child)

    visit_state = [0] * len(parent_index)

    def visit(index: int) -> None:
        """
        深度优先检查一个局部节点是否无环且可达。

        输入参数:
            - index: int, 当前 parent_index 的局部节点行
        """
        if visit_state[index] == 1:
            raise ValueError("parent_index 含有环。")
        if visit_state[index] == 2:
            return
        visit_state[index] = 1
        for child in children[index]:
            visit(child)
        visit_state[index] = 2

    roots = [index for index, parent in enumerate(parent_index) if int(parent) == -1]
    if not roots:
        raise ValueError("parent_index 至少需要一个 root。")
    for root in roots:
        visit(root)
    if any(state != 2 for state in visit_state):
        raise ValueError("parent_index 含有无法从 root 到达的节点。")
    return tuple(tuple(values) for values in children)


def exact_antichain_log_partition(
    selection_logit: torch.Tensor,
    parent_index: Sequence[int],
    candidate_index_by_node: Sequence[int],
    lambda_count: float,
) -> torch.Tensor:
    """
    用 logsumexp 树 DP 计算全部非空 candidate 反链的对数 partition。

    输入参数:
        - selection_logit: torch.Tensor, (N_candidate,), candidate 结构化选择能量 z_i
        - parent_index: Sequence[int], (N_closure,), 最小连接闭包局部父行；root 为 -1
        - candidate_index_by_node: Sequence[int], (N_closure,), 闭包节点到 candidate 行的映射；非 candidate 为 -1
        - lambda_count: float, 每选一个 candidate 扣除的计数惩罚

    输出:
        - log_partition: torch.Tensor, scalar，全部非空反链能量 exp(score) 之和的自然对数
    """
    logits = selection_logit.reshape(-1)
    candidate_map = tuple(int(value) for value in candidate_index_by_node)
    if len(parent_index) != len(candidate_map):
        raise ValueError("parent_index 与 candidate_index_by_node 长度必须一致。")
    used_candidates = sorted(value for value in candidate_map if value >= 0)
    if used_candidates != list(range(logits.numel())):
        raise ValueError("candidate_index_by_node 必须恰好覆盖 0..N_candidate-1。")

    children = _children_from_parent(parent_index)
    roots = tuple(index for index, parent in enumerate(parent_index) if int(parent) == -1)
    negative_infinity = logits.new_tensor(float("-inf"))
    zero = logits.new_zeros(())
    node_nonempty: list[torch.Tensor | None] = [None] * len(parent_index)

    def solve(node: int) -> torch.Tensor:
        cached = node_nonempty[node]
        if cached is not None:
            return cached
        # empty_state 恒为 log(1)=0；nonempty_state 只累计至少选择一个 candidate 的组合。
        combined_nonempty = negative_infinity
        for child in children[node]:
            child_nonempty = solve(child)
            child_all = torch.logaddexp(zero, child_nonempty)
            combined_nonempty = torch.logaddexp(
                combined_nonempty + child_all,
                child_nonempty,
            )
        candidate_index = candidate_map[node]
        if candidate_index >= 0:
            choose_node = logits[candidate_index] - float(lambda_count)
            result = torch.logaddexp(combined_nonempty, choose_node)
        else:
            result = combined_nonempty
        node_nonempty[node] = result
        return result

    forest_nonempty = negative_infinity
    for root in roots:
        root_nonempty = solve(root)
        root_all = torch.logaddexp(zero, root_nonempty)
        forest_nonempty = torch.logaddexp(forest_nonempty + root_all, root_nonempty)
    return forest_nonempty


def exact_antichain_map(
    selection_score: torch.Tensor,
    parent_index: Sequence[int],
    candidate_index_by_node: Sequence[int],
    lambda_count: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    用 max 树 DP 求非空 candidate 反链 MAP。

    输入参数:
        - selection_score: torch.Tensor, (N_candidate,), candidate 分数；预测时为 z_i，oracle 时为 q_i
        - parent_index: Sequence[int], (N_closure,), 最小连接闭包局部父行；root 为 -1
        - candidate_index_by_node: Sequence[int], (N_closure,), 闭包节点到 candidate 行的映射；非 candidate 为 -1
        - lambda_count: float, 每选一个 candidate 扣除的计数惩罚

    输出:
        - best_score: torch.Tensor, scalar，最佳非空反链总分
        - selected_candidate_index: torch.Tensor, (N_selected,), int64，按原 candidate 顺序升序排列
    """
    scores = selection_score.reshape(-1)
    candidate_map = tuple(int(value) for value in candidate_index_by_node)
    if len(parent_index) != len(candidate_map):
        raise ValueError("parent_index 与 candidate_index_by_node 长度必须一致。")
    used_candidates = sorted(value for value in candidate_map if value >= 0)
    if used_candidates != list(range(scores.numel())):
        raise ValueError("candidate_index_by_node 必须恰好覆盖 0..N_candidate-1。")

    children = _children_from_parent(parent_index)
    roots = tuple(index for index, parent in enumerate(parent_index) if int(parent) == -1)
    # 每个节点缓存“子树内最佳非空反链”；空解分数固定为 0，不作为此缓存的合法返回。
    cache: dict[int, tuple[torch.Tensor, tuple[int, ...]]] = {}

    def combine_nonempty(
        parts: Sequence[tuple[torch.Tensor, tuple[int, ...]]],
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        best_score: torch.Tensor | None = None
        best_selection: tuple[int, ...] = ()
        for required_index, (required_score, required_selection) in enumerate(parts):
            total = required_score
            selected = list(required_selection)
            for other_index, (other_score, other_selection) in enumerate(parts):
                if other_index == required_index:
                    continue
                if float(other_score.detach()) > 0.0:
                    total = total + other_score
                    selected.extend(other_selection)
            ordered = tuple(sorted(set(selected)))
            if best_score is None or float(total.detach()) > float(best_score.detach()):
                best_score = total
                best_selection = ordered
        if best_score is None:
            return scores.new_tensor(float("-inf")), ()
        return best_score, best_selection

    def solve(node: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        if node in cache:
            return cache[node]
        child_parts = [solve(child) for child in children[node]]
        skip_score, skip_selection = combine_nonempty(child_parts)
        candidate_index = candidate_map[node]
        if candidate_index >= 0:
            choose_score = scores[candidate_index] - float(lambda_count)
            if float(choose_score.detach()) >= float(skip_score.detach()):
                result = (choose_score, (candidate_index,))
            else:
                result = (skip_score, skip_selection)
        else:
            result = (skip_score, skip_selection)
        cache[node] = result
        return result

    best_score, best_selection = combine_nonempty([solve(root) for root in roots])
    selected_tensor = torch.as_tensor(best_selection, dtype=torch.long, device=scores.device)
    return best_score, selected_tensor
