"""在组件树上精确计算非空反链的对数配分函数和最大后验解。

反链是一组互不构成祖先—后代关系的候选节点。`build_candidate_tree_closure` 先把
一个 CLG 的候选压缩为从各候选到最低共同祖先的最小连接树；随后两个动态规划分别
在该闭包上执行 log-sum-exp 求和与 max 求解。两者都只允许非空反链，空解是否参与
比较由调用方（例如在线最优监督）在外层决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class CandidateTreeClosure:
    """
    表示一个 CLG 候选集合的最小连接闭包。

    输入参数:
        - node_id: tuple[int, ...], 闭包内森林节点编号，父节点总排在子节点之前
        - parent_index: tuple[int, ...], 指向 `node_id` 的局部父行号；闭包根为 -1
        - candidate_index_by_node: tuple[int, ...], 每个闭包节点对应的 CLG 候选
          局部下标；仅为连通性保留的非候选节点取 -1
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
    从一棵原始组件树构造候选集合的最小连接闭包。

    输入参数:
        - node_id: Sequence[int], `(N_tree,)`，当前组件树的节点编号
        - parent_node_id: Sequence[int], `(N_tree,)`，直接父节点编号；树根为 -1
        - candidate_node_id: Sequence[int], `(N_candidate,)`，CLG 候选节点编号；
          顺序即 CLG 来源顺序，后续所有局部候选下标均以此为准

    输出:
        - closure: CandidateTreeClosure，包含从全部候选到其最低共同祖先的最小
          连接子树；父节点先于子节点，候选映射完整覆盖 `0..N_candidate-1`
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
        沿原始父节点编号从一个候选追溯到组件树根。

        输入参数:
            - start: int, 当前候选的森林节点编号

        输出:
            - path: tuple[int, ...], 从 `start` 到树根的节点编号路径，包含两端
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

    # 每条路径方向均为 candidate→root，因此首条路径中首个共同节点就是最低共同祖先。
    paths = [path_to_root(value) for value in candidates]
    common_ancestors = set(paths[0]).intersection(*(set(path) for path in paths[1:]))
    if not common_ancestors:
        raise ValueError("CLG candidates 不属于同一棵有根树。")
    # paths[0] 从 candidate 指向 root；第一个共同节点就是所有 candidates 的最低共同祖先。
    lca_node_id = next(value for value in paths[0] if value in common_ancestors)

    # included 仅保留最低共同祖先到每个候选的路径，不包含其更高祖先或无关分支。
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

    # 先按深度再按节点编号排序，既保证父节点先于子节点，又使结果确定可复现。
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
    把局部父节点表转换为直接子节点表，并校验其为无环森林。

    输入参数:
        - parent_index: Sequence[int], `(N_node,)`，局部父行号；每棵树的根为 -1

    输出:
        - children: tuple[tuple[int, ...], ...], 每个节点的直接子节点局部行号；
          子节点保持 `parent_index` 的扫描顺序
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
        深度优先检查一个局部节点的后代是否无环。

        输入参数:
            - index: int, 当前 `parent_index` 中的局部节点行号

        状态:
            - `visit_state=0/1/2` 分别表示未访问、当前递归栈内、已经完成。
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
    用 log-sum-exp 树动态规划计算全部非空候选反链的对数配分函数。

    输入参数:
        - selection_logit: torch.Tensor, `(N_candidate,)`，候选结构化选择能量 `z_i`
        - parent_index: Sequence[int], `(N_closure,)`，最小连接闭包局部父行号；
          每棵树的根为 -1
        - candidate_index_by_node: Sequence[int], `(N_closure,)`，闭包节点到候选
          局部下标的映射；非候选节点为 -1
        - lambda_count: float, 每选一个 candidate 扣除的计数惩罚

    输出:
        - log_partition: torch.Tensor, scalar，全部非空反链的
          `log(sum(exp(sum(z_i-lambda_count))))`；梯度保持连接到全部候选能量
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
    # 每个缓存值表示“该节点子树内至少选择一个候选”的 log-sum-exp；空状态固定为 0。
    node_nonempty: list[torch.Tensor | None] = [None] * len(parent_index)

    def solve(node: int) -> torch.Tensor:
        """
        返回指定闭包节点子树内全部非空反链的 log-sum-exp。

        输入参数:
            - node: int, 当前闭包节点的局部行号

        输出:
            - nonempty: torch.Tensor, scalar；至少选择一个后代候选，或在当前节点
              本身为候选时选择当前节点

        递推:
            - 跳过当前节点时，各直接子树可独立取空或非空，但整体至少一个子树非空。
            - 选择当前节点时，由反链约束禁止选择任一后代，只保留当前节点能量。
        """
        cached = node_nonempty[node]
        if cached is not None:
            return cached
        # 逐个合并直接子树：已有非空×当前任意，或已有空×当前非空。
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

    # parent_index 可表示多根森林；根之间互不构成祖先关系，按同一递推合并。
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
    用 max 树动态规划求非空候选反链的最大后验解。

    输入参数:
        - selection_score: torch.Tensor, `(N_candidate,)`，候选分数；预测时为
          `z_i`，在线最优监督时为最大 IoU `q_i`
        - parent_index: Sequence[int], `(N_closure,)`，最小连接闭包局部父行号；
          每棵树的根为 -1
        - candidate_index_by_node: Sequence[int], `(N_closure,)`，闭包节点到候选
          局部下标的映射；非候选节点为 -1
        - lambda_count: float, 每选一个 candidate 扣除的计数惩罚

    输出:
        - best_score: torch.Tensor, scalar，最佳非空反链总分
        - selected_candidate_index: torch.Tensor, `(N_selected,)`，int64，按原候选
          顺序升序排列；函数不会返回空解
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
        """
        合并互不相关子树的最佳非空解，并强制整体至少一个子树非空。

        输入参数:
            - parts: Sequence[tuple[torch.Tensor, tuple[int, ...]]]，每项是一个
              子树的最佳非空分数及候选局部下标

        输出:
            - result: tuple[torch.Tensor, tuple[int, ...]]，组合后的最佳非空分数
              与升序去重候选下标；没有子树时返回负无穷和空元组

        规则:
            - 依次指定一个必须采用非空解的子树；其余子树只有在非空分数大于 0
              时才优于各自的空解。
            - 分数并列时保留扫描顺序中的首个组合，使解码确定可复现。
        """
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
        """
        返回指定节点子树内最佳非空反链及其候选局部下标。

        输入参数:
            - node: int, 当前闭包节点的局部行号

        输出:
            - result: tuple[torch.Tensor, tuple[int, ...]]，子树内最佳非空分数和
              升序候选下标

        递推:
            - 跳过当前节点时合并各直接子树的最佳非空解。
            - 当前节点是候选时，还比较“只选择当前节点”的分数；并列时选择当前
              节点，从而保持固定决胜规则。
        """
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
