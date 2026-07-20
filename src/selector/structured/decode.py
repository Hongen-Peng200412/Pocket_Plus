"""把 CLG 有效概率与候选结构化能量解码为正式选择结果。

门控失败时返回空选择；门控通过时调用精确树动态规划求非空最大后验反链。门控阈值
来自 calibration 冻结产物，本模块不拟合或调整它。
"""

from __future__ import annotations

from typing import Sequence

import torch

from .antichain_dp import exact_antichain_map


def decode_gated_antichain(
    clg_valid_probability: torch.Tensor,
    tau_g: float,
    selection_logit: torch.Tensor,
    parent_index: Sequence[int],
    candidate_index_by_node: Sequence[int],
    lambda_count: float,
) -> tuple[bool, torch.Tensor]:
    """
    先执行 CLG 门控，再解码非空预测 MAP 反链。

    输入参数:
        - clg_valid_probability: torch.Tensor, scalar，CLG 有效概率 `p_G`
        - tau_g: float, calibration 冻结的门控阈值 ``tau_G``
        - selection_logit: torch.Tensor, `(N_candidate,)`，候选树表示产生的反链
          能量 `z_i`
        - parent_index: Sequence[int], `(N_closure,)`，候选最小连接闭包父行号
        - candidate_index_by_node: Sequence[int], `(N_closure,)`，闭包节点到候选
          局部下标的映射
        - lambda_count: float, 预测反链计数惩罚

    输出:
        - result: tuple[bool, torch.Tensor]，组成项如下。
            - gate_pass: bool，是否通过 `p_G >= tau_g`
            - selected_candidate_index: torch.Tensor, `(N_selected,)`，int64，所属
              CLG 内的局部候选下标；门控失败时为空
    """
    gate_pass = float(clg_valid_probability.detach().reshape(())) >= float(tau_g)
    if not gate_pass:
        return False, torch.empty((0,), dtype=torch.long, device=selection_logit.device)
    _, selected = exact_antichain_map(
        selection_score=selection_logit,
        parent_index=parent_index,
        candidate_index_by_node=candidate_index_by_node,
        lambda_count=lambda_count,
    )
    return True, selected
