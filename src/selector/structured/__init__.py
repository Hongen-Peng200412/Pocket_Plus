"""公开 Selector 的在线最优监督、候选树闭包和精确反链求解接口。

本子包只依赖基础数组与 tensor 操作，不读取 Stage1 产物，也不负责 CLG 概率校准。
"""

from .antichain_dp import (
    CandidateTreeClosure,
    build_candidate_tree_closure,
    exact_antichain_log_partition,
    exact_antichain_map,
)
from .decode import decode_gated_antichain
from .oracle import OnlineOracle, build_online_oracle, compute_candidate_max_iou

__all__ = [
    "CandidateTreeClosure",
    "OnlineOracle",
    "build_candidate_tree_closure",
    "build_online_oracle",
    "compute_candidate_max_iou",
    "decode_gated_antichain",
    "exact_antichain_log_partition",
    "exact_antichain_map",
]
