"""
LigandSparseRefineDeltaLoss(ranking-only) 的数值与边界单元测试。

被测对象: src/modules/losses.py::LigandSparseRefineDeltaLoss(纯 torch, 不依赖 PTV3/torch_cluster)。
当前设计契约: 仅保留 pairwise ranking，preservation / p_best / base_logit 都不再参与损失。
"""
from __future__ import annotations

import torch

from src.modules.losses import LigandSparseRefineDeltaLoss


def _make_loss(m_rank: float = 0.5, topk: int = 512) -> LigandSparseRefineDeltaLoss:
    """
    构造 ranking-only delta loss 模块。

    输入参数:
        - m_rank: float, ranking margin
        - topk: int, 对称难例挖掘每 BOX 各取的难正/难负数

    输出:
        - loss: LigandSparseRefineDeltaLoss, delta 损失模块
    """
    return LigandSparseRefineDeltaLoss(m_rank=m_rank, topk=topk)


def test_ranking_direction_and_value() -> None:
    """
    ranking=max(0, m_rank - refined_pos + refined_neg): 正例 logit 远高于负例时趋零, 反之产生正损失。
    """
    loss = _make_loss(m_rank=0.5, topk=512)
    r_good = torch.tensor([5.0, -5.0])
    target = torch.tensor([1.0, 0.0])
    valid = torch.ones(2, dtype=torch.bool)
    bidx = torch.zeros(2, dtype=torch.long)
    base_prob = torch.tensor([0.5, 0.5])
    out_good = loss(
        refined_logit=r_good,
        base_prob=base_prob,
        target=target,
        valid=valid,
        batch_index=bidx,
    )
    assert float(out_good["rank"]) == 0.0

    r_bad = torch.tensor([0.0, 1.0])
    out_bad = loss(
        refined_logit=r_bad,
        base_prob=base_prob,
        target=target,
        valid=valid,
        batch_index=bidx,
    )
    assert abs(float(out_bad["rank"]) - 1.5) < 1e-5, float(out_bad["rank"])


def test_ranking_topk_hard_mining_caps_pairs() -> None:
    """
    topk 小于实际正/负数时, 每 BOX 仅取 base_prob 最难的 Kp 正例 × Kn 负例配对。
    """
    loss = _make_loss(m_rank=0.5, topk=2)
    n_pos, n_neg = 5, 5
    n = n_pos + n_neg
    refined = torch.zeros(n, requires_grad=True)
    target = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)])
    base_prob = torch.cat([torch.linspace(0.1, 0.9, n_pos), torch.linspace(0.1, 0.9, n_neg)])
    valid = torch.ones(n, dtype=torch.bool)
    bidx = torch.zeros(n, dtype=torch.long)
    out = loss(
        refined_logit=refined,
        base_prob=base_prob,
        target=target,
        valid=valid,
        batch_index=bidx,
    )
    assert abs(float(out["rank"]) - 0.5) < 1e-5, float(out["rank"])
    out["rank"].backward()
    assert refined.grad is not None and torch.isfinite(refined.grad).all()


def test_ranking_skips_box_without_pos_or_neg() -> None:
    """
    某 BOX 只有正例或只有负例时该 BOX 不产生配对; 全空 -> rank=0。
    """
    loss = _make_loss()
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    batch_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = loss(
        refined_logit=torch.randn(4),
        base_prob=torch.rand(4),
        target=target,
        valid=torch.ones(4, dtype=torch.bool),
        batch_index=batch_index,
    )
    assert float(out["rank"]) == 0.0


def test_ranking_returns_zero_when_all_entries_invalid() -> None:
    """
    全 invalid 时 ranking=0。
    """
    loss = _make_loss()
    out = loss(
        refined_logit=torch.randn(4),
        base_prob=torch.rand(4),
        target=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        valid=torch.zeros(4, dtype=torch.bool),
        batch_index=torch.zeros(4, dtype=torch.long),
    )
    assert float(out["rank"]) == 0.0


def test_gradient_flows_only_through_refined_under_detached_base() -> None:
    """
    真实契约: 调用方传入已 detach 的 base_prob(candidate_set 已 detach)。
    此时梯度只经 refined_logit 回流且有限。
    """
    loss = _make_loss(m_rank=0.5, topk=512)
    n = 32
    refined = torch.randn(n, requires_grad=True)
    base_prob = torch.rand(n)
    target = (torch.rand(n) > 0.5).float()
    out = loss(
        refined_logit=refined,
        base_prob=base_prob,
        target=target,
        valid=torch.ones(n, dtype=torch.bool),
        batch_index=torch.randint(0, 2, (n,)),
    )
    out["rank"].backward()
    assert refined.grad is not None and torch.isfinite(refined.grad).all()
    assert base_prob.grad is None


def test_legacy_arguments_are_ignored_for_compatibility() -> None:
    """
    兼容旧调用签名: 传入 base_logit / p_best 不应报错, 且不影响数值。
    """
    loss = _make_loss(m_rank=0.5, topk=1)
    refined = torch.tensor([0.0, 1.0])
    base_prob = torch.tensor([0.2, 0.8])
    target = torch.tensor([1.0, 0.0])
    valid = torch.ones(2, dtype=torch.bool)
    batch_index = torch.zeros(2, dtype=torch.long)
    out_new = loss(
        refined_logit=refined,
        base_prob=base_prob,
        target=target,
        valid=valid,
        batch_index=batch_index,
    )
    out_legacy = loss(
        refined_logit=refined,
        base_prob=base_prob,
        target=target,
        valid=valid,
        batch_index=batch_index,
        base_logit=torch.tensor([9.0, -9.0]),
        p_best=0.95,
    )
    assert torch.allclose(out_new["rank"], out_legacy["rank"])
