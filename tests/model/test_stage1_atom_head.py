from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.model.stage1_atom_head as atom_head_module
from src.model.stage1_atom_head import GeometricCrossAttention, Stage1AtomHead


def _make_point_state(num_points: int) -> dict[str, torch.Tensor | float]:
    """
    构造单 BOX point_state。

    输入参数:
        - num_points: int, 点数 N_all

    输出:
        - point_state: dict[str, torch.Tensor | float], 与 Stage1AtomHead.forward 契约一致的点状态
    """
    return {
        "batch": torch.zeros(num_points, dtype=torch.long),
        "coord": torch.arange(num_points, dtype=torch.float32).reshape(-1, 1).repeat(1, 3) * 0.1,
        "offset": torch.tensor([num_points], dtype=torch.long),
        "grid_size": 1.0,
    }


def _patch_radius_and_scatter(monkeypatch: pytest.MonkeyPatch, edge_index: torch.Tensor) -> None:
    """
    为几何 cross-attn 注入确定性的 radius/scatter 实现。

    输入参数:
        - monkeypatch: pytest.MonkeyPatch, pytest monkeypatch fixture
        - edge_index: torch.Tensor, (2,E), radius 返回的 query/source 边索引

    输出:
        - None, 原地替换 stage1_atom_head 模块内依赖
    """

    def fake_radius(**kwargs: object) -> torch.Tensor:
        source_coord = kwargs["x"]
        query_coord = kwargs["y"]
        assert torch.is_tensor(source_coord)
        assert torch.is_tensor(query_coord)
        return edge_index.to(device=query_coord.device)

    def fake_scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        out = torch.empty_like(src)
        for group_id in torch.unique(index):
            mask = index == group_id
            out[mask] = torch.softmax(src[mask], dim=0)
        return out

    def fake_scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
        assert dim == 0
        out = src.new_zeros((int(dim_size), src.shape[1]))
        out.index_add_(0, index, src)
        return out

    monkeypatch.setattr(atom_head_module, "torch_cluster", SimpleNamespace(radius=fake_radius))
    monkeypatch.setattr(
        atom_head_module,
        "torch_scatter",
        SimpleNamespace(scatter_softmax=fake_scatter_softmax, scatter_sum=fake_scatter_sum),
    )


def _make_head(
    point_channels: int,
    hidden_dim: int,
    atom_logit_dim: int,
    pseudo_ligand_logit_dim: int,
    prior_prob: float | None,
    prior_probs: list[float] | None,
    prior_prob_point_ligand: float | None,
) -> Stage1AtomHead:
    """
    构造当前 Stage1AtomHead 测试实例。

    输入参数:
        - point_channels: int, point backbone 输出通道数
        - hidden_dim: int, 分类尾部隐藏通道数
        - atom_logit_dim: int, real atom logits 通道数
        - pseudo_ligand_logit_dim: int, P ligand logits 通道数
        - prior_prob: float | None, real 单通道 sigmoid 先验
        - prior_probs: list[float] | None, real 多通道 softmax 先验
        - prior_prob_point_ligand: float | None, P 单通道 sigmoid 先验

    输出:
        - head: Stage1AtomHead, 测试实例
    """
    return Stage1AtomHead(
        point_channels=point_channels,
        hidden_dim=hidden_dim,
        atom_logit_dim=atom_logit_dim,
        pseudo_ligand_logit_dim=pseudo_ligand_logit_dim,
        act_layer=nn.GELU,
        interaction_radius=4.0,
        interaction_max_neighbors=8,
        interaction_num_heads=2,
        interaction_detach_source_feat=True,
        prior_prob=prior_prob,
        prior_probs=prior_probs,
        prior_prob_point_ligand=prior_prob_point_ligand,
    )


def test_real_only_outputs_current_contract() -> None:
    """
    验证 real-only 路径只产 real 特征/logits，P 字段为 None。
    """
    head = _make_head(6, 8, 3, 1, None, None, None)
    point_feat = torch.randn(5, 6)
    output = head(
        point_feat=point_feat,
        point_state=_make_point_state(5),
        atom_coord_centered_world=torch.randn(5, 3),
        pseudo_mask=None,
    )

    assert set(output) == {
        "real_feat_before_interaction",
        "real_feat_after_interaction",
        "pseudo_feat_before_interaction",
        "pseudo_feat_after_interaction",
        "atom_logits",
        "pseudo_logits",
    }
    torch.testing.assert_close(output["real_feat_before_interaction"], point_feat)
    torch.testing.assert_close(output["real_feat_after_interaction"], point_feat)
    assert output["atom_logits"].shape == (5, 3)
    assert output["pseudo_feat_before_interaction"] is None
    assert output["pseudo_feat_after_interaction"] is None
    assert output["pseudo_logits"] is None


def test_mixed_zero_initialized_cross_attention_keeps_before_after_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    验证 mixed 路径中 cross-attn 零初始化时 before/after 逐元素恒等。
    """
    _patch_radius_and_scatter(monkeypatch, torch.tensor([[0, 0], [0, 1]], dtype=torch.long))
    head = _make_head(6, 8, 2, 1, None, None, 0.1)
    point_feat = torch.randn(6, 6)
    pseudo_mask = torch.tensor([False, True, False, False, True, False])
    output = head(
        point_feat=point_feat,
        point_state=_make_point_state(6),
        atom_coord_centered_world=torch.randn(6, 3),
        pseudo_mask=pseudo_mask,
    )

    real_before = point_feat[~pseudo_mask]
    pseudo_before = point_feat[pseudo_mask]
    torch.testing.assert_close(output["real_feat_before_interaction"], real_before)
    torch.testing.assert_close(output["real_feat_after_interaction"], real_before)
    torch.testing.assert_close(output["pseudo_feat_before_interaction"], pseudo_before)
    torch.testing.assert_close(output["pseudo_feat_after_interaction"], pseudo_before)
    assert output["atom_logits"].shape == (4, 2)
    assert output["pseudo_logits"].shape == (2, 1)


def test_prior_bias_initialization() -> None:
    """
    验证 real 多分类先验与 P 单通道先验初始化到对应末层 bias。
    """
    head = _make_head(6, 8, 3, 1, None, [0.8, 0.1, 0.1], 0.2)

    expected_real = torch.tensor([math.log(0.8), math.log(0.1), math.log(0.1)])
    torch.testing.assert_close(head.real_atom_head[-1].bias.detach(), expected_real)
    expected_pseudo = -math.log((1.0 - 0.2) / 0.2)
    assert abs(float(head.pseudo_atom_head[-1].bias.item()) - expected_pseudo) < 1e-6


def test_all_pseudo_path_returns_empty_real_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    验证全 P mixed 输入时 real 输出为空，P 输出仍按当前契约存在。
    """
    _patch_radius_and_scatter(monkeypatch, torch.empty((2, 0), dtype=torch.long))
    head = _make_head(6, 8, 2, 1, None, None, None)
    point_feat = torch.randn(4, 6)
    output = head(
        point_feat=point_feat,
        point_state=_make_point_state(4),
        atom_coord_centered_world=torch.randn(4, 3),
        pseudo_mask=torch.ones(4, dtype=torch.bool),
    )

    assert output["real_feat_before_interaction"].shape == (0, 6)
    assert output["real_feat_after_interaction"].shape == (0, 6)
    assert output["atom_logits"].shape == (0, 2)
    torch.testing.assert_close(output["pseudo_feat_before_interaction"], point_feat)
    torch.testing.assert_close(output["pseudo_feat_after_interaction"], point_feat)
    assert output["pseudo_logits"].shape == (4, 1)


def test_geometric_cross_attention_uses_radius_source_query_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    验证 GeometricCrossAttention 调用 radius 时 x=source、y=query，并接受 source_bind_prob。
    """
    _patch_radius_and_scatter(monkeypatch, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    module = GeometricCrossAttention(
        query_channels=4,
        source_channels=4,
        num_heads=2,
        radius=2.0,
        max_neighbors=4,
        detach_source_feat=True,
        act_layer=nn.SiLU,
        use_source_bind_prob=True,
    )

    output = module(
        query_feat=torch.randn(2, 4),
        source_feat=torch.randn(3, 4),
        query_coord=torch.randn(2, 3),
        source_coord=torch.randn(3, 3),
        query_batch=torch.zeros(2, dtype=torch.long),
        source_batch=torch.zeros(3, dtype=torch.long),
        source_bind_prob=torch.rand(3, 1),
    )

    assert output.shape == (2, 4)
    torch.testing.assert_close(output, torch.zeros_like(output))
