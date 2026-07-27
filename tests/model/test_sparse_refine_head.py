from __future__ import annotations

import torch

from src.model.sparse_refine.sparse_refine_head import SparseRefineHead


def _make_head(
    mode: str,
    logit_dim: int,
    zero_init_residual: bool,
    enable_interface_norm: bool,
) -> SparseRefineHead:
    """
    构造测试用 sparse refine head. 

    输入参数:
        - mode: str, 输出模式 direct 或 residual
        - logit_dim: int, logits 通道数
        - zero_init_residual: bool, residual 增量末层是否零初始化
        - enable_interface_norm: bool, 是否启用接口 LayerNorm

    输出:
        - head: SparseRefineHead, 测试实例
    """
    return SparseRefineHead(
        mode=mode,
        edge_weight_activation="sigmoid",
        distance_weight={"mode": "softmax_negative_squared_distance", "temperature": 1.0, "learnable": False},
        message_dim=4,
        edge_hidden_dim=4,
        hidden_dim=6,
        num_layers=2,
        logit_dim=logit_dim,
        C_voxel_backbone_dim=2,
        P_final_point_dim=3,
        P_after_interaction_dim=2,
        P_voxel_backbone_dim=2,
        inputs={
            "use_C_voxel_logits": True,
            "use_C_voxel_backbone_feat": True,
            "use_P_final_point_feat": True,
            "use_P_after_interaction_feat": True,
            "use_P_voxel_backbone_feat": True,
            "use_relative_coords": True,
            "use_candidate_class_embedding": False,
        },
        candidate_class_ids=(1, 2),
        candidate_class_embedding_dim=2,
        zero_init_residual=zero_init_residual,
        enable_interface_norm=enable_interface_norm,
    )


def _inputs(logit_dim: int) -> dict[str, torch.Tensor]:
    """
    构造含一条有效消息和一条无消息 C 的输入. 

    输入参数:
        - logit_dim: int, C logits 通道数

    输出:
        - inputs: dict[str, torch.Tensor], SparseRefineHead.forward 输入
    """
    return {
        "voxel_logits": torch.randn(2, logit_dim, requires_grad=True),
        "C_voxel_backbone_feat": torch.randn(2, 2),
        "P_final_point_feat": torch.randn(1, 3),
        "P_after_interaction_feat": torch.randn(1, 2),
        "P_voxel_backbone_feat": torch.randn(1, 2),
        "anchor_class": torch.tensor([1]),
        "candidate_neighbor_index": torch.tensor([[0, 0], [0, 0]]),
        "candidate_neighbor_squared_distance": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "candidate_neighbor_relative_coords": torch.zeros(2, 2, 3),
        "candidate_neighbor_valid_mask": torch.tensor([[True, True], [False, False]]),
    }


def test_residual_zero_init_starts_from_voxel_logits_and_marks_missing_message() -> None:
    """
    验证 residual 零初始化时输出等于 base logits, 且无邻居 C 被标记为未收到消息. 
    """
    inputs = _inputs(logit_dim=1)
    output = _make_head("residual", 1, True, False)(**inputs)

    torch.testing.assert_close(output["ligand_refine_logits_C"], inputs["voxel_logits"])
    assert output["candidate_message_valid_mask"].tolist() == [True, False]
    assert "candidate_message_delta" not in output


def test_voxel_logits_gradient_flows_through_without_internal_detach() -> None:
    """
    验证 head 内部不再 detach voxel_logits, detach 职责属于调用方. 
    """
    inputs = _inputs(logit_dim=1)
    _make_head("residual", 1, False, False)(**inputs)["ligand_refine_logits_C"].sum().backward()

    assert inputs["voxel_logits"].grad is not None


def test_interface_norm_builds_current_source_modules_and_forward_runs() -> None:
    """
    验证 enable_interface_norm 开时构造 P_final/P_after/P_voxel/C_voxel LayerNorm. 
    """
    head_on = _make_head("residual", 1, False, True)
    assert head_on.interface_norm_P_final is not None
    assert head_on.interface_norm_P_after is not None
    assert head_on.interface_norm_P_voxel is not None
    assert head_on.interface_norm_C_voxel_edge is not None

    inputs = _inputs(logit_dim=1)
    output = head_on(**inputs)
    output["ligand_refine_logits_C"].sum().backward()
    assert output["ligand_refine_logits_C"].shape == (2, 1)
    assert inputs["voxel_logits"].grad is not None

    head_off = _make_head("residual", 1, True, False)
    assert head_off.interface_norm_P_final is None
    assert head_off.interface_norm_P_after is None
    assert head_off.interface_norm_C_voxel_edge is None


def test_direct_multiclass_preserves_logit_dimension() -> None:
    """
    验证 direct 多分类 head 输出完整 logits 通道. 
    """
    output = _make_head("direct", 3, False, False)(**_inputs(logit_dim=3))

    assert output["ligand_refine_logits_C"].shape == (2, 3)


def test_all_candidates_without_neighbors_stay_finite() -> None:
    """
    验证所有 C 都无有效邻居时 mask 全 False, 输出仍保持有限值. 
    """
    inputs = _inputs(logit_dim=1)
    inputs["candidate_neighbor_valid_mask"] = torch.zeros_like(inputs["candidate_neighbor_valid_mask"], dtype=torch.bool)
    output = _make_head("direct", 1, False, False)(**inputs)

    assert output["candidate_message_valid_mask"].tolist() == [False, False]
    assert bool(torch.isfinite(output["ligand_refine_logits_C"]).all())
