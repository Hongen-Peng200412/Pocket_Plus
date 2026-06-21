from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from src.model.pseudo_atoms import PseudoAtomLayout, inject_pseudo_atoms
from src.model.sparse_refine.anchor_sampler import SparseAnchorSampler
from src.model.sparse_refine.candidate_set import SparseCandidateSetBuilder
from src.model.sparse_refine.density_cube import DensityCubeEncoder
from src.model.sparse_refine.interpolation import AnchorToCandidateKnnSearch
from src.model.sparse_refine.sparse_refine_head import SparseRefineHead
from src.model.stage1_model import VolumePointStage1Model


class _VoxelBackboneStub(nn.Module):
    """
    测试用 voxel backbone stub。

    输入参数:
        - channels: int, voxel feature 通道数
        - ligand_logit_dim: int, ligand logits 通道数
    """

    def __init__(self, channels: int = 2, ligand_logit_dim: int = 3) -> None:
        super().__init__()
        self.return_feature_keys = ("feat", "voxel_final")
        self.feature_channels_by_name = {"feat": channels, "voxel_final": channels}
        self.voxel_ligand_logit_dim = int(ligand_logit_dim)
        self.voxel_aux_logit_dim = 1
        self.channels = int(channels)

    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
        return_feature_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        """
        返回最小 voxel 输出字典。

        输入参数:
            - voxel_grid: torch.Tensor, (B,C,D,H,W), voxel 输入
            - recycle_in: torch.Tensor | None, recycle 输入
            - return_feature_keys: tuple[str, ...], 请求返回的 voxel feature 名称

        输出:
            - output: dict[str, Any], voxel backbone 输出字段
        """
        del recycle_in, return_feature_keys
        batch_size = int(voxel_grid.shape[0])
        voxel_final = voxel_grid.new_zeros((batch_size, self.channels, 2, 2, 2))
        ligand_logits = voxel_grid.new_zeros((batch_size, self.voxel_ligand_logit_dim, 2, 2, 2))
        return {
            "voxel_features": {"feat": voxel_final, "voxel_final": voxel_final},
            "voxel_logits_aux": voxel_grid.new_zeros((batch_size, 1, 2, 2, 2)),
            "voxel_logits_ligand": ligand_logits,
            "voxel_recycle_out": voxel_grid.new_zeros((batch_size, 1, 2, 2, 2)),
        }


class _PointBackboneStub(nn.Module):
    """
    测试用 point backbone stub。
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend = "zeros"
        self.out_channels = 4
        self.feature_channels_by_name = {"point_feat": 4}
        self.atom_feature_dim = 2

    def build_zeros_output(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        return_feature_names: tuple[str, ...],
        pseudo_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        构造与 Stage1PointBackbone zeros 后端一致的输出。

        输入参数:
            - atom_feat: torch.Tensor, (N,F_atom), 输入点特征
            - atom_coord_centered_world: torch.Tensor, (N,3), 点坐标
            - atom_batch_index: torch.Tensor, (N,), 点所属 BOX
            - atom_offsets: torch.Tensor, (B,), 点 offset
            - return_feature_names: tuple[str, ...], 请求导出的点变量名
            - pseudo_mask: torch.Tensor | None, mixed 点类型掩码

        输出:
            - output: dict[str, Any], point backbone 输出字段
        """
        del return_feature_names
        point_feat = atom_feat.new_zeros((int(atom_feat.shape[0]), self.out_channels))
        point_state: dict[str, Any] = {
            "coord": atom_coord_centered_world,
            "batch": atom_batch_index,
            "offset": atom_offsets,
            "grid_size": 1.0,
        }
        if pseudo_mask is not None:
            point_state["pseudo_mask"] = pseudo_mask
        return {
            "point_feat": point_feat,
            "point_state": point_state,
            "point_recycle_out": point_feat,
            "point_feature_dict": {"point_feat": point_feat},
        }


def _candidate_builder(
    class_ids: list[int],
    warmup_topc: list[int],
    max_count: list[int],
) -> SparseCandidateSetBuilder:
    """
    构造 adaptive_threshold candidate builder。

    输入参数:
        - class_ids: list[int], 候选类别 ID
        - warmup_topc: list[int], warmup 每类 top-C
        - max_count: list[int], 每类候选上限

    输出:
        - builder: SparseCandidateSetBuilder, 测试实例
    """
    return SparseCandidateSetBuilder(class_ids, warmup_topc, [2.0 for _ in class_ids], max_count, [0 for _ in class_ids], "adaptive_threshold")


def _base_model_kwargs() -> dict[str, Any]:
    """
    返回 VolumePointStage1Model 的当前显式基础参数。

    输出:
        - kwargs: dict[str, Any], 可直接展开给 VolumePointStage1Model
    """
    return {
        "voxel_backbone": _VoxelBackboneStub(),
        "point_backbone": _PointBackboneStub(),
        "point_fusion_map": {},
        "point_fusion_modes": (),
        "sampler_modes": (),
        "fusion_mlp_ratio": 1.0,
        "fusion_proj_drop": 0.0,
        "atom_head_hidden_dim": 8,
        "atom_head_interaction_radius": 4.0,
        "atom_head_interaction_max_neighbors": 8,
        "atom_head_interaction_num_heads": 2,
        "atom_head_interaction_detach_source_feat": True,
        "atom_logit_dim": 1,
        "enable_recycling": True,
        "max_recycles": 1,
        "randomize_recycles": False,
        "detach_recycle_states": True,
        "act_layer_name": "gelu",
        "ffn_type": "mlp",
        "enable_atom_head": True,
        "embed_head": None,
        "pseudo_atom_cfg": None,
        "prior_prob": None,
        "prior_probs": None,
        "online_pdb_feature": False,
        "online_pdb_feature_reduce": "sum",
        "online_pdb_feature_dim": 2,
    }


def _make_model(**overrides: Any) -> VolumePointStage1Model:
    """
    构造最小 VolumePointStage1Model 测试实例。

    输入参数:
        - overrides: Any, 覆盖 _base_model_kwargs 的参数

    输出:
        - model: VolumePointStage1Model, stub backbone 版本
    """
    kwargs = _base_model_kwargs()
    kwargs.update(overrides)
    return VolumePointStage1Model(**kwargs)


def _make_batch() -> dict[str, torch.Tensor]:
    """
    构造单 BOX real-only batch。

    输出:
        - batch: dict[str, torch.Tensor], N=3 的 Stage1 输入 batch
    """
    return {
        "voxel_grid": torch.zeros(1, 1, 2, 2, 2),
        "box_origin_world": torch.zeros(1, 3),
        "box_shape_zyx": torch.tensor([[2, 2, 2]], dtype=torch.long),
        "voxel_size_world": torch.ones(1, 3),
        "atom_feat": torch.randn(3, 2),
        "atom_coord_centered_world": torch.randn(3, 3),
        "atom_coord_local_voxel": torch.rand(3, 3),
        "atom_coord_world": torch.randn(3, 3),
        "atom_batch_index": torch.zeros(3, dtype=torch.long),
        "atom_offsets": torch.tensor([3], dtype=torch.long),
        "atom_counts": torch.tensor([3], dtype=torch.long),
        "atom_label": torch.tensor([1, 0, 1], dtype=torch.long),
        "atom_is_in_core_box": torch.tensor([True, True, False]),
        "atom_global_indices": torch.tensor([10, 11, 12], dtype=torch.long),
    }


def test_stage1_model_holds_single_atom_head_member() -> None:
    """
    验证主模型只持有 self.atom_head，不再暴露旧 atom head 成员。
    """
    model = _make_model(enable_atom_head=True)

    assert model.atom_head is not None
    assert not hasattr(model, "atom_token_proj")
    assert not hasattr(model, "atom_attention_stack")
    assert not hasattr(model, "atom_logit_head")


def test_stage1_model_rejects_non_null_legacy_pseudo_atom_cfg() -> None:
    """
    验证 legacy pseudo_atom_cfg 非空时 fail-fast。
    """
    with pytest.raises(ValueError, match="pseudo_atom_cfg"):
        _make_model(pseudo_atom_cfg={"base_count": 1})


def test_prepare_pseudo_batch_called_only_on_final_recycle() -> None:
    """
    验证 _prepare_pseudo_batch 只在最后一次 recycle 调用。
    """
    model = _make_model(enable_atom_head=False, max_recycles=3)
    call_indices: list[int] = []
    recycle_counter = {"idx": -1}
    original_run_voxel = model._run_voxel_backbone

    def wrapped_run_voxel(voxel_input: torch.Tensor, voxel_recycle_in: torch.Tensor | None) -> dict[str, Any]:
        recycle_counter["idx"] += 1
        return original_run_voxel(voxel_input, voxel_recycle_in)

    def wrapped_prepare(
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], PseudoAtomLayout | None, dict[str, Any]]:
        del voxel_output_dict
        call_indices.append(recycle_counter["idx"])
        return batch, None, {}

    model._run_voxel_backbone = wrapped_run_voxel  # type: ignore[method-assign]
    model._prepare_pseudo_batch = wrapped_prepare  # type: ignore[method-assign]
    outputs = model(_make_batch())

    assert outputs["recycle_passes_used"] == 3
    assert call_indices == [2]


def test_final_mixed_point_output_is_trimmed_only_for_real_outputs() -> None:
    """
    验证 final mixed point 输出只在 point_outputs/recycle 输出处裁剪 real-only 视图。
    """
    model = _make_model(enable_atom_head=False, max_recycles=2)

    def final_prepare(
        batch: dict[str, Any],
        voxel_output_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], PseudoAtomLayout | None, dict[str, Any]]:
        del voxel_output_dict
        pseudo_dict = {
            "pseudo_counts": torch.tensor([2], dtype=torch.long),
            "pseudo_batch_index": torch.tensor([0, 0], dtype=torch.long),
            "pseudo_feat": torch.randn(2, 2),
            "pseudo_coord_centered_world": torch.randn(2, 3),
            "pseudo_coord_local_voxel": torch.randn(2, 3),
            "pseudo_coord_world": torch.randn(2, 3),
        }
        mixed_batch, layout = inject_pseudo_atoms(batch, pseudo_dict)
        return mixed_batch, layout, {"pseudo_marker": torch.tensor([1])}

    model._prepare_pseudo_batch = final_prepare  # type: ignore[method-assign]
    outputs = model(_make_batch())

    assert outputs["fused_point_feat"].shape[0] == 5
    assert outputs["point_state"]["coord"].shape[0] == 5
    assert outputs["point_outputs"]["point_feat"].shape[0] == 3
    assert outputs["point_recycle_out"].shape[0] == 3
    assert outputs["atom_target"].shape[0] == 3
    assert outputs["atom_is_in_core_box"].shape[0] == 3
    assert outputs["pseudo_marker"].tolist() == [1]


def test_prepare_pseudo_batch_outputs_candidates_without_pseudo_layout() -> None:
    """
    验证 candidate builder 只输出 C 字段，不注入 P anchor。
    """
    model = _make_model(enable_atom_head=False)
    model.candidate_set_builder = _candidate_builder([1, 2], [1, 1], [10, 10])
    model.set_sparse_candidate_runtime(global_step=0, candidate_warmup_steps=10, allow_warmup_fixed_topk=True)

    outputs = model(_make_batch())

    assert outputs["candidate_voxel_zyx"].shape[1] == 3
    assert outputs["candidate_counts"].tolist() == [1]
    assert outputs["point_outputs"]["point_feat"].shape[0] == 3
    assert outputs["atom_target"].shape[0] == 3
    assert outputs["pseudo_feat_before_interaction"] is None


def test_stage1_forward_routes_unique_candidates_before_anchor_sampling() -> None:
    """
    端到端验证 Stage1 在进入 sampler 前仅保留唯一 C，P 继承其路由类别。
    """
    model = _make_model(enable_atom_head=False)
    model.candidate_set_builder = _candidate_builder([1, 2], [8, 8], [8, 8])
    model.anchor_sampler = SparseAnchorSampler([1, 2], [8, 8], "topk_nms", 1.0, 8192, 0, False)
    model.density_cube_encoder = DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0)
    model.set_sparse_candidate_runtime(global_step=0, candidate_warmup_steps=10, allow_warmup_fixed_topk=True)

    outputs = model(_make_batch())

    assert int(outputs["candidate_counts"].sum().item()) == 8
    assert int(outputs["candidate_counts_by_class"].sum().item()) == 8
    assert int(outputs["anchor_counts"].sum().item()) == 8
    assert int(outputs["anchor_counts_by_class"].sum().item()) == 8
    assert outputs["fused_point_feat"].shape[0] == 11


def test_anchor_class_conditioning_breaks_same_voxel_class_symmetry() -> None:
    """
    验证 P 来源类别 embedding 会打破同 voxel 多类别 P 的初始特征对称性。
    """
    model = _make_model(
        enable_atom_head=False,
        candidate_set_cfg=_candidate_builder([1, 2], [8, 8], [8, 8]),
        anchor_sampler_cfg=SparseAnchorSampler([1, 2], [8, 8], "topk_nms", 1.0, 8192, 0, False),
        density_cube_cfg=DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0),
        anchor_class_conditioning_cfg={"mode": "add_embedding", "init_std": 0.02},
    )
    model._anchor_class_ids = torch.tensor([1, 2], dtype=torch.long)
    model.anchor_class_embedding = nn.Embedding(2, 2)
    with torch.no_grad():
        model.anchor_class_embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    conditioned = model._condition_anchor_pseudo_feat(torch.zeros(2, 2), torch.tensor([1, 2], dtype=torch.long))

    assert conditioned.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_prepare_pseudo_batch_injects_anchor_pseudo_atoms() -> None:
    """
    验证 candidate builder + anchor sampler + density cube 启用后注入 mixed P anchors。
    """
    model = _make_model(enable_atom_head=True)
    model.candidate_set_builder = _candidate_builder([1, 2], [1, 1], [10, 10])
    model.anchor_sampler = SparseAnchorSampler([1, 2], [1, 1], "topk_nms", 1.0, 8192, 0, False)
    model.density_cube_encoder = DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0)
    model.set_sparse_candidate_runtime(global_step=0, candidate_warmup_steps=10, allow_warmup_fixed_topk=True)

    outputs = model(_make_batch())

    assert outputs["anchor_voxel_zyx"].shape[1] == 3
    assert outputs["anchor_counts_by_class"].shape == (1, 2)
    assert outputs["fused_point_feat"].shape[0] == 3 + int(outputs["anchor_counts"].sum().item())
    assert outputs["point_outputs"]["point_feat"].shape[0] == 3
    assert outputs["atom_logits"].shape[0] == 3
    assert outputs["atom_target"].shape[0] == 3
    assert outputs["pseudo_feat_after_interaction"].shape[0] == int(outputs["anchor_counts"].sum().item())
    assert outputs["pseudo_logits"].shape[0] == int(outputs["anchor_counts"].sum().item())


def test_stage1_final_sparse_refine_outputs_candidate_logits() -> None:
    """
    验证 final atom head 后执行 P -> C 聚合并输出 refined candidate logits。
    """
    model = _make_model(enable_atom_head=True)
    model.candidate_set_builder = _candidate_builder([1, 2], [2, 2], [2, 2])
    model.anchor_sampler = SparseAnchorSampler([1, 2], [2, 2], "topk_nms", 1.0, 8192, 0, False)
    model.density_cube_encoder = DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0)
    model.anchor_to_candidate = AnchorToCandidateKnnSearch("knn_message", 3, True, 8192)
    model.sparse_refine_head = SparseRefineHead(
        mode="residual",
        edge_weight_activation="sigmoid",
        distance_weight={"mode": "softmax_negative_squared_distance", "temperature": 1.0, "learnable": False},
        message_dim=4,
        edge_hidden_dim=4,
        hidden_dim=6,
        num_layers=2,
        logit_dim=3,
        C_voxel_backbone_dim=2,
        P_final_point_dim=4,
        P_after_interaction_dim=4,
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
        zero_init_residual=True,
    )
    model.set_sparse_candidate_runtime(global_step=0, candidate_warmup_steps=10, allow_warmup_fixed_topk=True)

    outputs = model(_make_batch())

    assert outputs["ligand_refine_logits_C"].shape == outputs["candidate_logits"].shape
    torch.testing.assert_close(outputs["ligand_refine_logits_C"], outputs["candidate_logits"])
    assert outputs["candidate_message_valid_mask"].shape == (int(outputs["candidate_counts"].sum().item()),)
    assert "C_voxel_backbone_feat" not in outputs


def test_stage1_model_anchor_sampler_requires_candidate_builder() -> None:
    """
    验证启用 anchor sampler 但关闭 candidate builder 时构造 fail-fast。
    """
    with pytest.raises(ValueError, match="candidate_set_builder"):
        _make_model(
            enable_atom_head=False,
            anchor_sampler_cfg=SparseAnchorSampler([1], [1], "topk_nms", 1.0, 8192, 0, False),
            density_cube_cfg=DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0),
        )


def test_stage1_model_anchor_sampler_requires_density_cube() -> None:
    """
    验证启用 anchor sampler 但关闭 density cube 时构造 fail-fast。
    """
    with pytest.raises(ValueError, match="density_cube_encoder"):
        _make_model(
            enable_atom_head=False,
            candidate_set_cfg=_candidate_builder([1], [1], [10]),
            anchor_sampler_cfg=SparseAnchorSampler([1], [1], "topk_nms", 1.0, 8192, 0, False),
        )


def test_stage1_model_density_out_dim_checked_at_init() -> None:
    """
    验证 density cube out_dim 与 point_backbone.atom_feature_dim 不一致时构造 fail-fast。
    """
    with pytest.raises(ValueError, match="out_dim"):
        _make_model(
            enable_atom_head=False,
            candidate_set_cfg=_candidate_builder([1], [1], [10]),
            anchor_sampler_cfg=SparseAnchorSampler([1], [1], "topk_nms", 1.0, 8192, 0, False),
            density_cube_cfg=DensityCubeEncoder(1, 3, 4, 1, 3, "conv_gap", "group", 2, "silu", 2, 0),
        )


def test_stage1_model_candidate_builder_fails_without_ligand_logits() -> None:
    """
    验证 builder 启用但 voxel_logits_ligand 缺失时 fail-fast。
    """
    model = _make_model(enable_atom_head=False)
    model.candidate_set_builder = _candidate_builder([1], [1], [10])

    with pytest.raises(RuntimeError, match="voxel_logits_ligand"):
        model._prepare_pseudo_batch(_make_batch(), {"voxel_logits_ligand": None})


def test_atom_supervision_outputs_are_real_only_aligned() -> None:
    """
    构造 fake mixed final outputs，验证 atom supervised 字段裁成 real-only。
    """
    model = _make_model(enable_atom_head=True)
    real_batch = _make_batch()
    pseudo_dict = {
        "pseudo_counts": torch.tensor([2], dtype=torch.long),
        "pseudo_batch_index": torch.tensor([0, 0], dtype=torch.long),
        "pseudo_feat": torch.randn(2, 2),
        "pseudo_coord_centered_world": torch.randn(2, 3),
        "pseudo_coord_local_voxel": torch.randn(2, 3),
        "pseudo_coord_world": torch.randn(2, 3),
    }
    mixed_batch, layout = inject_pseudo_atoms(real_batch, pseudo_dict)
    outputs: dict[str, Any] = {
        "fused_point_feat": torch.randn(5, 4),
        "point_state": {
            "coord": mixed_batch["atom_coord_centered_world"],
            "batch": mixed_batch["atom_batch_index"],
            "offset": mixed_batch["atom_offsets"],
            "grid_size": 1.0,
        },
    }

    model._run_atom_head(outputs, atom_head_batch=mixed_batch, pseudo_layout=layout)

    assert outputs["atom_logits"].shape[0] == 3
    assert outputs["atom_target"].shape[0] == 3
    assert outputs["atom_is_in_core_box"].shape[0] == 3
    assert outputs["pseudo_feat_after_interaction"].shape[0] == 2
