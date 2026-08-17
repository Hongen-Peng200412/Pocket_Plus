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
from src.model.stage1_embed_head import Stage1EmbedHead
from src.model.stage1_model import VolumePointStage1Model


class _VoxelBackboneStub(nn.Module):
    """
    测试用 voxel backbone stub. 

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
        返回最小 voxel 输出字典. 

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
    测试用 point backbone stub. 
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
        构造与 Stage1PointBackbone zeros 后端一致的输出. 

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


class _VoxelFeatureBackboneStub(_VoxelBackboneStub):
    """返回 centered 契约五路 V 特征的轻量 voxel stub. """

    def __init__(self) -> None:
        super().__init__(channels=2, ligand_logit_dim=3)
        self.return_feature_keys = (
            "voxel_ds_2",
            "voxel_ds_3",
            "voxel_ds_4",
            "voxel_c4",
            "voxel_final",
        )
        self.feature_channels_by_name = {key: 2 for key in self.return_feature_keys}

    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
        return_feature_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        batch_size = int(voxel_grid.shape[0])
        final = voxel_grid.new_zeros((batch_size, 2, 2, 2, 2))
        features = {
            "voxel_ds_2": voxel_grid.new_zeros((batch_size, 2, 2, 2, 2)),
            "voxel_ds_3": voxel_grid.new_zeros((batch_size, 2, 1, 1, 1)),
            "voxel_ds_4": voxel_grid.new_zeros((batch_size, 2, 1, 1, 1)),
            "voxel_c4": voxel_grid.new_zeros((batch_size, 2, 1, 1, 1)),
            "voxel_final": final,
        }
        return {
            "voxel_features": {key: features[key] for key in return_feature_keys},
            "voxel_logits_aux": voxel_grid.new_zeros((batch_size, 1, 2, 2, 2)),
            "voxel_logits_ligand": voxel_grid.new_zeros((batch_size, 3, 2, 2, 2)),
            "voxel_recycle_out": voxel_grid.new_zeros((batch_size, 1, 2, 2, 2)),
        }


class _InputSensitiveVoxelBackboneStub(_VoxelFeatureBackboneStub):
    """让 logits 同时依赖 voxel 输入与 recycle 的等价性测试 stub. """

    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
        return_feature_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        base = voxel_grid.sum(dim=1, keepdim=True)
        if recycle_in is not None:
            base = base + recycle_in
        final = base.repeat(1, 2, 1, 1, 1)
        features = {key: final for key in return_feature_keys}
        return {
            "voxel_features": features,
            "voxel_logits_aux": base,
            "voxel_logits_ligand": base.repeat(1, 3, 1, 1, 1),
            "voxel_recycle_out": base,
        }


class _EmbedHeadStub(nn.Module):
    """保持行对齐并显式提供 A L1 的 point-side embed stub. """

    has_point_output = True
    has_voxel_output = False

    def forward(self, **batch: torch.Tensor) -> dict[str, Any]:
        atom_feat = batch["atom_feat"]
        keep = batch["atom_is_in_core_box"].bool()
        kept_batch = batch["atom_batch_index"][keep]
        batch_size = int(batch["atom_offsets"].shape[0])
        counts = torch.bincount(kept_batch, minlength=batch_size)
        return {
            "voxel_pdb_embed_grid": None,
            "embed_point_feat": atom_feat[keep],
            "atom_feat": atom_feat[keep],
            "atom_coord_centered_world": batch["atom_coord_centered_world"][keep],
            "atom_batch_index": kept_batch,
            "atom_offsets": counts.cumsum(dim=0),
            "atom_coord_local_voxel": batch["atom_coord_local_voxel"][keep],
            "atom_is_in_core_box": batch["atom_is_in_core_box"][keep],
            "global_keep_mask": keep,
        }


class _VoxelOnlyEmbedHeadStub(nn.Module):
    """为 ligand-area-only Find 测试提供不产生点特征的原子到体素嵌入。"""

    has_point_output = False
    has_voxel_output = True
    voxel_embed_as_tune = False

    def forward(self, **batch: torch.Tensor) -> dict[str, Any]:
        atom_feat = batch["atom_feat"]
        batch_size = int(batch["box_shape_zyx"].shape[0])
        voxel_grid = atom_feat.new_zeros((batch_size, 1, 2, 2, 2))
        voxel_grid[:, :, 0, 0, 0] = atom_feat.sum()
        return {
            "voxel_pdb_embed_grid": voxel_grid,
            "embed_point_feat": None,
            "atom_feat": atom_feat,
            "atom_coord_centered_world": batch["atom_coord_centered_world"],
            "atom_batch_index": batch["atom_batch_index"],
            "atom_offsets": batch["atom_offsets"],
            "atom_coord_local_voxel": batch["atom_coord_local_voxel"],
            "atom_is_in_core_box": batch["atom_is_in_core_box"],
            "global_keep_mask": torch.ones(
                atom_feat.shape[0], dtype=torch.bool, device=atom_feat.device
            ),
        }


def _candidate_builder(
    class_ids: list[int],
    warmup_topc: list[int],
    max_count: list[int],
) -> SparseCandidateSetBuilder:
    """
    构造 adaptive_threshold candidate builder. 

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
    返回 VolumePointStage1Model 的当前显式基础参数. 

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
    构造最小 VolumePointStage1Model 测试实例. 

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
    构造单 BOX real-only batch. 

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


def test_model_boundary_appends_backbone_flag_for_50d_embed_input() -> None:
    """Dataset 保持 49 维资产，模型输入边界按既有配置形成第 50 维。"""

    embed_head = _EmbedHeadStub()
    embed_head.atom_feature_dim = 50
    model = _make_model(embed_head=embed_head)
    batch = {
        "density_input": torch.zeros(1, 1, 2, 2, 2),
        "atom_feat": torch.zeros(2, 49),
        "atom_is_backbone": torch.tensor([True, False]),
        "atom_offsets": torch.tensor([0, 2], dtype=torch.long),
    }

    canonical = model._canonicalize_stage1_batch(batch)

    assert canonical["atom_feat"].shape == (2, 50)
    assert canonical["atom_feat"][:, -1].tolist() == [1.0, 0.0]
    assert canonical["atom_offsets"].tolist() == [2]
    assert "atom_is_backbone" not in canonical


@pytest.mark.parametrize("feature_source", ["point_backbone", "online_pdb_feature_dim"])
def test_model_boundary_uses_documented_feature_dimension_fallbacks(
    feature_source: str,
) -> None:
    """embed head 缺席时依次采用 point backbone 与在线特征维数。"""

    model = _make_model(embed_head=None)
    if feature_source == "point_backbone":
        model.point_backbone.atom_feature_dim = 49
    else:
        model.point_backbone = None
        model.online_pdb_feature_dim = 49
    batch = {
        "density_input": torch.zeros(1, 1, 2, 2, 2),
        "atom_feat": torch.zeros(2, 49),
        "atom_is_backbone": torch.tensor([True, False]),
        "atom_offsets": torch.tensor([0, 2], dtype=torch.long),
    }

    canonical = model._canonicalize_stage1_batch(batch)

    assert canonical["atom_feat"].shape == (2, 49)
    assert canonical["atom_offsets"].tolist() == [2]
    assert "atom_is_backbone" not in canonical


def test_stage1_model_holds_single_atom_head_member() -> None:
    """
    验证主模型只持有 self.atom_head, 不再暴露旧 atom head 成员. 
    """
    model = _make_model(enable_atom_head=True)

    assert model.atom_head is not None
    assert not hasattr(model, "atom_token_proj")
    assert not hasattr(model, "atom_attention_stack")
    assert not hasattr(model, "atom_logit_head")


def test_stage1_model_rejects_non_null_legacy_pseudo_atom_cfg() -> None:
    """
    验证 legacy pseudo_atom_cfg 非空时 fail-fast. 
    """
    with pytest.raises(ValueError, match="pseudo_atom_cfg"):
        _make_model(pseudo_atom_cfg={"base_count": 1})


def test_prepare_pseudo_batch_called_only_on_final_recycle() -> None:
    """
    验证 _prepare_pseudo_batch 只在最后一次 recycle 调用. 
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
    验证 final mixed point 输出只在 point_outputs/recycle 输出处裁剪 real-only 视图. 
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
    验证 candidate builder 只输出 C 字段, 不注入 P anchor. 
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
    端到端验证 Stage1 在进入 sampler 前仅保留唯一 C, P 继承其路由类别. 
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
    验证 P 来源类别 embedding 会打破同 voxel 多类别 P 的初始特征对称性. 
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
    验证 candidate builder + anchor sampler + density cube 启用后注入 mixed P anchors. 
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
    验证 final atom head 后执行 P -> C 聚合并输出 refined candidate logits. 
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
    验证启用 anchor sampler 但关闭 candidate builder 时构造 fail-fast. 
    """
    with pytest.raises(ValueError, match="candidate_set_builder"):
        _make_model(
            enable_atom_head=False,
            anchor_sampler_cfg=SparseAnchorSampler([1], [1], "topk_nms", 1.0, 8192, 0, False),
            density_cube_cfg=DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0),
        )


def test_stage1_model_anchor_sampler_requires_density_cube() -> None:
    """
    验证启用 anchor sampler 但关闭 density cube 时构造 fail-fast. 
    """
    with pytest.raises(ValueError, match="density_cube_encoder"):
        _make_model(
            enable_atom_head=False,
            candidate_set_cfg=_candidate_builder([1], [1], [10]),
            anchor_sampler_cfg=SparseAnchorSampler([1], [1], "topk_nms", 1.0, 8192, 0, False),
        )


def test_stage1_model_density_out_dim_checked_at_init() -> None:
    """
    验证 density cube out_dim 与 point_backbone.atom_feature_dim 不一致时构造 fail-fast. 
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
    验证 builder 启用但 voxel_logits_ligand 缺失时 fail-fast. 
    """
    model = _make_model(enable_atom_head=False)
    model.candidate_set_builder = _candidate_builder([1], [1], [10])

    with pytest.raises(RuntimeError, match="voxel_logits_ligand"):
        model._prepare_pseudo_batch(_make_batch(), {"voxel_logits_ligand": None})


def test_atom_supervision_outputs_are_real_only_aligned() -> None:
    """
    构造 fake mixed final outputs, 验证 atom supervised 字段裁成 real-only. 
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


def test_full_forward_publishes_centered_v_a_p_features_from_true_sources() -> None:
    """验证完整 Find forward 的 V/A/P 直键来自既有真实层出口且逐实体对齐. """

    model = _make_model(
        voxel_backbone=_VoxelFeatureBackboneStub(),
        embed_head=_EmbedHeadStub(),
        enable_atom_head=True,
        max_recycles=3,
    )
    model.candidate_set_builder = _candidate_builder([1, 2], [1, 1], [1, 1])
    model.anchor_sampler = SparseAnchorSampler([1, 2], [1, 1], "topk_nms", 1.0, 8192, 0, False)
    model.density_cube_encoder = DensityCubeEncoder(1, 3, 4, 1, 2, "conv_gap", "group", 2, "silu", 2, 0)
    model.set_sparse_candidate_runtime(global_step=0, candidate_warmup_steps=10, allow_warmup_fixed_topk=True)
    model.eval()

    outputs = model(_make_batch())

    assert outputs["voxel_features"] is outputs["voxel_outputs"]["voxel_features"]
    assert tuple(outputs["voxel_features"]) == (
        "voxel_ds_2",
        "voxel_ds_3",
        "voxel_ds_4",
        "voxel_c4",
        "voxel_final",
    )
    assert outputs["voxel_features"]["voxel_final"].shape == (1, 2, 2, 2, 2)

    assert outputs["A_feat_L1"].shape == (2, 2)
    assert outputs["A_feat_L2"] is outputs["A_feat_L1"]
    assert outputs["A_feat_L3"] is outputs["real_feat_before_interaction"]
    assert outputs["A_feat_L4"] is outputs["real_feat_after_interaction"]
    assert outputs["A_feat_L3"].shape == outputs["A_feat_L4"].shape == (2, 4)
    assert outputs["atom_global_indices"].shape == (2,)

    pseudo_count = int(outputs["anchor_counts"].sum().item())
    assert outputs["P_feat_L2"] is outputs["pseudo_density_feat"]
    assert outputs["P_feat_L3"] is outputs["pseudo_feat_before_interaction"]
    assert outputs["P_feat_L4"] is outputs["pseudo_feat_after_interaction"]
    assert outputs["P_feat_L2"].shape == (pseudo_count, 2)
    assert outputs["P_feat_L3"].shape == outputs["P_feat_L4"].shape == (pseudo_count, 4)
    assert outputs["anchor_coord_local_voxel"].shape == (pseudo_count, 3)
    assert outputs["anchor_batch_index"].shape == (pseudo_count,)
    assert outputs["pseudo_logits"].shape[0] == pseudo_count


def test_unet_full_forward_publishes_only_five_v_features() -> None:
    """验证纯 voxel producer 导出五路 V, 同时不伪造任何 A/P 字段. """

    model = _make_model(
        voxel_backbone=_VoxelFeatureBackboneStub(),
        point_backbone=None,
        embed_head=None,
        enable_atom_head=False,
        max_recycles=3,
    )
    model.eval()
    outputs = model({"density_input": torch.zeros(1, 1, 2, 2, 2)})

    assert outputs["voxel_features"] is outputs["voxel_outputs"]["voxel_features"]
    assert set(outputs["voxel_features"]) == {
        "voxel_ds_2",
        "voxel_ds_3",
        "voxel_ds_4",
        "voxel_c4",
        "voxel_final",
    }
    assert not any(key.startswith("A_feat_") or key.startswith("P_feat_") for key in outputs)


def test_find0_voxel_only_matches_eval_forward_and_skips_point_path() -> None:
    """验证 Find_0 最短入口逐元素等价、固定三次 recycle 且不运行 point backbone. """

    model = _make_model(
        voxel_backbone=_InputSensitiveVoxelBackboneStub(),
        embed_head=_EmbedHeadStub(),
        enable_atom_head=False,
        online_pdb_feature=True,
        online_pdb_feature_reduce="sum",
        online_pdb_feature_dim=2,
        max_recycles=3,
    )
    model.eval()
    batch = _make_batch()
    batch["density_input"] = batch.pop("voxel_grid").requires_grad_(True)
    batch["atom_offsets"] = torch.tensor([0, 3], dtype=torch.long)
    point_calls = 0
    original_point_forward = model.point_backbone.build_zeros_output

    def counted_point_forward(**kwargs: Any) -> dict[str, Any]:
        nonlocal point_calls
        point_calls += 1
        return original_point_forward(**kwargs)

    model.point_backbone.build_zeros_output = counted_point_forward  # type: ignore[method-assign]
    full_logits = model(batch)["voxel_logits_ligand"]
    assert point_calls == 3
    point_calls = 0
    short_logits = model.forward_voxel_probability(batch)

    torch.testing.assert_close(short_logits, full_logits, rtol=0.0, atol=0.0)
    assert point_calls == 0
    short_logits.sum().backward()
    assert batch["density_input"].grad is not None
    assert torch.isfinite(batch["density_input"].grad).all()


def test_find1_voxel_only_matches_three_recycle_eval_forward_and_skips_point_path() -> None:
    """验证 Find_1 完整模型等价、三次 recycle 及短入口 point 分支零调用. """

    torch.manual_seed(29)
    embed_head = Stage1EmbedHead(
        atom_feature_dim=49,
        embed_hidden_dim=128,
        embed_voxel_out_channels=49,
        embed_point_out_channels=64,
        num_trunk_blocks=0,
        num_voxel_blocks=0,
        num_point_blocks=3,
        trunk_buffer_radii=(),
        voxel_buffer_radii=(),
        point_buffer_radii=(8.0, 4.0, 0.0),
        num_heads=4,
        patch_size=16,
        serialization_orders=("z",),
        shuffle_orders=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        scatter_reduce="sum",
        ffn_type="gated",
        mlp_ratio=3,
        act_layer_name="gelu",
        point_grid_size=0.25,
        cpe_impl="none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        drop_path=0.0,
        pre_norm=True,
        embed_residual_enabled=True,
        embed_point_gate_enabled=False,
        embed_voxel_gate_enabled=False,
        add_occupancy_channels=True,
        use_soft_splatting=True,
        use_centroid_encoding=True,
    )
    model = _make_model(
        voxel_backbone=_InputSensitiveVoxelBackboneStub(),
        embed_head=embed_head,
        enable_atom_head=False,
        max_recycles=3,
    )
    model.eval()
    density_input = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    atom_feat = torch.randn(4, 49, requires_grad=True)
    atom_coord_local = torch.tensor(
        [[1.25, 1.50, 1.75], [1.80, 1.20, 1.40], [15.50, 8.0, 8.0], [18.0, 8.0, 8.0]],
        dtype=torch.float32,
    )
    batch = {
        "density_input": density_input,
        "box_origin_world": torch.zeros(1, 3),
        "box_shape_zyx": torch.tensor([[16, 16, 16]], dtype=torch.long),
        "voxel_size_world": torch.ones(1, 3),
        "atom_feat": atom_feat,
        "atom_coord_centered_world": atom_coord_local - 8.0,
        "atom_coord_local_voxel": atom_coord_local,
        "atom_coord_world": atom_coord_local,
        "atom_batch_index": torch.zeros(4, dtype=torch.long),
        "atom_offsets": torch.tensor([0, 4], dtype=torch.long),
        "atom_counts": torch.tensor([4], dtype=torch.long),
        "atom_is_in_core_box": torch.tensor([True, True, True, False]),
        "atom_global_indices": torch.arange(4),
    }

    embed_point_calls = 0
    point_backbone_calls = 0
    voxel_calls = 0
    original_embed_point = embed_head._run_blocks_with_trim
    original_point_backbone = model.point_backbone.build_zeros_output
    original_voxel = model._run_voxel_backbone

    def counted_embed_point(*args: Any, **kwargs: Any) -> Any:
        nonlocal embed_point_calls
        embed_point_calls += 1
        return original_embed_point(*args, **kwargs)

    def counted_point_backbone(**kwargs: Any) -> dict[str, Any]:
        nonlocal point_backbone_calls
        point_backbone_calls += 1
        return original_point_backbone(**kwargs)

    def counted_voxel(voxel_input: torch.Tensor, recycle_in: torch.Tensor | None) -> dict[str, Any]:
        nonlocal voxel_calls
        voxel_calls += 1
        return original_voxel(voxel_input, recycle_in)

    embed_head._run_blocks_with_trim = counted_embed_point  # type: ignore[method-assign]
    model.point_backbone.build_zeros_output = counted_point_backbone  # type: ignore[method-assign]
    model._run_voxel_backbone = counted_voxel  # type: ignore[method-assign]
    with torch.no_grad():
        full_logits = model(batch)["voxel_logits_ligand"]
    # embed head 在 recycle 外运行一次；独立的 voxel blocks 与 point blocks 各调用一次。
    # 最短入口进一步跳过 point blocks，只保留非块式 voxel 构造。
    assert embed_point_calls == 2
    assert point_backbone_calls == 3
    assert voxel_calls == 3

    embed_point_calls = 0
    point_backbone_calls = 0
    voxel_calls = 0
    short_logits = model.forward_voxel_probability(batch)

    torch.testing.assert_close(short_logits, full_logits, rtol=0.0, atol=0.0)
    assert embed_point_calls == 0
    assert point_backbone_calls == 0
    assert voxel_calls == 3
    short_logits.square().mean().backward()
    assert density_input.grad is not None and torch.isfinite(density_input.grad).all()
    assert atom_feat.grad is not None and torch.isfinite(atom_feat.grad).all()


def test_unet_voxel_only_matches_eval_forward_and_backpropagates() -> None:
    """验证 unet_c1 最短入口与完整纯 voxel forward 的最终 logits 完全一致. """

    model = _make_model(
        voxel_backbone=_InputSensitiveVoxelBackboneStub(),
        point_backbone=None,
        embed_head=None,
        enable_atom_head=False,
        max_recycles=3,
    )
    model.eval()
    density_input = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
    batch = {"density_input": density_input}

    full_logits = model(batch)["voxel_logits_ligand"]
    short_logits = model.forward_voxel_probability(batch)

    torch.testing.assert_close(short_logits, full_logits, rtol=0.0, atol=0.0)
    short_logits.square().mean().backward()
    assert density_input.grad is not None
    assert torch.isfinite(density_input.grad).all()
