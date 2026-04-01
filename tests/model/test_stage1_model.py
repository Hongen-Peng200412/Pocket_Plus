from __future__ import annotations

import torch
from torch import nn

from src.model.stage1_model import VolumePointStage1Model


class _DummyVoxelBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.return_feature_keys = ("voxel_c0",)
        self.feature_channels_by_name = {"voxel_c0": 2}

    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        return_feature_keys: tuple[str, ...] = (),
    ) -> dict[str, object]:
        batch_size = int(voxel_grid.shape[0])
        voxel_feat = voxel_grid.new_ones((batch_size, 2, 1, 1, 1))
        return {
            "voxel_features": {"voxel_c0": voxel_feat},
            "voxel_logits_aux": None,
            "voxel_recycle_out": None,
        }


class _DummyZerosPointBackbone(nn.Module):
    def __init__(self, out_channels: int, point_grid_size: float) -> None:
        super().__init__()
        self.backend = "zeros"
        self.out_channels = int(out_channels)
        self.point_grid_size = float(point_grid_size)
        self.feature_channels_by_name = {"point_feat": self.out_channels}
        self.forward_called = False
        self.build_zeros_output_called = False

    def forward(self, *args, **kwargs) -> dict[str, object]:
        self.forward_called = True
        raise AssertionError("backend='zeros' 时不应调用 point_backbone.forward()")

    def build_zeros_output(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor,
        atom_batch_index: torch.Tensor,
        atom_offsets: torch.Tensor,
        return_feature_names=None,
        point_input_feat=None,
    ) -> dict[str, object]:
        self.build_zeros_output_called = True
        point_feat = atom_feat.new_zeros((atom_feat.shape[0], self.out_channels))
        point_state = {
            "coord": atom_coord_centered_world,
            "batch": atom_batch_index.long(),
            "offset": atom_offsets.long(),
            "grid_size": self.point_grid_size,
        }
        point_feature_dict = {}
        if return_feature_names is not None and "point_feat" in tuple(return_feature_names):
            point_feature_dict["point_feat"] = point_feat
        return {
            "point_feat": point_feat,
            "point_state": point_state,
            "point_recycle_out": point_feat,
            "point_feature_dict": point_feature_dict,
        }


class _DummyAttentionStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_point_state: dict[str, object] | None = None
        self.last_token_feat: torch.Tensor | None = None

    def forward(self, point_state: dict[str, object], token_feat: torch.Tensor) -> torch.Tensor:
        self.last_point_state = point_state
        self.last_token_feat = token_feat
        return token_feat


def _make_model(**overrides) -> VolumePointStage1Model:
    voxel_backbone = _DummyVoxelBackbone()
    point_backbone = _DummyZerosPointBackbone(out_channels=4, point_grid_size=0.25)
    kwargs = dict(
        voxel_backbone=voxel_backbone,
        point_backbone=point_backbone,
        point_fusion_map={"point_feat": "voxel_c0"},
        point_fusion_modes=("concat_linear",),
        sampler_modes=("nearest",),
        fusion_mlp_ratio=1.0,
        fusion_proj_drop=0.0,
        atom_head_hidden_dim=8,
        atom_head_num_heads=1,
        atom_head_patch_size=4,
        atom_head_num_layers=1,
        atom_head_serialization_orders=("z",),
        atom_head_shuffle_orders=False,
        atom_head_qkv_bias=False,
        atom_head_qk_scale=None,
        atom_head_attn_drop=0.0,
        atom_head_proj_drop=0.0,
        atom_head_enable_rpe=False,
        atom_head_enable_flash=False,
        atom_head_upcast_attention=False,
        atom_head_upcast_softmax=False,
        atom_logit_dim=1,
        enable_recycling=False,
        max_recycles=1,
        randomize_recycles=False,
        detach_recycle_states=False,
        act_layer_name="gelu",
        ffn_type="mlp",
        atom_head_ffn_type="none",
        atom_head_mlp_ratio=4,
        atom_head_cpe_impl="none",
        atom_head_cpe_kernel_size=5,
        atom_head_cpe_receptive_field=2.0,
        atom_head_pointconv_max_neighbors=16,
        atom_head_drop_path=0.0,
        atom_head_pre_norm=True,
    )
    kwargs.update(overrides)
    return VolumePointStage1Model(**kwargs)


def test_volume_point_stage1_skips_point_forward_when_backend_is_zeros() -> None:
    point_backbone = _DummyZerosPointBackbone(out_channels=4, point_grid_size=0.25)
    model = _make_model(point_backbone=point_backbone)
    attention_stack = _DummyAttentionStack()
    model.atom_token_proj = nn.Identity()
    model.atom_attention_stack = attention_stack
    model.atom_logit_head = nn.Identity()

    batch = {
        "voxel_grid": torch.zeros((2, 1, 2, 2, 2), dtype=torch.float32),
        "atom_feat": torch.randn((3, 5), dtype=torch.float32),
        "atom_coord_centered_world": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [0.5, 0.5, -0.5]],
            dtype=torch.float32,
        ),
        "atom_batch_index": torch.tensor([0, 0, 1], dtype=torch.long),
        "atom_offsets": torch.tensor([2, 3], dtype=torch.long),
        "atom_valid_mask": torch.tensor([True, False, True]),
    }

    outputs = model(batch)

    expected_tokens = torch.cat(
        [
            torch.zeros((3, point_backbone.out_channels), dtype=torch.float32),
            batch["atom_coord_centered_world"],
            batch["atom_valid_mask"].float().unsqueeze(-1),
        ],
        dim=-1,
    )
    assert point_backbone.build_zeros_output_called is True
    assert point_backbone.forward_called is False
    assert outputs["sampled_point_fusion_feat_dict"] == {}
    assert torch.equal(outputs["fused_point_feat"], torch.zeros_like(outputs["fused_point_feat"]))
    assert torch.allclose(outputs["atom_tokens"], expected_tokens)
    assert attention_stack.last_point_state is not None
    assert attention_stack.last_point_state["grid_size"] == point_backbone.point_grid_size

def test_sample_voxel_feature_single_box_respects_centered_world_geometry() -> None:
    model = _make_model()
    voxel_feat = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 2, 2)
    sampled = model._sample_voxel_feature_single_box(
        voxel_feat_one_box=voxel_feat,
        point_coord_centered_world_one_box=torch.tensor(
            [
                [1.0, -2.0, 4.0],
                [-1.0, 2.0, -4.0],
            ],
            dtype=torch.float32,
        ),
        voxel_size_world_one_box=torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32),
        box_shape_zyx_one_box=torch.tensor([2, 2, 2], dtype=torch.long),
        fusion_mode="concat_linear",
        sampler_mode="nearest",
    )

    assert sampled.shape == (2, 1)
    assert torch.equal(sampled[:, 0], torch.tensor([5.0, 2.0]))


def test_update_split_info_after_trim_tracks_real_and_pseudo_counts_per_box() -> None:
    split_info = [(2, 1), (1, 2)]
    global_keep_mask = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)

    updated = VolumePointStage1Model._update_split_info_after_trim(
        split_info=split_info,
        global_keep_mask=global_keep_mask,
    )

    assert updated == [(1, 1), (0, 1)]
