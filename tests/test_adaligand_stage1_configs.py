from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
FIND_EXPERIMENTS = (
    "CPC1/Find_0",
    "CPC1/Find_1",
    "CPC2/Find_0",
    "CPC2/Find_1",
)


def _compose(experiment: str):
    """组合一份 AdaLigand Stage1 实验配置。"""

    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return hydra.compose(config_name="base", overrides=[f"+experiment={experiment}"])


@pytest.mark.parametrize("experiment", FIND_EXPERIMENTS)
def test_find_configs_encode_common_stage1_contract(experiment: str) -> None:
    """验证四份 Find 配置的公共结构、训练预算和 56D density 契约。"""

    cfg = _compose(experiment)
    model_name = Path(experiment).name
    assert cfg.dataset.stage1_model_name == model_name
    assert len(cfg.dataset.density_channel_config.enabled_channels) == 56
    assert cfg.dataset.atom_buffer_radius == 8.0
    assert cfg.model.backbone.max_recycles == 3
    assert cfg.model.backbone.randomize_recycles is True
    assert cfg.model.backbone.voxel_backbone.num_conv3d_aux == 0
    assert cfg.model.backbone.voxel_backbone.num_conv3d_ligand == 0
    assert cfg.model.backbone.embed_head.num_trunk_blocks == 0
    assert cfg.model.backbone.embed_head.num_voxel_blocks == 0
    assert cfg.model.backbone.embed_head.num_point_blocks == 3
    assert list(cfg.model.backbone.embed_head.point_buffer_radii) == [8.0, 4.0, 0.0]
    assert cfg.train.global_batch_size == 64
    assert cfg.train.max_epochs == 20
    assert cfg.train.scheduler.threshold == 0.001
    assert cfg.model.monitor_metric == "val_loss/global/total"
    assert cfg.model.monitor_mode == "min"
    assert cfg.model.voxel_aux_loss_weight in {0.0, 0.1}
    assert cfg.model.pseudo_loss_weight == 0.1
    assert cfg.model.ligand_sparse_refine_loss is None


def test_find0_and_find1_only_change_voxel_receptor_construction() -> None:
    """验证两个 Find 共享 point 配置，仅 voxel receptor grid recipe 不同。"""

    find0 = _compose("CPC1/Find_0")
    find1 = _compose("CPC1/Find_1")
    assert find0.model.backbone.online_pdb_feature is True
    assert find0.model.backbone.embed_head.embed_voxel_out_channels == 0
    assert find0.model.backbone.online_pdb_feature_dim == 49
    assert find1.model.backbone.online_pdb_feature is False
    assert find1.model.backbone.embed_head.embed_voxel_out_channels == 49
    assert find1.model.backbone.embed_head.use_centroid_encoding is True
    assert find1.model.backbone.embed_head.use_soft_splatting is True
    assert find1.model.backbone.embed_head.add_occupancy_channels is True
    assert find1.model.backbone.embed_head.embed_residual_enabled is True


def test_cpc_stage_boundary_and_scheduler_contract() -> None:
    """验证 CPC1 从头训练、CPC2 strict model-only 接续及冻结/loss 边界。"""

    cpc1 = _compose("CPC1/Find_0")
    cpc2 = _compose("CPC2/Find_0")
    assert cpc1.init_from is None
    assert cpc1.train.scheduler.warmup_ratio == 0.025
    assert cpc1.train.scheduler.patience == 3
    assert cpc1.train.scheduler.stop_after_lr_reductions == 3
    assert cpc1.model.voxel_aux_loss_weight == 0.1
    assert cpc1.model.voxel_ligand_loss_weight == 1.0

    assert cpc2.init_from == "***"
    assert cpc2.train.scheduler.warmup_steps == 0
    assert cpc2.train.scheduler.warmup_ratio == 0.0
    assert cpc2.train.scheduler.patience == 1
    assert cpc2.train.scheduler.stop_after_lr_reductions == 1
    assert cpc2.model.voxel_aux_loss_weight == 0.0
    assert cpc2.model.voxel_ligand_loss_weight == 0.0
    assert any("voxel_backbone" in pattern for pattern in cpc2.frozen_module.patterns)


def test_unet_c1_config_is_density_only_and_exports_five_v_levels() -> None:
    """验证 unet_c1 无 Point/A/P，并显式请求 centered 的五路 V。"""

    cfg = _compose("unet_c1")
    assert cfg.dataset.stage1_model_name == "unet_c1"
    assert list(cfg.dataset.density_channel_config.enabled_channels) == ["exp_clipnorm_nopost"]
    assert cfg.model.backbone.point_backbone is None
    assert cfg.model.backbone.embed_head is None
    assert cfg.model.backbone.enable_atom_head is False
    assert list(cfg.model.backbone.voxel_backbone.return_feature_keys) == [
        "voxel_ds_2",
        "voxel_ds_3",
        "voxel_ds_4",
        "voxel_c4",
        "voxel_final",
    ]
    assert cfg.model.voxel_ligand_loss_weight == 1.0
    assert cfg.model.voxel_aux_loss_weight == 0.1
    assert cfg.model.atom_loss is None
    assert cfg.model.ligand_pseudo_loss is None
