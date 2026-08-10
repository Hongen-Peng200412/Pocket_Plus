from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
FIND_EXPERIMENTS = (
    "CPC1/Find_0",
    "CPC1/Find_1",
    "CPC1/Find_2",
    "CPC2/Find_0",
    "CPC2/Find_1",
    "CPC2/Find_2",
)
FORMAL_STAGE1_PREPARATION_ROOT = (
    "/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation"
)
LIGAND_PRAUC_MONITORED_EXPERIMENTS = (
    "CPC1/Find_0",
    "CPC2/Find_0",
    "CPC1/Find_1",
    "CPC2/Find_1",
    "unet_c1",
)


def _compose(experiment: str):
    """组合一份 AdaLigand Stage1 实验配置. """

    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return hydra.compose(config_name="base", overrides=[f"+experiment={experiment}"])


@pytest.mark.parametrize(
    ("experiment", "launcher_name"),
    (("CPC1/Find_0", "Find_0.sh"), ("CPC1/Find_1", "Find_1.sh"), ("unet_c1", "unet_c1.sh")),
)
def test_stage1_training_defaults_to_formal_preparation(
    experiment: str,
    launcher_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset 配置和正式启动脚本默认读取同一份 Stage1 preparation. """

    monkeypatch.delenv("ADALIGAND_STAGE1_PREPARATION_ROOT", raising=False)
    cfg = _compose(experiment)
    assert cfg.dataset.box_pool_root == f"{FORMAL_STAGE1_PREPARATION_ROOT}/box_pool"

    launcher_text = (
        PROJECT_ROOT / "训练与运行" / "sh" / launcher_name
    ).read_text(encoding="utf-8")
    expected_default = (
        "${ADALIGAND_STAGE1_PREPARATION_ROOT:-"
        f"{FORMAL_STAGE1_PREPARATION_ROOT}"
        "}"
    )
    assert expected_default in launcher_text


@pytest.mark.parametrize("experiment", FIND_EXPERIMENTS)
def test_find_configs_encode_common_stage1_contract(experiment: str) -> None:
    """验证六份 Find 配置的公共结构、训练预算和 56D density 契约. """

    cfg = _compose(experiment)
    model_name = Path(experiment).name
    assert cfg.dataset.stage1_model_name == model_name
    assert len(cfg.dataset.density_channel_config.enabled_channels) == 56
    assert cfg.dataset.atom_buffer_radius == 8.0
    assert cfg.model.backbone.max_recycles == 3
    assert cfg.model.backbone.randomize_recycles is True
    assert cfg.model.backbone.embed_head.num_trunk_blocks == 0
    assert cfg.model.backbone.embed_head.num_voxel_blocks == 0
    assert cfg.model.backbone.embed_head.num_point_blocks == 3
    assert list(cfg.model.backbone.embed_head.point_buffer_radii) == [8.0, 4.0, 0.0]
    assert cfg.model.backbone.density_cube_cfg.cube_size == 9
    assert cfg.model.backbone.real_density_cube_cfg.cube_size == 9
    assert cfg.model.backbone.density_cube_cfg.num_conv == 0
    assert cfg.model.backbone.real_density_cube_cfg.num_conv == 0
    assert cfg.model.backbone.density_cube_cfg.chunk_size == 2048
    assert cfg.model.backbone.real_density_cube_cfg.chunk_size == 4096
    assert cfg.train.global_batch_size == 64
    assert cfg.train.max_epochs == 20
    assert cfg.train.val_per_epoch == 3
    assert cfg.train.num_workers == 32
    assert cfg.train.prefetch_factor == 4
    assert cfg.train.persistent_workers is False
    assert cfg.dataset.occurrence_cap_per_pdb == 50
    assert cfg.dataset.occurrence_ratio == pytest.approx(0.75)
    assert cfg.dataset.cache_max_bytes == 32 * 1024**3
    assert dict(cfg.dataset.entry_ratio) == {
        "center": 0,
        "bias": 1,
        "context": 1,
    }
    assert cfg.train.optimizer.lr == pytest.approx(5.0e-5)
    assert cfg.train.scheduler.threshold == 0.001
    assert cfg.model.voxel_aux_loss_weight in {0.0, 0.1}
    assert cfg.model.pseudo_loss_weight == 0.1
    assert cfg.model.ligand_sparse_refine_loss is None
    if model_name == "Find_1":
        assert cfg.model.backbone.voxel_backbone.feature_channels == 64
        assert cfg.model.backbone.voxel_backbone.enable_multiscale_output is False
        assert cfg.model.backbone.voxel_backbone.enable_structure_heads is True


@pytest.mark.parametrize("experiment", LIGAND_PRAUC_MONITORED_EXPERIMENTS)
def test_current_stage1_models_select_checkpoints_by_ligand_prauc(experiment: str) -> None:
    """Find_0、Find_1 与 unet_c1 都按配体区域 PRAUC 选择 checkpoint 并调节学习率。"""

    cfg = _compose(experiment)
    assert cfg.model.monitor_metric == "val_score/global/voxel_ligand_PRAUC"
    assert cfg.model.monitor_mode == "max"


def test_find_models_only_change_voxel_receptor_construction() -> None:
    """验证三个 Find 共享 point 配置, 仅 voxel receptor grid recipe 不同. """

    find0 = _compose("CPC1/Find_0")
    find1 = _compose("CPC1/Find_1")
    find2 = _compose("CPC1/Find_2")
    assert find0.model.backbone.online_pdb_feature is True
    assert find0.model.backbone.voxel_backbone.enable_structure_heads is False
    assert find0.model.backbone.embed_head.embed_voxel_out_channels == 0
    assert find0.model.backbone.online_pdb_feature_dim == 49
    assert find1.model.backbone.online_pdb_feature is False
    assert find1.model.backbone.embed_head.embed_voxel_out_channels == 49
    assert find1.model.backbone.embed_head.use_centroid_encoding is True
    assert find1.model.backbone.embed_head.use_gaussian_splatting is True
    assert find1.model.backbone.embed_head.use_soft_splatting is True
    assert find1.model.backbone.embed_head.add_occupancy_channels is True
    assert find1.model.backbone.embed_head.embed_residual_enabled is True
    assert find1.model.backbone.embed_head.voxel_embed_as_tune is False
    assert find2.model.backbone.online_pdb_feature is False
    assert find2.model.backbone.embed_head.embed_voxel_out_channels == 49
    assert find2.model.backbone.embed_head.embed_point_out_channels == 64
    assert find2.model.backbone.embed_head.use_gaussian_splatting is True
    assert find2.model.backbone.embed_head.voxel_embed_as_tune is True


@pytest.mark.parametrize("experiment", (*FIND_EXPERIMENTS, "unet_c1"))
def test_current_experiments_only_use_stage1_dataset_contract(experiment: str) -> None:
    """验证当前实验只实例化 Stage1Dataset, 不再暴露旧 BOX 目录平衡采样入口. """

    cfg = _compose(experiment)
    assert cfg.dataset._target_ == "src.datasets.stage1_dataset.Stage1Dataset"
    assert "use_balanced_foreground_sampler" not in cfg.train
    assert "balanced_foreground_ratio" not in cfg.train


def test_cpc_stage_boundary_and_scheduler_contract() -> None:
    """验证 CPC1 从头训练、CPC2 strict model-only 接续及冻结/loss 边界. """

    cpc1 = _compose("CPC1/Find_1")
    cpc2 = _compose("CPC2/Find_1")
    assert cpc1.init_from is None
    assert cpc1.train.scheduler.warmup_ratio == 0.005
    assert cpc1.train.scheduler.patience == 2
    assert cpc1.train.scheduler.stop_after_lr_reductions == 3
    assert cpc1.model.voxel_aux_loss_weight == 0.1
    assert cpc1.model.voxel_ligand_loss_weight == 1.0
    assert cpc1.model.protein_mainchain_loss_weight == 0.05
    assert cpc1.model.nucleic_mainchain_loss_weight == 0.05
    assert cpc1.model.ligand_distance_loss_weight == 0.3

    assert cpc2.init_from == "***"
    assert cpc2.train.scheduler.warmup_steps == 0
    assert cpc2.train.scheduler.warmup_ratio == 0.0
    assert cpc2.train.scheduler.patience == 1
    assert cpc2.train.scheduler.stop_after_lr_reductions == 1
    assert cpc2.model.voxel_aux_loss_weight == 0.0
    assert cpc2.model.voxel_ligand_loss_weight == 0.0
    assert cpc2.model.protein_mainchain_loss_weight == 0.0
    assert cpc2.model.nucleic_mainchain_loss_weight == 0.0
    assert cpc2.model.ligand_distance_loss_weight == 0.0
    assert any("voxel_backbone" in pattern for pattern in cpc2.frozen_module.patterns)


def test_find2_cpc2_keeps_an_independent_checkpoint_lineage() -> None:
    """验证 Find_2 CPC2 只从同名 CPC1 BEST 严格 model-only 初始化. """

    cpc1 = _compose("CPC1/Find_2")
    cpc2 = _compose("CPC2/Find_2")
    assert cpc1.name == "Find_2_CPC1"
    assert cpc1.init_from is None
    assert cpc2.name == "Find_2_CPC2"
    assert cpc2.init_from == "***"
    assert cpc2.dataset.stage1_model_name == "Find_2"


def test_unet_c1_config_is_density_only_and_exports_final_v_feature() -> None:
    """验证 unet_c1 保留 U-Net 骨干，并只向归档导出最终体素特征。"""

    cfg = _compose("unet_c1")
    find1 = _compose("CPC1/Find_1")
    assert cfg.dataset.stage1_model_name == "unet_c1"
    assert list(cfg.dataset.density_channel_config.enabled_channels) == ["exp_clipnorm_nopost"]
    assert cfg.model.backbone.point_backbone is None
    assert cfg.model.backbone.embed_head is None
    assert cfg.model.backbone.enable_atom_head is False
    assert cfg.model.backbone.prior_prob_init_enabled is True
    assert cfg.model.backbone.voxel_backbone.feature_channels == 64
    assert cfg.model.backbone.voxel_backbone.enable_multiscale_output is False
    assert cfg.model.backbone.voxel_backbone.enable_structure_heads is True
    assert list(cfg.model.backbone.voxel_backbone.return_feature_keys) == ["voxel_final"]
    assert cfg.model.voxel_ligand_loss_weight == 1.0
    assert cfg.model.voxel_aux_loss_weight == 0.1
    assert cfg.model.ligand_distance_loss_weight == 0.3
    assert cfg.model.protein_mainchain_loss_weight == 0.05
    assert cfg.model.nucleic_mainchain_loss_weight == 0.05
    assert cfg.model.atom_loss is None
    assert cfg.model.ligand_pseudo_loss is None
    assert cfg.train.optimizer.lr == pytest.approx(1.0e-4)
    assert cfg.dataset.occurrence_cap_per_pdb == 50
    assert cfg.dataset.occurrence_ratio == pytest.approx(0.75)
    assert cfg.dataset.cache_max_bytes == 32 * 1024**3
    assert dict(cfg.dataset.entry_ratio) == {
        "center": 0,
        "bias": 1,
        "context": 1,
    }
    assert "excluded_pdb_ids" not in cfg.dataset
    assert "excluded_pdb_ids" not in find1.dataset
    assert cfg.train.val_per_epoch == 3
    assert cfg.train.num_workers == 32
    assert cfg.train.prefetch_factor == 4
    assert cfg.train.persistent_workers is False
