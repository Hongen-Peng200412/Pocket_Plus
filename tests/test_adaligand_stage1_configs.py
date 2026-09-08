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
    "/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3"
)
LIGAND_PRAUC_MONITORED_EXPERIMENTS = (
    "CPC1/Find_0",
    "CPC2/Find_0",
    "CPC1/Find_1",
    "CPC1/Find_1_pdb_centric_2",
    "CPC2/Find_1",
    "unet_c1",
)


def _compose(experiment: str):
    """组合一份 AdaLigand Stage1 实验配置."""

    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return hydra.compose(config_name="base", overrides=[f"+experiment={experiment}"])


@pytest.mark.parametrize(
    ("experiment", "launcher_name"),
    (
        ("CPC1/Find_0", "Find_0.sh"),
        ("CPC1/Find_1", "Find_1.sh"),
        ("unet_base", "unet_base.sh"),
        ("unet_c1", "unet_c1.sh"),
        ("unet_diff", "unet_diff.sh"),
    ),
)
def test_stage1_training_defaults_to_formal_preparation(
    experiment: str,
    launcher_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset 配置和正式启动脚本默认读取同一份 Stage1 preparation."""

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


def test_training_launchers_use_v3_worker_and_scope_contracts() -> None:
    """一键入口保持正式 worker 数量, 并使用新的训练与验证预算."""

    shell_root = PROJECT_ROOT / "训练与运行" / "sh"
    expected_training_scope = {
        "Find_0.sh": (16, 70, 10),
        "unet_base.sh": (16, 70, 12),
        "unet_c1.sh": (30, 110, 8),
        "unet_diff.sh": (16, 70, 12),
    }
    for launcher_name, (
        num_workers,
        max_epochs,
        val_per_epoch,
    ) in expected_training_scope.items():
        launcher_text = (shell_root / launcher_name).read_text(encoding="utf-8")
        assert f'"train.num_workers={num_workers}"' in launcher_text
        assert '"train.prefetch_factor=4"' in launcher_text
        assert f'"train.max_epochs={max_epochs}"' in launcher_text
        assert f'"train.val_per_epoch={val_per_epoch}"' in launcher_text
        assert '"train.scheduler.warmup_ratio=0.005"' in launcher_text
        assert "训练与运行/runtime/launch_training_python.sh" in launcher_text
    for launcher_name in ("Find_0.sh", "Find_1.sh", "Find_1_pdb_centric_2.sh"):
        launcher_text = (shell_root / launcher_name).read_text(encoding="utf-8")
        assert "+experiment=CPC1/" in launcher_text
        assert "+experiment=CPC2/" not in launcher_text
        assert "训练与运行/runtime/launch_training_python.sh" in launcher_text

    find1_text = (shell_root / "Find_1.sh").read_text(encoding="utf-8")
    assert 'devices="${TASK_GPUS:-2}"' in find1_text
    assert 'nnodes="${TASK_NNODES:-1}"' in find1_text
    submit_text = (PROJECT_ROOT / "训练与运行" / "submit_task.sh").read_text(
        encoding="utf-8"
    )
    assert "--sh Find_1.sh --resource h100 --gpus 2 --cpus 64" in submit_text
    assert "--sh unet_c1.sh --resource h100 --gpus 1 --cpus 32" in submit_text

    unet_text = (shell_root / "unet_c1.sh").read_text(encoding="utf-8")
    no_mainchain_text = (shell_root / "unet_c1_no_mainchain.sh").read_text(
        encoding="utf-8"
    )
    assert 'protein_mainchain_weight="0.05"' in unet_text
    assert 'nucleic_mainchain_weight="0.05"' in unet_text
    assert "UNET_C1_VARIANT=no_mainchain" in no_mainchain_text
    assert 'TASK_GPUS="${TASK_GPUS:-2}"' in no_mainchain_text


@pytest.mark.parametrize("experiment", FIND_EXPERIMENTS)
def test_find_configs_encode_common_stage1_contract(experiment: str) -> None:
    """验证六份 Find 配置的公共结构, 训练预算和 56D density 契约."""

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
    if experiment == "CPC1/Find_1":
        assert cfg.train.global_batch_size == 48
        assert cfg.train.batch_size == 6
        assert cfg.train.gradient_clip_mode == "find_voxel_point"
    else:
        assert cfg.train.global_batch_size == 64
    assert cfg.train.max_epochs == 70
    assert cfg.train.val_per_epoch == 12
    assert cfg.train.optimizer.lr == pytest.approx(5.0e-5)
    assert cfg.train.scheduler.threshold == 0.003
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
    """验证三个 Find 共享 point 配置, 仅 voxel receptor grid recipe 不同."""

    find0 = _compose("CPC1/Find_0")
    find1 = _compose("CPC1/Find_1")
    find2 = _compose("CPC1/Find_2")
    assert find0.model.backbone.online_pdb_feature is True
    assert find0.model.backbone.voxel_backbone.enable_structure_heads is False
    assert find0.model.backbone.embed_head.embed_voxel_out_channels == 0
    assert find0.model.backbone.online_pdb_feature_dim == 50
    assert find1.model.backbone.online_pdb_feature is False
    assert find1.model.backbone.embed_head.embed_voxel_out_channels == 50
    assert find1.model.backbone.embed_head.use_centroid_encoding is True
    assert find1.model.backbone.embed_head.use_gaussian_splatting is True
    assert find1.model.backbone.embed_head.use_soft_splatting is True
    assert find1.model.backbone.embed_head.add_occupancy_channels is True
    assert find1.model.backbone.embed_head.embed_residual_enabled is True
    assert find1.model.backbone.embed_head.voxel_embed_as_tune is False
    assert find2.model.backbone.online_pdb_feature is False
    assert find2.model.backbone.embed_head.embed_voxel_out_channels == 50
    assert find2.model.backbone.embed_head.embed_point_out_channels == 64
    assert find2.model.backbone.embed_head.use_gaussian_splatting is True
    assert find2.model.backbone.embed_head.voxel_embed_as_tune is True


@pytest.mark.parametrize(
    "experiment",
    (*FIND_EXPERIMENTS, "unet_base", "unet_diff"),
)
def test_method_two_experiments_keep_the_v1_dataset_contract(experiment: str) -> None:
    """验证方式二实验继续使用 V1 冻结文件与 25/0.5/25 参数."""

    cfg = _compose(experiment)
    assert cfg.dataset._target_ == "src.datasets.stage1_dataset.Stage1Dataset"
    assert cfg.dataset.split_val.endswith("/validation_selection_pdb_centric.npz")
    assert cfg.dataset.pdb_foreground_box_num == 25
    assert cfg.dataset.pdb_foreground_fraction_target == pytest.approx(0.5)
    assert cfg.dataset.pdb_occurrence_foreground_box_cap == 25
    assert "use_balanced_foreground_sampler" not in cfg.train
    assert "balanced_foreground_ratio" not in cfg.train


def test_find1_pdb_centric_configs_are_explicit_and_independent() -> None:
    """两份 Find_1 配置固定各自的 Dataset、validation 与训练预算."""

    first = _compose("CPC1/Find_1")
    second = _compose("CPC1/Find_1_pdb_centric_2")

    assert first.dataset.name == "stage1_find_pdb_centric_1"
    assert first.dataset.split_val.endswith("/validation_selection_pdb_centric.npz")
    assert first.dataset.pdb_foreground_box_num == 25
    assert first.dataset.pdb_foreground_fraction_target == pytest.approx(0.5)
    assert first.dataset.pdb_occurrence_foreground_box_cap == 25
    assert first.train.global_batch_size == 48
    assert first.train.batch_size == 6
    assert first.train.num_workers == 24
    assert first.train.max_epochs == 70
    assert first.train.val_per_epoch == 12

    assert second.dataset.name == "stage1_find_pdb_centric_2"
    assert second.dataset.split_val.endswith(
        "/validation_selection_pdb_centric_v2.npz"
    )
    assert second.dataset.pdb_foreground_box_num == 50
    assert second.dataset.pdb_foreground_fraction_target == pytest.approx(25 / 33)
    assert second.dataset.pdb_occurrence_foreground_box_cap == 1
    assert second.train.global_batch_size == 48
    assert second.train.batch_size == 6
    assert second.train.num_workers == 30
    assert second.train.max_epochs == 110
    assert second.train.val_per_epoch == 8

    for cfg in (first, second):
        assert cfg.train.gradient_clip_val == pytest.approx(0.5)
        assert cfg.train.gradient_clip_mode == "find_voxel_point"
        assert cfg.train.optimizer.lr == pytest.approx(5.0e-5)
        assert cfg.train.scheduler.threshold == pytest.approx(0.003)

    second_shell = (
        PROJECT_ROOT / "训练与运行" / "sh" / "Find_1_pdb_centric_2.sh"
    ).read_text(encoding="utf-8")
    assert "+experiment=CPC1/Find_1_pdb_centric_2" in second_shell
    assert "if [[" not in second_shell


def test_cpc_stage_boundary_and_scheduler_contract() -> None:
    """验证 CPC1 从头训练, CPC2 strict model-only 接续及冻结/loss 边界."""

    cpc1 = _compose("CPC1/Find_1")
    cpc2 = _compose("CPC2/Find_1")
    assert cpc1.init_from is None
    assert cpc1.train.scheduler.warmup_ratio == 0.005
    assert cpc1.train.scheduler.patience == 3
    assert cpc1.train.scheduler.stop_after_lr_reductions == 2
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
    """验证 Find_2 CPC2 只从同名 CPC1 BEST 严格 model-only 初始化."""

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
    assert cfg.dataset._target_ == "src.datasets.stage1_dataset.Stage1Dataset"
    assert cfg.dataset.split_val.endswith(
        "/validation_selection_pdb_centric_v2.npz"
    )
    assert cfg.dataset.pdb_foreground_box_num == 50
    assert cfg.dataset.pdb_foreground_fraction_target == pytest.approx(25 / 33)
    assert cfg.dataset.pdb_occurrence_foreground_box_cap == 1
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
    assert "box_sample_fraction" not in cfg.dataset
    assert cfg.train.num_workers == 16
    assert cfg.train.prefetch_factor == 4
    assert "excluded_pdb_ids" not in cfg.dataset
    assert "excluded_pdb_ids" not in find1.dataset
    assert cfg.train.max_epochs == 70
    assert cfg.train.val_per_epoch == 12


@pytest.mark.parametrize(
    ("experiment", "channels"),
    (
        ("unet_base", None),
        ("unet_diff", ["exp_clipnorm_nopost", "diff_clipnorm_nopost"]),
    ),
)
def test_unet_density_ablation_configs_keep_common_training_and_auxiliary_losses(
    experiment: str,
    channels: list[str] | None,
) -> None:
    """验证两项 U-Net 消融只改变 density 通道，并共同启用正式训练与辅助损失。"""

    cfg = _compose(experiment)
    resolved_channels = list(cfg.dataset.density_channel_config.enabled_channels)
    if channels is None:
        assert len(resolved_channels) == 56
    else:
        assert resolved_channels == channels
    assert cfg.dataset.stage1_model_name == experiment
    assert cfg.model.backbone.point_backbone is None
    assert cfg.model.backbone.embed_head is None
    assert cfg.model.backbone.enable_atom_head is False
    assert cfg.model.backbone.voxel_backbone.enable_structure_heads is True
    assert cfg.model.voxel_ligand_loss_weight == pytest.approx(1.0)
    assert cfg.model.voxel_aux_loss_weight == pytest.approx(0.1)
    assert cfg.model.ligand_distance_loss_weight == pytest.approx(0.3)
    assert cfg.model.protein_mainchain_loss_weight == pytest.approx(0.05)
    assert cfg.model.nucleic_mainchain_loss_weight == pytest.approx(0.05)
    assert cfg.train.optimizer.lr == pytest.approx(1.0e-4)
    assert cfg.train.num_workers == 16
    assert cfg.train.max_epochs == 70
    assert cfg.train.val_per_epoch == 12
