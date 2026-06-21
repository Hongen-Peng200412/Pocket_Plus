from __future__ import annotations

import pytest
import torch
from torch import nn

from src.wrappers.voxel_point_stage1_logging import build_metric_key
from src.wrappers.voxel_point_stage1_metrics import MetricBranchSpec, ValidationMetricManager


def test_binary_metric_key_omits_task_class_suffix() -> None:
    """
    验证二分类指标 key 不追加 task class suffix。
    """
    key = build_metric_key(
        panel="val_score",
        metric="refined_F1",
        num_classes=2,
        scope="global",
        task_class_name="foreground",
    )

    assert key == "val_score/global/refined_F1"


def test_multiclass_metric_key_appends_task_class_suffix() -> None:
    """
    验证多分类指标 key 在 leaf 上追加 task class suffix。
    """
    key = build_metric_key(
        panel="val_refined",
        metric="F1",
        num_classes=3,
        scope="global",
        task_class_name="small_molecule",
    )

    assert key == "val_refined/global/F1_small_molecule"


def test_metric_key_rejects_non_global_scope() -> None:
    """
    验证 metric key builder 只接受 global 作用域。
    """
    with pytest.raises(ValueError, match="scope"):
        build_metric_key(
            panel="val_score",
            metric="refined_F1",
            num_classes=2,
            scope="per_dataset",
            task_class_name=None,
        )


def test_validation_metric_manager_is_module_and_omits_binary_macro() -> None:
    """
    验证常规 validation metric manager 是 nn.Module, 且二分类不输出 macro。
    """
    manager = ValidationMetricManager(
        branches=[
            MetricBranchSpec(
                name="voxel_ligand",
                enabled=True,
                num_classes=2,
                class_names=("background", "foreground"),
                thresholds=4,
            )
        ],
        metric_device_policy="auto",
    )
    logits = torch.tensor([[[[[0.0, 2.0, -2.0]]]]])
    target = torch.tensor([[[0, 1, 0]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)

    manager.update_branch(branch_name="voxel_ligand", logits=logits, target=target, mask=mask)
    payload = manager.compute_payload()

    assert isinstance(manager, nn.Module)
    assert "val_score/global/voxel_ligand_PRAUC" in payload
    assert not any(key.endswith("macro") for key in payload)
    assert manager.state_dict() == {}


def test_validation_metric_manager_outputs_multiclass_suffix_and_macro() -> None:
    """
    验证多分类常规 validation metric 输出前景类 suffix 与 macro。
    """
    manager = ValidationMetricManager(
        branches=[
            MetricBranchSpec(
                name="voxel_ligand",
                enabled=True,
                num_classes=3,
                class_names=("background", "metal_ion", "small_molecule"),
                thresholds=4,
            )
        ],
        metric_device_policy="auto",
    )
    logits = torch.tensor([[[[[0.0, 4.0, -4.0], [0.0, -4.0, 4.0]]]]]).permute(0, 4, 1, 2, 3)
    target = torch.tensor([[[1, 2]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)

    manager.update_branch(branch_name="voxel_ligand", logits=logits, target=target, mask=mask)
    payload = manager.compute_payload()

    assert "val_score/global/voxel_ligand_PRAUC_metal_ion" in payload
    assert "val_score/global/voxel_ligand_PRAUC_small_molecule" in payload


def test_wrapper_receptor_metric_name_replaces_voxel_aux_name() -> None:
    """
    验证新 wrapper 对外使用 receptor PR-AUC 名称, 不再注册 voxel_aux PR-AUC 名称。
    """
    from types import SimpleNamespace

    import torch
    from torch import nn

    from src.modules.losses import AdaptiveClassificationCompositeLoss
    from src.wrappers.voxel_point_stage1 import VoxelPointStage1Wrapper

    class _Backbone(nn.Module):
        def forward(self, batch: dict) -> dict:
            return batch

    wrapper = VoxelPointStage1Wrapper(
        backbone=_Backbone(),
        voxel_aux_loss=AdaptiveClassificationCompositeLoss(
            num_classes=2,
            hard_label_threshold=1.7,
            focal_gamma=2.0,
            focal_alpha=[0.5, 0.5],
            focal_eps=1.0e-6,
            tversky_alpha=0.5,
            tversky_beta=0.5,
            tversky_smooth=1.0,
            w_focal=0.7,
            w_tversky=0.3,
            w_mse=0.0,
        ),
        class_names=["background", "foreground"],
    )

    assert "receptor" in wrapper.val_metrics.branch_specs
    assert "receptor__binary" in wrapper.val_metrics.metrics
    assert "voxel_aux" not in wrapper.val_metrics.branch_specs
