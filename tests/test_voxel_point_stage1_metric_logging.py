from __future__ import annotations

import pytest
import torch
from torch import nn

from src.wrappers.voxel_point_stage1_logging import build_metric_key, log_scalar_payload
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


def test_cpu_metric_uses_gloo_group_inside_nccl_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL 多进程训练中的 CPU PRAUC 状态改用 Gloo 通信组。"""

    manager = ValidationMetricManager(
        branches=[
            MetricBranchSpec(
                name="receptor",
                enabled=True,
                num_classes=2,
                class_names=("background", "foreground"),
                thresholds=None,
            ),
            MetricBranchSpec(
                name="voxel_ligand",
                enabled=True,
                num_classes=2,
                class_names=("background", "foreground"),
                thresholds=8,
            ),
        ],
        metric_device_policy="auto",
    )
    gloo_group = object()
    created_backends: list[str] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")

    def _new_group(*, backend: str) -> object:
        created_backends.append(backend)
        return gloo_group

    monkeypatch.setattr(torch.distributed, "new_group", _new_group)

    manager._configure_distributed_process_groups()
    manager._configure_distributed_process_groups()

    assert created_backends == ["gloo"]
    assert manager.metrics["receptor__binary"].process_group is gloo_group
    assert manager.metrics["voxel_ligand__binary"].process_group is None


def test_globally_reduced_metric_payload_is_logged_without_lightning_resync() -> None:
    """已经完成跨卡聚合的指标不再由 Lightning 重复同步。"""

    class _LoggingModule:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def log(self, key: str, value: torch.Tensor, **kwargs: object) -> None:
            self.calls.append({"key": key, "value": value, **kwargs})

    module = _LoggingModule()
    log_scalar_payload(
        module=module,
        payload={"val_score/global/atom_PRAUC": torch.tensor(0.75)},
        monitor_metric="val_score/global/atom_PRAUC",
        sync_dist=False,
    )

    assert len(module.calls) == 1
    assert module.calls[0]["sync_dist"] is False
    assert module.calls[0]["on_epoch"] is True


def test_validation_macro_skips_classes_without_positive_targets() -> None:
    """结构类别宏平均只纳入整个验证集中实际出现的前景类别。"""

    manager = ValidationMetricManager(
        branches=[
            MetricBranchSpec(
                name="protein_mainchain",
                enabled=True,
                num_classes=4,
                class_names=("background", "A", "B", "C"),
                thresholds=8,
                report_per_class=False,
                macro_present_classes_only=True,
            )
        ],
        metric_device_policy="auto",
    )
    logits = torch.tensor(
        [
            [
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[4.0, 1.0, -2.0, -3.0]]],
                [[[-2.0, -2.0, -2.0, -2.0]]],
                [[[-3.0, -1.0, 1.0, 4.0]]],
            ]
        ]
    )
    target = torch.tensor([[[[1, 0, 0, 3]]]])
    mask = torch.ones_like(target, dtype=torch.bool)

    manager.update_branch(
        branch_name="protein_mainchain",
        logits=logits,
        target=target,
        mask=mask,
    )
    payload = manager.compute_payload()

    expected = torch.stack(
        [
            manager.metrics["protein_mainchain__class_1"].compute(),
            manager.metrics["protein_mainchain__class_3"].compute(),
        ]
    ).mean()
    torch.testing.assert_close(
        payload["val_score/global/protein_mainchain_PRAUC_macro"],
        expected,
    )
    assert len(payload) == 1


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
