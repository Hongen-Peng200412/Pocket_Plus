from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _minimal_box_sample() -> dict[str, Any]:
    """
    构造不含旧 mask 字段的最小推理 BOX 样本。

    输出:
        - sample: dict[str, Any], 与当前 box_sample_builder / box_point_collate 契约一致的单 BOX 样本
    """
    return {
        "voxel_grid": torch.zeros((1, 2, 2, 2), dtype=torch.float32),
        "voxel_label": torch.zeros((2, 2, 2), dtype=torch.long),
        "hardmask": torch.zeros((2, 2, 2), dtype=torch.long),
        "box_origin_world": torch.zeros((3,), dtype=torch.float32),
        "voxel_size_world": torch.ones((3,), dtype=torch.float32),
        "box_shape_zyx": torch.tensor([2, 2, 2], dtype=torch.long),
        "atom_coord_world": torch.zeros((1, 3), dtype=torch.float32),
        "atom_coord_local_voxel": torch.zeros((1, 3), dtype=torch.float32),
        "atom_coord_centered_world": torch.zeros((1, 3), dtype=torch.float32),
        "atom_feat": torch.zeros((1, 4), dtype=torch.float32),
        "atom_label": torch.zeros((1,), dtype=torch.long),
        "atom_is_in_core_box": torch.ones((1,), dtype=torch.bool),
        "atom_global_indices": torch.zeros((1,), dtype=torch.long),
        "sample_name": "infer_box_0",
        "pdb_id": "infer",
        "class_name": "infer",
        "instance_id": 0,
        "is_center_box": False,
        "box_position_zyx": (0, 0, 0),
    }


def test_prepare_batched_boxes_accepts_samples_without_legacy_masks() -> None:
    """
    验证推理 batch 构造不再要求 voxel_valid_mask / atom_valid_mask。
    """
    from src.inference.parse_input import prepare_batched_boxes

    box_dicts = [_minimal_box_sample()]
    batch = next(prepare_batched_boxes(box_dicts=box_dicts, batch_size=1, device="cpu"))

    assert "voxel_valid_mask" not in batch
    assert "atom_valid_mask" not in batch
    assert tuple(batch["voxel_grid"].shape) == (1, 1, 2, 2, 2)
    assert tuple(batch["atom_is_in_core_box"].shape) == (1,)
    assert batch["_box_meta"][0]["box_position_zyx"] == (0, 0, 0)


def test_voxel_pipeline_does_not_require_valid_crop_margin(monkeypatch, tmp_path) -> None:
    """
    验证首个 voxel 推理样本的前向组装路径不再读取 valid_crop_margin。
    """
    from src.inference.main import voxel_pipeline

    split_kwargs: dict[str, Any] = {}

    def fake_load_from_raw_cif(**kwargs: Any) -> dict[str, Any]:
        return {
            "resampled_emdb": np.zeros((2, 2, 2), dtype=np.float32),
            "resampled_sim": None,
            "density_config": object(),
            "atom_coords": np.zeros((1, 3), dtype=np.float32),
            "atom_feat": np.zeros((1, 4), dtype=np.float32),
            "origin": np.zeros((3,), dtype=np.float32),
            "voxel_size": np.ones((3,), dtype=np.float32),
            "full_shape_zyx": (2, 2, 2),
            "hardmask": np.zeros((2, 2, 2), dtype=np.int64),
            "density_channel_names": ["emdb"],
        }

    def fake_split_volume_to_boxes(**kwargs: Any) -> list[dict[str, Any]]:
        split_kwargs.update(kwargs)
        return [_minimal_box_sample()]

    def fake_get_voxel_pred(**kwargs: Any) -> dict[str, Any]:
        return {
            "ligand_pred": np.zeros((2, 2, 2), dtype=np.float32),
            "head_weight_maps": {"ligand": np.ones((2, 2, 2), dtype=np.float32)},
        }

    monkeypatch.setattr(voxel_pipeline, "load_from_raw_cif", fake_load_from_raw_cif)
    monkeypatch.setattr(voxel_pipeline, "split_volume_to_boxes", fake_split_volume_to_boxes)
    monkeypatch.setattr(voxel_pipeline, "get_voxel_pred", fake_get_voxel_pred)

    cfg = {
        "use_cache": False,
        "save_cache": False,
        "structure_input_source": "cif_path",
        "sim_map_source": "sim_map_path",
        "cif_path": "sample.cif",
        "map_path": "sample.map",
        "sim_map_path": None,
        "target_voxel_size": 1.0,
        "compute_density": False,
        "select_first_model": True,
        "error_dir": str(tmp_path),
        "density_channel_config": {"enabled_channels": ["emdb"]},
        "window_size": 2,
        "stride": 2,
        "atom_buffer_radius": 4.0,
        "num_box_workers": 1,
        "batch_size": 1,
        "output_heads": ["ligand"],
        "merge_mode": "mean",
        "core_offset": 0,
        "show_progress": False,
        "eval_gt": False,
        "class_names": ["background", "foreground"],
        "cache_root": str(tmp_path),
        "sample_name": "sample",
    }

    voxel_pipeline._build_cache_or_forward(
        cfg_dict=cfg,
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        cache_path=str(tmp_path / "sample.npz"),
    )

    assert "valid_crop_margin" not in split_kwargs
    assert split_kwargs["atom_buffer_radius"] == 4.0
