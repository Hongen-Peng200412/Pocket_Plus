"""冻结 input_CLG_list、聚合 NPZ 冷读与 overlap index 语义测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.selector.dataset import (
    SelectorDataset,
    freeze_input_clg_list,
    recover_a_feat_l0,
)


def _write_zero_clg_unet_pdb(root: Path, split: str, pdb_id: str) -> None:
    """写出已完整发布但不含任何 CLG 的最小 PDB。"""

    pdb_root = root / "unet_c1" / split / pdb_id
    (pdb_root / "status" / "CLG_centered").mkdir(parents=True)
    (pdb_root / "status" / "CLG_centered" / "_COMPLETE").write_text(
        "", encoding="utf-8"
    )
    (pdb_root / "centered").mkdir(parents=True)
    (pdb_root / "components").mkdir(parents=True)
    np.savez_compressed(
        pdb_root / "centered" / "CLG_centered.npz",
        centered_box_index=np.empty(0, dtype=np.int32),
    )
    np.savez_compressed(
        pdb_root / "components" / "clg.npz",
        CLG_id=np.empty(0, dtype=np.int32),
        candidate_offsets=np.zeros(1, dtype=np.int64),
    )
    np.savez_compressed(
        pdb_root / "components" / "forest.npz",
        tree_id=np.empty(0, dtype=np.int32),
    )


def _write_unet_pdb(root: Path, upstream: Path, split: str, pdb_id: str) -> None:
    """写出一个字段完整、数值极小的 unet_c1 CLG_centered 测试 PDB。"""
    pdb_root = root / "unet_c1" / split / pdb_id
    (pdb_root / "status" / "CLG_centered").mkdir(parents=True)
    (pdb_root / "status" / "CLG_centered" / "_COMPLETE").write_text("", encoding="utf-8")
    (pdb_root / "centered").mkdir(parents=True)
    (pdb_root / "components").mkdir(parents=True)
    (pdb_root / "probability").mkdir(parents=True)

    shape = (80, 80, 80)
    linear_a = np.ravel_multi_index((1, 1, 1), shape)
    linear_b = np.ravel_multi_index((2, 2, 2), shape)
    probability = np.zeros(shape, dtype=np.float32)
    probability.reshape(-1)[linear_a] = 0.8
    probability.reshape(-1)[linear_b] = 0.6
    np.savez_compressed(pdb_root / "probability" / "probability_map.npz", probability_map=probability)

    np.savez_compressed(
        pdb_root / "components" / "forest.npz",
        tree_id=np.asarray([0, 0, 0], dtype=np.int32),
        node_id=np.asarray([0, 1, 2], dtype=np.int32),
        threshold_grid_index=np.asarray([100, 200, 200], dtype=np.int32),
        threshold_value=np.asarray([100, 200, 200], dtype=np.float32) / 32768.0,
        parent_node_id=np.asarray([-1, 0, 0], dtype=np.int32),
        children_offsets=np.asarray([0, 2, 2, 2], dtype=np.int64),
        children_node_id=np.asarray([1, 2], dtype=np.int32),
        node_voxel_offsets=np.asarray([0, 2, 3, 4], dtype=np.int64),
        node_voxel_global_linear_index=np.asarray([linear_a, linear_b, linear_a, linear_b], dtype=np.int64),
        voxel_count=np.asarray([2, 1, 1], dtype=np.int32),
        bbox_min_zyx=np.asarray([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=np.int32),
        bbox_max_zyx=np.asarray([[2, 2, 2], [1, 1, 1], [2, 2, 2]], dtype=np.int32),
        centroid_zyx=np.asarray([[1.5, 1.5, 1.5], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
        probability_mean=np.asarray([0.7, 0.8, 0.6], dtype=np.float32),
        probability_max=np.asarray([0.8, 0.8, 0.6], dtype=np.float32),
        candidate_eligible=np.asarray([True, True, True]),
        ineligible_reason_code=np.zeros((3,), dtype=np.uint8),
    )
    np.savez_compressed(
        pdb_root / "components" / "clg.npz",
        CLG_id=np.asarray([0], dtype=np.int32),
        tree_id=np.asarray([0], dtype=np.int32),
        CLG_seed_node_id=np.asarray([1], dtype=np.int32),
        CLG_oldest_node_id=np.asarray([0], dtype=np.int32),
        candidate_offsets=np.asarray([0, 3], dtype=np.int64),
        candidate_node_id=np.asarray([0, 1, 2], dtype=np.int32),
        candidate_threshold_grid_index=np.asarray([100, 200, 200], dtype=np.int32),
    )
    np.savez_compressed(
        pdb_root / "components" / "overlap.npz",
        candidate_occurrence_offsets=np.asarray([0, 1, 2, 2], dtype=np.int64),
        overlap_occurrence_index=np.asarray([0, 1], dtype=np.int32),
        intersection_voxel_count=np.asarray([1, 1], dtype=np.int32),
        occurrence_id=np.asarray([42, 99], dtype=np.int32),
        occurrence_voxel_count=np.asarray([2, 2], dtype=np.int32),
    )
    np.savez_compressed(
        pdb_root / "centered" / "CLG_centered.npz",
        centered_box_index=np.asarray([0], dtype=np.int32),
        box_start_zyx=np.asarray([[0, 0, 0]], dtype=np.int32),
        box_shape_zyx=np.asarray([[80, 80, 80]], dtype=np.uint8),
        box_origin_world=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        voxel_size_world=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        source_tree_id=np.asarray([0], dtype=np.int32),
        source_node_id=np.asarray([0], dtype=np.int32),
        source_threshold_grid_index=np.asarray([100], dtype=np.int32),
        source_threshold_value=np.asarray([100 / 32768.0], dtype=np.float32),
        CLG_id=np.asarray([0], dtype=np.int32),
        CLG_seed_node_id=np.asarray([1], dtype=np.int32),
        CLG_oldest_node_id=np.asarray([0], dtype=np.int32),
        voxel_offsets=np.asarray([0, 2], dtype=np.int64),
        voxel_index_local_zyx=np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.int16),
        centered_probability=np.asarray([0.75, 0.55], dtype=np.float32),
        voxel_final=np.arange(96, dtype=np.float16).reshape(2, 48),
        voxel_ds_2=np.zeros((1, 2, 4, 4, 4), dtype=np.float16),
        voxel_ds_3=np.zeros((1, 2, 2, 2, 2), dtype=np.float16),
        voxel_ds_4=np.zeros((1, 2, 1, 1, 1), dtype=np.float16),
        voxel_c4=np.zeros((1, 2, 1, 1, 1), dtype=np.float16),
        candidate_offsets=np.asarray([0, 3], dtype=np.int64),
        candidate_node_id=np.asarray([0, 1, 2], dtype=np.int32),
        candidate_threshold_grid_index=np.asarray([100, 200, 200], dtype=np.int32),
        candidate_voxel_offsets=np.asarray([0, 2, 3, 4], dtype=np.int64),
        candidate_voxel_index=np.asarray([0, 1, 0, 1], dtype=np.int32),
    )
    density_root = upstream / "density" / pdb_id
    density_root.mkdir(parents=True)
    experimental = np.linspace(-1.0, 1.0, num=np.prod(shape), dtype=np.float32).reshape((1, *shape))
    np.savez(density_root / "exp.npz", grid=experimental)


def test_freeze_is_run_scoped_and_dataset_does_not_grow(tmp_path: Path) -> None:
    """启动扫描后的清单应固定；后来发布的新 PDB 不改变当前 Dataset。"""
    outputs = tmp_path / "stage1_outputs"
    upstream = tmp_path / "upstream"
    _write_unet_pdb(outputs, upstream, "train", "1abc")
    frozen = freeze_input_clg_list(
        stage1_outputs_root=outputs,
        selector_run_dir=tmp_path / "run_a",
        stage1_model_name="unet_c1",
        split_order=("train",),
        input_clg_list_path=None,
        formal_run=False,
        expected_validation_pdb_ids_path=None,
    )
    dataset = SelectorDataset(
        input_clg_list_path=frozen,
        stage1_outputs_root=outputs,
        upstream_root=upstream,
        split="train",
        lambda_count=0.05,
        require_oracle=True,
        density_clip_percentile=(0.001, 0.999),
        pdb_cache_size=1,
    )
    assert len(dataset) == 1

    _write_unet_pdb(outputs, upstream, "train", "2def")
    assert len(dataset) == 1
    resumed = freeze_input_clg_list(
        stage1_outputs_root=outputs,
        selector_run_dir=tmp_path / "run_a",
        stage1_model_name="unet_c1",
        split_order=("train",),
        input_clg_list_path=None,
        formal_run=False,
        expected_validation_pdb_ids_path=None,
    )
    resumed_payload = json.loads(resumed.read_text(encoding="utf-8"))
    assert resumed_payload["split_counts"] == {"train": 1}
    copied = freeze_input_clg_list(
        stage1_outputs_root=outputs,
        selector_run_dir=tmp_path / "run_b",
        stage1_model_name="unet_c1",
        split_order=("train",),
        input_clg_list_path=frozen,
        formal_run=False,
        expected_validation_pdb_ids_path=None,
    )
    payload = json.loads(copied.read_text(encoding="utf-8"))
    assert payload["split_counts"] == {"train": 1}


def test_a_feat_l0_is_recovered_by_global_identity_without_reordering() -> None:
    """A_feat_L0 不重复落盘，并严格按 A_global_index 保持 centered 行序。"""

    receptor_feat = np.arange(5 * 49, dtype=np.float32).reshape(5, 49)
    a_global_index = np.asarray([4, 1, 3], dtype=np.int64)

    recovered = recover_a_feat_l0(receptor_feat, a_global_index)

    assert recovered.dtype == np.float32
    np.testing.assert_array_equal(recovered, receptor_feat[[4, 1, 3]])
    recovered[0, 0] = -1.0
    assert receptor_feat[4, 0] != -1.0

    with pytest.raises(ValueError, match="越过"):
        recover_a_feat_l0(receptor_feat, np.asarray([5], dtype=np.int64))


def test_dataset_reconstructs_v_only_sample_and_online_oracle(tmp_path: Path) -> None:
    """一个聚合 CLG entry 应恢复 memberships、16D 属性、8D tree pair 与 q。"""
    outputs = tmp_path / "stage1_outputs"
    upstream = tmp_path / "upstream"
    _write_unet_pdb(outputs, upstream, "train", "1abc")
    frozen = freeze_input_clg_list(
        stage1_outputs_root=outputs,
        selector_run_dir=tmp_path / "run",
        stage1_model_name="unet_c1",
        split_order=("train",),
        input_clg_list_path=None,
        formal_run=False,
        expected_validation_pdb_ids_path=None,
    )
    dataset = SelectorDataset(
        input_clg_list_path=frozen,
        stage1_outputs_root=outputs,
        upstream_root=upstream,
        split="train",
        lambda_count=0.05,
        require_oracle=True,
        density_clip_percentile=(0.001, 0.999),
        pdb_cache_size=1,
    )
    sample = dataset[0]
    assert sample["candidate_attributes"].shape == (3, 16)
    assert sample["tree_relative_feature"].shape == (3, 3, 8)
    assert sample["V_sources"]["voxel_final"].shape == (2, 48)
    assert sample["density_input"].shape == (1, 1, 80, 80, 80)
    np.testing.assert_allclose(sample["candidate_max_iou"].numpy(), [1 / 3, 0.5, 0.0])
    assert sample["closure_candidate_index_by_node"] == (0, 1, 2)
    assert "P_sources" not in sample and "A_sources" not in sample


def test_formal_run_requires_all_expected_validation_pdbs(tmp_path: Path) -> None:
    """formal_run 不得在固定 validation PDB 尚未全部可读时产生 BEST。"""
    outputs = tmp_path / "stage1_outputs"
    upstream = tmp_path / "upstream"
    _write_unet_pdb(outputs, upstream, "validation", "1abc")
    expected = tmp_path / "validation.json"
    expected.write_text(json.dumps(["1abc", "2def"]), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        freeze_input_clg_list(
            stage1_outputs_root=outputs,
            selector_run_dir=tmp_path / "run",
            stage1_model_name="unet_c1",
            split_order=("validation",),
            input_clg_list_path=None,
            formal_run=True,
            expected_validation_pdb_ids_path=expected,
        )


def test_zero_clg_pdb_remains_in_frozen_inventory_and_satisfies_formal_run(
    tmp_path: Path,
) -> None:
    """零 CLG 是完整 PDB 结果，不能从正式 validation 完整性检查中消失。"""

    outputs = tmp_path / "stage1_outputs"
    _write_zero_clg_unet_pdb(outputs, "validation", "empty")
    expected = tmp_path / "validation.json"
    expected.write_text(json.dumps(["empty"]), encoding="utf-8")

    frozen = freeze_input_clg_list(
        stage1_outputs_root=outputs,
        selector_run_dir=tmp_path / "run",
        stage1_model_name="unet_c1",
        split_order=("validation",),
        input_clg_list_path=None,
        formal_run=True,
        expected_validation_pdb_ids_path=expected,
    )

    payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert payload["items"] == []
    assert payload["split_counts"] == {"validation": 0}
    assert payload["pdb_ids_by_split"] == {"validation": ["empty"]}
    assert payload["split_pdb_counts"] == {"validation": 1}
