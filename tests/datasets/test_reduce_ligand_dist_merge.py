"""
验证 _load_box_npz_raw 内联的 ligand_dist reduce 逻辑:
  - 无 class_mapping 时, 全通道 min
  - 二分类 class_mapping 时, 选中通道后 min
  - 多分类 class_mapping 时, 输出含背景占位通道的类别距离图
  - 无 ligand_dist 目录时, 返回 None
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.box_point_collate import box_point_collate
from src.datasets.box_point_dataset import BoxPointDataset


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _make_meta_arrays():
    """生成一套合法的 npz 元信息数组。"""
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    x_range = np.array([0.0, 4.0], dtype=np.float32)
    y_range = np.array([0.0, 4.0], dtype=np.float32)
    z_range = np.array([0.0, 4.0], dtype=np.float32)
    return dict(origin=origin, voxel_size=voxel_size,
                x_range=x_range, y_range=y_range, z_range=z_range)


def _build_dataset_with_ligand_dist(
    tmp_path: Path,
    *,
    class_mapping: list[int] | None = None,
    ligand_dist_grid: np.ndarray | None = None,
) -> BoxPointDataset:
    """构造一个包含 emdb_exp_BOX + pdb_label_BOX + ligand_dist_BOX 的最小数据集。"""
    class_name = "small_molecule"
    sample_name = "abcd_7_0_0_0_C"

    split_path = tmp_path / "split.json"
    random_split_path = tmp_path / "random_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([sample_name]), encoding="utf-8")
    random_split_path.write_text(json.dumps([]), encoding="utf-8")

    meta = _make_meta_arrays()

    all_data_path = tmp_path / "all_data"

    # emdb_exp_BOX: (1, 4, 4, 4) float32
    emdb_grid = np.ones((1, 4, 4, 4), dtype=np.float32)
    _write_npz(all_data_path / "emdb_exp_BOX" / class_name / f"{sample_name}.npz",
               grid=emdb_grid, **meta)

    # emdb_sim_BOX: (1, 4, 4, 4) float32
    sim_grid = np.ones((1, 4, 4, 4), dtype=np.float32)
    _write_npz(all_data_path / "emdb_sim_BOX" / class_name / f"{sample_name}.npz",
               grid=sim_grid, **meta)

    # pdb_label_BOX: (1, 4, 4, 4) float32 label
    label_grid = np.zeros((1, 4, 4, 4), dtype=np.float32)
    _write_npz(all_data_path / "pdb_label_BOX" / class_name / f"{sample_name}.npz",
               grid=label_grid, **meta)

    # 决定 data_folder_names
    folder_names = ["emdb_exp_BOX", "emdb_sim_BOX", "pdb_label_BOX"]

    if ligand_dist_grid is not None:
        _write_npz(all_data_path / "ligand_dist_BOX" / class_name / f"{sample_name}.npz",
                   grid=ligand_dist_grid, **meta)
        folder_names.append("ligand_dist_BOX")

    # 最小结构数据
    sample_root_path = tmp_path / "structures"
    atom_coords = np.array([[1.5, 1.5, 1.5]], dtype=np.float32)
    atom_features = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    binding_mask = np.array([True], dtype=bool)
    pocket_class_ids = np.array([1], dtype=np.int64)
    instance_ids = np.array([7], dtype=np.int64)
    _write_npz(sample_root_path / "abcd" / "atoms.npz",
               coords=atom_coords, features=atom_features)
    _write_npz(sample_root_path / "abcd" / "labels.npz",
               binding_mask=binding_mask, pocket_class_ids=pocket_class_ids,
               instance_ids=instance_ids)

    return BoxPointDataset(
        all_data_path=str(all_data_path),
        sample_root_path=str(sample_root_path),
        split_file=[str(split_path), str(random_split_path)],
        mode="val",
        data_folder_names=folder_names,
        class_folder_names=[class_name, "random_BOX"],
        class_mapping=class_mapping,
        atom_buffer_radius=0.0,
        enable_random_rotation=False,
    )


def test_no_ligand_dist_returns_none(tmp_path: Path) -> None:
    """不含 ligand_dist 目录时, ligand_dist_map 应该是 None。"""
    dataset = _build_dataset_with_ligand_dist(tmp_path, ligand_dist_grid=None)
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")
    assert box_raw["ligand_dist_map"] is None


def test_reduce_no_class_mapping(tmp_path: Path) -> None:
    """无 class_mapping 时, 对所有通道做 min 归约。"""
    # 3 通道, (3, 4, 4, 4)
    rng = np.random.RandomState(42)
    ligand_grid = rng.rand(3, 4, 4, 4).astype(np.float32) + 1.0  # 确保正数

    dataset = _build_dataset_with_ligand_dist(
        tmp_path, ligand_dist_grid=ligand_grid, class_mapping=None,
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (4, 4, 4)
    # np.ndarray, (4, 4, 4), float32, 期望全通道 min
    expected = np.min(ligand_grid, axis=0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_reduce_with_class_mapping(tmp_path: Path) -> None:
    """有 class_mapping 时, 只选映射到前景(>0)的通道做 min 归约。
    class_mapping = [0, 1, 1] 表示原始 class 1 和 2 都映射到前景 1。
    距离图通道 i 对应 class_id = i+1 (见 bind.py), 所以:
      - channel 0 → class_id 1, mapping[1]=1 (前景) → 选中
      - channel 1 → class_id 2, mapping[2]=1 (前景) → 选中
    selected_channels = [0, 1], min(5.0, 3.0) = 3.0
    """
    ch0 = np.full((4, 4, 4), 5.0, dtype=np.float32)
    ch1 = np.full((4, 4, 4), 3.0, dtype=np.float32)
    ligand_grid = np.stack([ch0, ch1], axis=0)

    dataset = _build_dataset_with_ligand_dist(
        tmp_path, ligand_dist_grid=ligand_grid, class_mapping=[0, 1, 1],
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (4, 4, 4)
    # selected_channels = [0, 1], min(5.0, 3.0) = 3.0
    expected = np.full((4, 4, 4), 3.0, dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_reduce_with_class_mapping_single_class(tmp_path: Path) -> None:
    """class_mapping = [0, 1], 只有一个前景类, 应选 channel 0。"""
    ch0 = np.full((4, 4, 4), 2.0, dtype=np.float32)
    ligand_grid = np.stack([ch0], axis=0)

    dataset = _build_dataset_with_ligand_dist(
        tmp_path, ligand_dist_grid=ligand_grid, class_mapping=[0, 1],
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (4, 4, 4)
    # selected_channels = [0], min over single channel = 2.0
    expected = np.full((4, 4, 4), 2.0, dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_reduce_excludes_background_mapped_channels(tmp_path: Path) -> None:
    """二分类映射把原始类映射到背景时, 对应通道应被排除。"""
    ch0 = np.full((4, 4, 4), 1.0, dtype=np.float32)
    ch1 = np.full((4, 4, 4), 5.0, dtype=np.float32)
    ch2 = np.full((4, 4, 4), 3.0, dtype=np.float32)
    ligand_grid = np.stack([ch0, ch1, ch2], axis=0)

    dataset = _build_dataset_with_ligand_dist(
        tmp_path, ligand_dist_grid=ligand_grid, class_mapping=[0, 0, 1, 1],
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (4, 4, 4)
    expected = np.full((4, 4, 4), 3.0, dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_multiclass_mapping_inserts_background_placeholder_channel(tmp_path: Path) -> None:
    """多分类距离图应输出 (C,D,H,W), 其中 channel 0 是 inf 背景占位。"""
    ch0 = np.full((4, 4, 4), 2.0, dtype=np.float32)
    ch1 = np.full((4, 4, 4), 4.0, dtype=np.float32)
    ch2 = np.full((4, 4, 4), 6.0, dtype=np.float32)
    ch3 = np.full((4, 4, 4), 8.0, dtype=np.float32)
    ligand_grid = np.stack([ch0, ch1, ch2, ch3], axis=0)

    dataset = _build_dataset_with_ligand_dist(
        tmp_path,
        ligand_dist_grid=ligand_grid,
        class_mapping=[0, 1, 0, 0, 2],
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (3, 4, 4, 4)
    assert np.isinf(result[0]).all()
    np.testing.assert_allclose(result[1], ch0, atol=1e-6)
    np.testing.assert_allclose(result[2], ch3, atol=1e-6)


def test_multiclass_mapping_merges_original_classes_by_min_distance(tmp_path: Path) -> None:
    """多个原始类别映射到同一新类别时, 新类别距离通道应取逐体素最小值。"""
    ch0 = np.full((4, 4, 4), 9.0, dtype=np.float32)
    ch1 = np.full((4, 4, 4), 5.0, dtype=np.float32)
    ch2 = np.full((4, 4, 4), 3.0, dtype=np.float32)
    ch3 = np.full((4, 4, 4), 7.0, dtype=np.float32)
    ligand_grid = np.stack([ch0, ch1, ch2, ch3], axis=0)

    dataset = _build_dataset_with_ligand_dist(
        tmp_path,
        ligand_dist_grid=ligand_grid,
        class_mapping=[0, 1, 2, 2, 1],
    )
    box_raw = dataset._load_box_npz_raw("small_molecule", "abcd_7_0_0_0_C")

    result = box_raw["ligand_dist_map"]
    assert result is not None
    assert result.shape == (3, 4, 4, 4)
    assert np.isinf(result[0]).all()
    np.testing.assert_allclose(result[1], np.minimum(ch0, ch3), atol=1e-6)
    np.testing.assert_allclose(result[2], np.minimum(ch1, ch2), atol=1e-6)


def test_collate_preserves_multiclass_ligand_dist_channel_dimension(tmp_path: Path) -> None:
    """collate 后多分类 ligand_dist_map 应保持为 (B,C,D,H,W)。"""
    ligand_grid = np.stack(
        [
            np.full((4, 4, 4), 2.0, dtype=np.float32),
            np.full((4, 4, 4), 4.0, dtype=np.float32),
            np.full((4, 4, 4), 6.0, dtype=np.float32),
            np.full((4, 4, 4), 8.0, dtype=np.float32),
        ],
        axis=0,
    )
    dataset = _build_dataset_with_ligand_dist(
        tmp_path,
        ligand_dist_grid=ligand_grid,
        class_mapping=[0, 1, 0, 0, 2],
    )

    sample = dataset[0]
    batch = box_point_collate([sample, sample])

    assert batch["ligand_dist_map"].shape == (2, 3, 4, 4, 4)
    assert torch.isinf(batch["ligand_dist_map"][:, 0]).all()
