"""
验证 _apply_synced_rotation 在立方体 BOX 下的同步旋转逻辑:
  - voxel_grid, voxel_label, hardmask 同步旋转
  - ligand_dist_map 同步旋转
  - atom_coord_local_voxel 与 voxel_grid 对齐
  - atom_coord_centered_world 与 atom_coord_world 关系一致
  - box_shape_zyx 不变 (立方体)
  - 非立方体 BOX 应被拒绝
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_cubic_sample(side: int = 4) -> dict:
    """构造一个最小立方体 BOX 的 numpy sample dict, 边长为 side。"""
    rng = np.random.RandomState(0)
    C = 2
    # np.ndarray, (C, D, H, W), float32, 体素特征: 每个通道用不同值填充
    voxel_grid = rng.rand(C, side, side, side).astype(np.float32)
    # np.ndarray, (D, H, W), int64, 在某个角打一个标签
    voxel_label = np.zeros((side, side, side), dtype=np.int64)
    voxel_label[0, 0, 0] = 1
    # np.ndarray, (D, H, W), int64, hardmask
    hardmask = np.zeros((side, side, side), dtype=np.int64)
    hardmask[1, 1, 1] = 1
    # np.ndarray, (D, H, W), float32, ligand 距离图
    ligand_dist_map = rng.rand(side, side, side).astype(np.float32)

    # 2 个原子: 都在 core box 内, 用于验证坐标同步旋转
    # np.ndarray, (2, 3), float32, atom local voxel (x, y, z)
    atom_coord_local_voxel = np.array(
        [[1.5, 1.5, 1.5], [0.2, 0.2, 0.2]], dtype=np.float32
    )
    box_origin_world = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    voxel_size_world = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    box_shape_zyx = np.array([side, side, side], dtype=np.int64)
    box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
    box_center_world = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world

    atom_coord_world = box_origin_world[None, :] + atom_coord_local_voxel * voxel_size_world[None, :]
    atom_coord_centered_world = atom_coord_world - box_center_world[None, :]
    atom_is_in_core_box = np.array([True, True], dtype=bool)

    return {
        "voxel_grid": voxel_grid,
        "voxel_label": voxel_label,
        "hardmask": hardmask,
        "ligand_dist_map": ligand_dist_map,
        "box_origin_world": box_origin_world,
        "voxel_size_world": voxel_size_world,
        "box_shape_zyx": box_shape_zyx,
        "atom_coord_world": atom_coord_world,
        "atom_coord_local_voxel": atom_coord_local_voxel,
        "atom_coord_centered_world": atom_coord_centered_world,
        "atom_is_in_core_box": atom_is_in_core_box,
    }


def _get_dataset_cls():
    """延迟导入, 避免在收集时就触发 import 问题。"""
    from src.datasets.box_point_dataset import BoxPointDataset
    return BoxPointDataset


def test_cubic_box_shape_unchanged_after_rotation() -> None:
    """立方体 BOX 旋转后, box_shape_zyx 应保持不变。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    # 创建一个空壳 dataset 只用它的旋转方法
    ds = object.__new__(DatasetCls)
    ds._sample_rotation_params = lambda: (0, 2, 1)  # Z-X 平面, k=1

    rotated = ds._apply_synced_rotation(sample)
    assert rotated["box_shape_zyx"].tolist() == [4, 4, 4]


def test_voxel_arrays_rotate_in_sync() -> None:
    """各 3D/4D voxel 数组在同一旋转下保持同步。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    orig_label = sample["voxel_label"].copy()
    orig_grid = sample["voxel_grid"].copy()
    orig_dist = sample["ligand_dist_map"].copy()

    ds = object.__new__(DatasetCls)
    axis1, axis2, k = 0, 1, 1  # Z-Y 平面, k=1
    ds._sample_rotation_params = lambda: (axis1, axis2, k)

    rotated = ds._apply_synced_rotation(sample)

    # 手动旋转验证
    expected_label = np.rot90(orig_label, k=k, axes=(axis1, axis2))
    expected_grid = np.rot90(orig_grid, k=k, axes=(axis1 + 1, axis2 + 1))
    expected_dist = np.rot90(orig_dist, k=k, axes=(axis1, axis2))

    np.testing.assert_array_equal(rotated["voxel_label"], expected_label)
    np.testing.assert_allclose(rotated["voxel_grid"], expected_grid, atol=1e-6)
    np.testing.assert_allclose(rotated["ligand_dist_map"], expected_dist, atol=1e-6)


def test_atom_coord_world_equals_centered_plus_center() -> None:
    """旋转后, atom_coord_world == atom_coord_centered_world + box_center_world。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)

    ds = object.__new__(DatasetCls)
    ds._sample_rotation_params = lambda: (1, 2, 3)  # Y-X 平面, k=3

    rotated = ds._apply_synced_rotation(sample)

    box_shape_xyz = rotated["box_shape_zyx"][[2, 1, 0]].astype(np.float32)
    box_center_world = rotated["box_origin_world"] + 0.5 * box_shape_xyz * rotated["voxel_size_world"]
    expected_world = rotated["atom_coord_centered_world"] + box_center_world[None, :]
    np.testing.assert_allclose(rotated["atom_coord_world"], expected_world, atol=1e-6)


def test_local_voxel_consistent_with_grid_rotation() -> None:
    """旋转后, 原子的 local voxel 坐标应该指向旋转后 grid 中正确的位置。

    方法: 在 (0,0,0) 处放一个有标记的 label=1, 原子位于 (0.5, 0.5, 0.5) local voxel (即该 voxel 中心)。
    旋转后, label=1 的位置和原子 floor(local_voxel) 的位置应一致。
    """
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    # 只保留一个原子, 放在 voxel (0,0,0) 的中心
    sample["atom_coord_local_voxel"] = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)  # (x, y, z)
    box_center_world = sample["box_origin_world"] + 2.0 * sample["voxel_size_world"]
    sample["atom_coord_world"] = sample["box_origin_world"][None, :] + sample["atom_coord_local_voxel"] * sample["voxel_size_world"][None, :]
    sample["atom_coord_centered_world"] = sample["atom_coord_world"] - box_center_world[None, :]
    sample["atom_is_in_core_box"] = np.array([True], dtype=bool)
    # label: 只在 (z=0, y=0, x=0) 处为 1
    sample["voxel_label"] = np.zeros((4, 4, 4), dtype=np.int64)
    sample["voxel_label"][0, 0, 0] = 1

    ds = object.__new__(DatasetCls)
    axis1, axis2, k = 0, 2, 1  # Z-X 平面, k=1
    ds._sample_rotation_params = lambda: (axis1, axis2, k)

    rotated = ds._apply_synced_rotation(sample)

    # 找到旋转后 label=1 的 voxel 位置 (z, y, x)
    label_pos = np.argwhere(rotated["voxel_label"] == 1)
    assert label_pos.shape[0] == 1
    label_zyx = label_pos[0]  # (z, y, x)

    # 原子的 local voxel -> floor -> (x_idx, y_idx, z_idx) -> 转成 (z, y, x)
    atom_xyz = rotated["atom_coord_local_voxel"][0]  # (x, y, z)
    atom_voxel_idx = np.floor(atom_xyz).astype(np.int64)  # (x, y, z) floor
    atom_zyx = atom_voxel_idx[[2, 1, 0]]  # (z, y, x)

    np.testing.assert_array_equal(atom_zyx, label_zyx)


def test_non_cubic_box_raises_error() -> None:
    """非立方体 BOX 应该抛出 ValueError。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    sample["box_shape_zyx"] = np.array([4, 4, 8], dtype=np.int64)  # 非立方体

    ds = object.__new__(DatasetCls)
    ds._sample_rotation_params = lambda: (0, 1, 1)

    with pytest.raises(ValueError, match="立方体"):
        ds._apply_synced_rotation(sample)


def test_k_zero_returns_unchanged() -> None:
    """k=0 (或 k=4) 时应该原样返回, 不做任何修改。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    orig_label = sample["voxel_label"].copy()

    ds = object.__new__(DatasetCls)
    ds._sample_rotation_params = lambda: (0, 1, 0)  # k=0

    result = ds._apply_synced_rotation(sample)
    np.testing.assert_array_equal(result["voxel_label"], orig_label)


def test_four_rotations_return_to_original() -> None:
    """连续 4 次 k=1 旋转应该回到原始状态。"""
    DatasetCls = _get_dataset_cls()
    sample = _make_cubic_sample(side=4)
    orig_label = sample["voxel_label"].copy()
    orig_atom = sample["atom_coord_local_voxel"].copy()

    ds = object.__new__(DatasetCls)

    for i in range(4):
        ds._sample_rotation_params = lambda: (0, 2, 1)  # 每次 k=1
        sample = ds._apply_synced_rotation(sample)

    np.testing.assert_array_equal(sample["voxel_label"], orig_label)
    np.testing.assert_allclose(sample["atom_coord_local_voxel"], orig_atom, atol=1e-5)
