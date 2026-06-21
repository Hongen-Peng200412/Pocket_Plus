from __future__ import annotations

import numpy as np

from src.inference.voxel_tuning import load_voxel_prediction_cache, save_voxel_prediction_cache


def test_cache_roundtrip_ligand_receptor_gt(tmp_path: Path) -> None:
    cache_path = str(tmp_path / "cache.npz")
    ligand = np.ones((2, 2, 2), dtype=np.float32)
    receptor = np.zeros((2, 2, 2), dtype=np.float32)
    hardmask = np.zeros((2, 2, 2), dtype=np.int64)
    gt_mask = np.zeros((2, 2, 2), dtype=bool)
    gt_label = np.zeros((2, 2, 2), dtype=np.int32)
    gt_label[0, 0, 0] = 1

    save_voxel_prediction_cache(
        cache_path=cache_path,
        ligand_pred=ligand,
        receptor_pred=receptor,
        hardmask=hardmask,
        resampled_emdb=ligand,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        meta={"sample_name": "x"},
        gt_ligand_mask=gt_mask,
        gt_instance_label=gt_label,
        gt_ligand_mask_by_class=None,
        gt_instance_label_by_class=None,
        gt_instance_meta=[{"instance_id": 1}],
    )
    loaded = load_voxel_prediction_cache(cache_path)

    assert np.array_equal(loaded.ligand_pred, ligand)
    assert np.array_equal(loaded.receptor_pred, receptor)
    assert np.array_equal(loaded.gt_instance_label, gt_label)
    assert loaded.gt_instance_meta == [{"instance_id": 1}]
    assert loaded.meta["sample_name"] == "x"


def test_cache_roundtrip_multiclass_gt_by_class(tmp_path: Path) -> None:
    cache_path = str(tmp_path / "cache_multi.npz")
    ligand = np.ones((3, 2, 2, 2), dtype=np.float32)
    receptor = np.zeros((3, 2, 2, 2), dtype=np.float32)
    hardmask = np.zeros((2, 2, 2), dtype=np.int64)
    gt_mask = np.zeros((2, 2, 2), dtype=bool)
    gt_label = np.zeros((2, 2, 2), dtype=np.int32)
    metal_mask = np.zeros((2, 2, 2), dtype=bool)
    metal_mask[0, 0, 0] = True
    small_mask = np.zeros((2, 2, 2), dtype=bool)
    small_mask[1, 1, 1] = True
    metal_label = metal_mask.astype(np.int32)
    small_label = small_mask.astype(np.int32)

    save_voxel_prediction_cache(
        cache_path=cache_path,
        ligand_pred=ligand,
        receptor_pred=receptor,
        hardmask=hardmask,
        resampled_emdb=ligand[0],
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        meta={"sample_name": "x", "class_names": ["background", "metal_ion", "small_molecule"]},
        gt_ligand_mask=gt_mask,
        gt_instance_label=gt_label,
        gt_ligand_mask_by_class={"metal_ion": metal_mask, "small_molecule": small_mask},
        gt_instance_label_by_class={"metal_ion": metal_label, "small_molecule": small_label},
        gt_instance_meta=[{"class_id": 1, "class_instance_id": 1}],
    )
    loaded = load_voxel_prediction_cache(cache_path)

    assert loaded.ligand_pred.shape == (3, 2, 2, 2)
    assert loaded.receptor_pred is not None
    assert loaded.gt_ligand_mask_by_class is not None
    assert loaded.gt_instance_label_by_class is not None
    assert set(loaded.gt_ligand_mask_by_class) == {"metal_ion", "small_molecule"}
    assert np.array_equal(loaded.gt_ligand_mask_by_class["metal_ion"], metal_mask)
    assert np.array_equal(loaded.gt_instance_label_by_class["small_molecule"], small_label)
    assert loaded.gt_instance_meta == [{"class_id": 1, "class_instance_id": 1}]
