from __future__ import annotations

import numpy as np

from src.inference.utils.voxel_types import VoxelPredCacheData
from src.inference.voxel_tuning import optimize_postprocess_params_by_class, select_cache_view_for_class


def _multiclass_cache() -> VoxelPredCacheData:
    ligand_pred = np.zeros((3, 2, 2, 2), dtype=np.float32)
    ligand_pred[1, 0, 0, 0] = 0.9
    ligand_pred[2, 1, 1, 1] = 0.9
    receptor_pred = np.zeros((3, 2, 2, 2), dtype=np.float32)
    receptor_pred[1] = 0.1
    receptor_pred[2] = 0.2
    metal_mask = np.zeros((2, 2, 2), dtype=bool)
    metal_mask[0, 0, 0] = True
    small_mask = np.zeros((2, 2, 2), dtype=bool)
    small_mask[1, 1, 1] = True
    metal_label = metal_mask.astype(np.int32)
    small_label = small_mask.astype(np.int32)
    return VoxelPredCacheData(
        ligand_pred=ligand_pred,
        receptor_pred=receptor_pred,
        hardmask=np.zeros((2, 2, 2), dtype=np.int64),
        resampled_emdb=np.zeros((2, 2, 2), dtype=np.float32),
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        gt_ligand_mask=np.logical_or(metal_mask, small_mask),
        gt_instance_label=(metal_label + small_label).astype(np.int32),
        gt_ligand_mask_by_class={"metal_ion": metal_mask, "small_molecule": small_mask},
        gt_instance_label_by_class={"metal_ion": metal_label, "small_molecule": small_label},
        gt_instance_meta=[{"class_id": 1}, {"class_id": 2}],
        meta={"class_names": ["background", "metal_ion", "small_molecule"]},
    )


def test_select_class_cache_view_uses_class_prediction_and_gt() -> None:
    data = _multiclass_cache()

    class_data = select_cache_view_for_class(data, "small_molecule")

    assert np.array_equal(class_data.ligand_pred, data.ligand_pred[2])
    assert np.array_equal(class_data.receptor_pred, data.receptor_pred[2])
    assert np.array_equal(class_data.gt_ligand_mask, data.gt_ligand_mask_by_class["small_molecule"])
    assert class_data.meta["selected_class_id"] == 2


def test_select_class_cache_view_accepts_absent_class_zero_gt() -> None:
    data = _multiclass_cache()
    data.gt_ligand_mask_by_class["metal_ion"] = np.zeros((2, 2, 2), dtype=bool)
    data.gt_instance_label_by_class["metal_ion"] = np.zeros((2, 2, 2), dtype=np.int32)

    class_data = select_cache_view_for_class(data, "metal_ion")

    assert class_data.gt_ligand_mask.sum() == 0
    assert class_data.gt_instance_label.sum() == 0


def test_optimize_postprocess_params_by_class_writes_independent_best_params() -> None:
    data = _multiclass_cache()
    fixed_postprocess_params = {
        "threshold": 0.2,
        "min_component_voxels": 1,
        "filter_strength": "basic",
        "connectivity_policy": "7_none",
        "sigma_nearby": 1.0,
        "kernel_nearby": 3,
        "sigma_response": 1.0,
        "kernel_response": 3,
        "score_add": 1.0,
        "score_minus": 1.0,
        "voxel_score_min": 0.0,
        "instance_score_min": 0.0,
        "merge_dist": 0.0,
    }

    result = optimize_postprocess_params_by_class(
        cache_paths=[],
        loaded_cache_items=[("cache.npz", data)],
        cache_data_mode="memory",
        fixed_postprocess_params=fixed_postprocess_params,
        search_space={"threshold": {"values": [0.2, 0.8]}},
        search_space_by_class={},
        search_strategy="grid",
        eval_params={"compute_instance_metrics": False, "alpha": 0.5, "beta": 0.5},
        optimizer_params={
            "objective_expr": "avg_voxel_f1",
            "fixed_search_params": [],
            "max_iter": 1,
            "popsize": 1,
            "random_seed": 1,
        },
        n_jobs=1,
        show_progress=False,
    )

    assert set(result["best_params"]["by_class"]) == {"metal_ion", "small_molecule"}
    assert "threshold" in result["best_params"]["by_class"]["metal_ion"]
    assert set(result["best_metrics"]["by_class"]) == {"metal_ion", "small_molecule"}
    assert "macro" in result["best_metrics"]
