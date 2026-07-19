"""三类 centered 的权威 voxel、CLG membership 与 Selected 状态测试。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from src.artifacts.io import pack_centered_entries
from src.component_lineage.structures import CLG, ComponentNode
from src.inference.centered import (
    CenteredGeometry,
    adapt_stage1_centered_output,
    produce_clg_centered_entries,
    produce_f1_centered_entries,
    produce_selected_refined_entries,
    resolve_component_centered_start,
)
from src.selector.dataset import recover_a_feat_l0


FULL_SHAPE = (80, 80, 80)
FIXED_GRIDS = {
    "voxel_ds_2": np.zeros((256, 20, 20, 20), dtype=np.float16),
    "voxel_ds_3": np.zeros((256, 10, 10, 10), dtype=np.float16),
    "voxel_ds_4": np.zeros((256, 5, 5, 5), dtype=np.float16),
    "voxel_c4": np.zeros((256, 5, 5, 5), dtype=np.float16),
}


def _node(node_id: int, voxels: list[tuple[int, int, int]], threshold: float = 0.5) -> ComponentNode:
    zyx = np.asarray(voxels, dtype=np.int64)
    linear = np.ravel_multi_index(zyx.T, FULL_SHAPE).astype(np.int64)
    linear.sort()
    return ComponentNode(
        tree_id=0,
        node_id=node_id,
        threshold_grid_index=int(threshold * 32768),
        threshold_value=threshold,
        voxel_global_linear_index=linear,
        bbox_min_zyx=zyx.min(axis=0),
        bbox_max_zyx=zyx.max(axis=0),
        centroid_zyx=zyx.mean(axis=0),
        probability_mean=0.8,
        probability_max=0.9,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )


def _payload(probability: np.ndarray) -> dict[str, np.ndarray]:
    # 广播视图避免为测试物化整张 48 通道 full-resolution feature。
    voxel_final = np.broadcast_to(
        np.arange(48, dtype=np.float16)[:, None, None, None],
        (48, 80, 80, 80),
    )
    return {
        "ligand_probability": probability,
        "hardmask": np.zeros(FULL_SHAPE, dtype=np.bool_),
        "voxel_aux_probability_grid": np.full(FULL_SHAPE, 0.25, dtype=np.float32),
        "voxel_final_grid": voxel_final,
        **FIXED_GRIDS,
    }


def test_f1_and_clg_keep_global_source_identity_and_memberships() -> None:
    oldest = _node(0, [(10, 10, 10), (10, 10, 11)])
    child = _node(1, [(10, 10, 10)], threshold=0.75)
    child.parent = oldest
    oldest.children.append(child)
    clg = CLG(0, 0, seed_node=child, oldest_node=oldest, candidate_nodes=(child, oldest))
    geometry = CenteredGeometry(
        full_shape_zyx=FULL_SHAPE,
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )

    requests = []

    def callback(request):
        requests.append(request)
        probability = np.zeros(FULL_SHAPE, dtype=np.float32)
        probability[10, 10, 10:12] = 0.8
        return _payload(probability)

    f1_entries = produce_f1_centered_entries(
        nodes=(oldest,),
        f1_threshold_grid_index=oldest.threshold_grid_index,
        geometry=geometry,
        stage1_model_name="unet_c1",
        split="train",
        pdb_id="sample_001",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        full_forward=callback,
    )
    f1_arrays = pack_centered_entries(f1_entries, "F1_centered")
    assert f1_arrays["source_node_id"].tolist() == [0]
    assert f1_arrays["voxel_offsets"].tolist() == [0, 2]
    assert np.allclose(f1_arrays["centered_probability"], 0.8)
    assert "CLG_id" not in f1_arrays

    clg_entries = produce_clg_centered_entries(
        clgs=(clg,),
        geometry=geometry,
        stage1_model_name="unet_c1",
        split="train",
        pdb_id="sample_001",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        full_forward=callback,
    )
    clg_arrays = pack_centered_entries(clg_entries, "CLG_centered")
    assert clg_arrays["candidate_offsets"].tolist() == [0, 2]
    assert clg_arrays["candidate_voxel_offsets"].tolist() == [0, 1, 3]
    assert clg_arrays["candidate_voxel_index"].tolist() == [0, 0, 1]
    assert {(request.stage1_model_name, request.split, request.pdb_id) for request in requests} == {
        ("unet_c1", "train", "sample_001")
    }


def test_selected_status_codes_and_refined_blob_semantics(caplog) -> None:
    nodes = [
        _node(0, [(10, 10, 10), (10, 10, 11), (10, 10, 12)]),
        _node(1, [(20, 20, 20)]),
        _node(2, [(30, 30, 30)]),
        _node(3, [(40, 40, 40)]),
    ]
    geometry = CenteredGeometry(
        full_shape_zyx=FULL_SHAPE,
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )

    def callback(request):
        if request.source_node_id == 3:
            raise RuntimeError("模拟 forward 失败")
        probability = np.zeros(FULL_SHAPE, dtype=np.float32)
        if request.source_node_id == 0:
            # 两个局部组件与 source 的 IoU 同为 1/3；自然 CCL label 顺序应取 voxel 10。
            probability[10, 10, 10] = 0.9
            probability[10, 10, 12] = 0.9
            probability[70, 70, 70] = 0.9
        elif request.source_node_id == 2:
            probability[0, 0, 0] = 0.9
        return _payload(probability)

    with caplog.at_level(logging.ERROR, logger="src.inference.centered"):
        entries = produce_selected_refined_entries(
            selected_nodes=nodes,
            geometry=geometry,
            stage1_model_name="unet_c1",
            split="validation",
            pdb_id="sample_002",
            resolve_box_start=lambda centroid, shape: (0, 0, 0),
            full_forward=callback,
        )
    arrays = pack_centered_entries(entries, "Selected_Refined_Centered")
    assert arrays["refine_status"].tolist() == [0, 1, 2, 3]
    assert arrays["voxel_offsets"].tolist() == [0, 1, 1, 1, 1]
    assert arrays["feature_entry_index"].tolist() == [0]
    assert arrays["voxel_ds_2"].shape == (1, 256, 20, 20, 20)
    assert arrays["voxel_ds_3"].shape == (1, 256, 10, 10, 10)
    assert arrays["voxel_ds_4"].shape == (1, 256, 5, 5, 5)
    assert arrays["voxel_c4"].shape == (1, 256, 5, 5, 5)
    success_voxels = arrays["voxel_index_local_zyx"]
    assert success_voxels.tolist() == [[10, 10, 10]]
    assert arrays["source_node_id"].tolist() == [0, 1, 2, 3]
    assert "pdb_id=sample_002" in caplog.text
    assert "source_node_id=3" in caplog.text


def test_find_centered_masks_probability_and_exports_only_10A_core_atoms() -> None:
    oldest = _node(0, [(10, 10, 10), (10, 10, 11)])
    child = _node(1, [(10, 10, 10)], threshold=0.75)
    child.parent = oldest
    oldest.children.append(child)
    clg = CLG(0, 0, seed_node=child, oldest_node=oldest, candidate_nodes=(child, oldest))
    geometry = CenteredGeometry(
        full_shape_zyx=FULL_SHAPE,
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )

    def callback(request):
        del request
        probability = np.full(FULL_SHAPE, 0.8, dtype=np.float32)
        payload = _payload(probability)
        payload["hardmask"][10, 10, 10] = True
        payload.update(
            {
                "P_coord_local_xyz": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                "P_probability": np.asarray([0.7], dtype=np.float32),
                "P_feat_L2": np.ones((1, 2), dtype=np.float16),
                "P_feat_L3": np.ones((1, 3), dtype=np.float16),
                "P_feat_L4": np.ones((1, 4), dtype=np.float16),
                "A_global_index": np.asarray([5, 6, 7], dtype=np.int64),
                "A_coord_local_xyz": np.asarray(
                    [[10.5, 10.5, 10.5], [70.5, 70.5, 70.5], [81.0, 10.0, 10.0]],
                    dtype=np.float32,
                ),
                "A_coord_centered_world": np.zeros((3, 3), dtype=np.float32),
                "A_probability": np.asarray([0.9, 0.1, 0.2], dtype=np.float32),
                "A_feat_L1": np.ones((3, 2), dtype=np.float16),
                "A_feat_L2": np.ones((3, 3), dtype=np.float16),
                "A_feat_L3": np.ones((3, 4), dtype=np.float16),
                "A_feat_L4": np.ones((3, 5), dtype=np.float16),
            }
        )
        return payload

    entries = produce_clg_centered_entries(
        clgs=(clg,),
        geometry=geometry,
        stage1_model_name="Find_1",
        split="calibration",
        pdb_id="sample_003",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        full_forward=callback,
    )
    arrays = pack_centered_entries(entries, "CLG_centered")
    assert arrays["A_global_index"].tolist() == [5]
    assert arrays["A_offsets"].tolist() == [0, 1]
    assert arrays["candidate_A_offsets"].tolist() == [0, 1, 2]
    assert arrays["candidate_A_index"].tolist() == [0, 0]
    assert arrays["centered_probability"].tolist() == pytest.approx([0.0, 0.8])
    assert "A_feat_L0" not in arrays
    receptor_raw49 = np.arange(8 * 49, dtype=np.float32).reshape(8, 49)
    restored_l0 = recover_a_feat_l0(
        receptor_raw49,
        arrays["A_global_index"],
    )
    assert restored_l0.shape == (1, 49)
    np.testing.assert_array_equal(restored_l0, receptor_raw49[[5]])


def test_centered_start_delegates_to_shared_centroid_resolver() -> None:
    assert resolve_component_centered_start(
        centroid_zyx=np.asarray([119.0, 129.0, 139.0]),
        full_shape_zyx=(120, 130, 140),
    ) == (40, 50, 60)


def test_formal_centered_adapter_reads_named_unet_and_find_outputs() -> None:
    shape = (2, 2, 2)
    voxel_features = {
        "voxel_final": np.ones((1, 48, *shape), dtype=np.float32),
        "voxel_ds_2": np.ones((1, 256, 1, 1, 1), dtype=np.float32),
        "voxel_ds_3": np.ones((1, 256, 1, 1, 1), dtype=np.float32),
        "voxel_ds_4": np.ones((1, 256, 1, 1, 1), dtype=np.float32),
        "voxel_c4": np.ones((1, 256, 1, 1, 1), dtype=np.float32),
    }
    batch = {
        "hardmask": np.zeros((1, *shape), dtype=np.bool_),
        "box_shape_zyx": np.asarray([[2, 2, 2]], dtype=np.int64),
        "voxel_size_world": np.asarray([[1.5, 1.0, 0.5]], dtype=np.float32),
    }
    common = {
        "voxel_logits_ligand": np.zeros((1, 1, *shape), dtype=np.float32),
        "voxel_logits_aux": np.zeros((1, 1, *shape), dtype=np.float32),
        "voxel_features": voxel_features,
    }
    unet = adapt_stage1_centered_output(common, batch, "unet_c1")
    assert unet["ligand_probability"].shape == shape
    assert np.all(unet["ligand_probability"] == 0.5)
    assert "A_global_index" not in unet and "P_probability" not in unet

    find_output = {
        **common,
        "atom_coord_local_voxel": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "atom_global_indices": np.asarray([17], dtype=np.int64),
        "atom_logits": np.asarray([[0.0]], dtype=np.float32),
        "A_feat_L1": np.ones((1, 2), dtype=np.float32),
        "A_feat_L2": np.ones((1, 3), dtype=np.float32),
        "A_feat_L3": np.ones((1, 4), dtype=np.float32),
        "A_feat_L4": np.ones((1, 5), dtype=np.float32),
        "anchor_coord_local_voxel": np.asarray([[0.5, 1.5, 0.5]], dtype=np.float32),
        "pseudo_logits": np.asarray([[0.0]], dtype=np.float32),
        "P_feat_L2": np.ones((1, 3), dtype=np.float32),
        "P_feat_L3": np.ones((1, 4), dtype=np.float32),
        "P_feat_L4": np.ones((1, 5), dtype=np.float32),
    }
    find = adapt_stage1_centered_output(find_output, batch, "Find_1")
    assert find["A_global_index"].tolist() == [17]
    assert find["A_probability"].tolist() == pytest.approx([0.5])
    assert find["P_probability"].tolist() == pytest.approx([0.5])
    assert find["A_coord_centered_world"].tolist() == [[0.0, 0.0, 0.0]]


def test_formal_centered_adapter_bridges_bfloat16_features_to_numpy() -> None:
    """验证 BF16 autocast 特征可进入 centered adapter，并保持后续可压缩的数值。"""

    shape = (2, 2, 2)
    voxel_features = {
        "voxel_final": torch.ones((1, 48, *shape), dtype=torch.bfloat16),
        "voxel_ds_2": torch.ones((1, 256, 1, 1, 1), dtype=torch.bfloat16),
        "voxel_ds_3": torch.ones((1, 256, 1, 1, 1), dtype=torch.bfloat16),
        "voxel_ds_4": torch.ones((1, 256, 1, 1, 1), dtype=torch.bfloat16),
        "voxel_c4": torch.ones((1, 256, 1, 1, 1), dtype=torch.bfloat16),
    }
    output = {
        "voxel_logits_ligand": torch.zeros((1, 1, *shape), dtype=torch.bfloat16),
        "voxel_logits_aux": torch.zeros((1, 1, *shape), dtype=torch.bfloat16),
        "voxel_features": voxel_features,
    }
    batch = {
        "hardmask": torch.zeros((1, *shape), dtype=torch.bool),
        "box_shape_zyx": torch.tensor([[2, 2, 2]], dtype=torch.long),
        "voxel_size_world": torch.ones((1, 3), dtype=torch.float32),
    }

    payload = adapt_stage1_centered_output(output, batch, "unet_c1")

    assert payload["voxel_final_grid"].dtype == np.float32
    assert payload["voxel_final_grid"].shape == (48, *shape)
    np.testing.assert_array_equal(payload["voxel_final_grid"], 1.0)
