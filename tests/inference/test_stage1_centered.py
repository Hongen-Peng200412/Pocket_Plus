"""Stage1 centered 产物的批量前向、权威体素、成员关系与 Selected 状态测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.artifacts.io import pack_centered_entries
from src.component_lineage.structures import CLG, ComponentNode
from src.datasets.stage1_requests import centered_start_from_centroid_zyx
from src.inference.centered import (
    CenteredGeometry,
    CenteredRequest,
    iter_model_centered_payloads,
    iter_stage1_centered_batch_payloads,
    produce_clg_centered_entries,
    produce_f1_centered_entries,
    produce_selected_refined_entries,
)


FULL_SHAPE = (80, 80, 80)


def _node(
    node_id: int,
    voxels: list[tuple[int, int, int]],
    threshold: float = 0.5,
) -> ComponentNode:
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
        (48, *FULL_SHAPE),
    )
    return {
        "ligand_probability": probability,
        "hardmask": np.zeros(FULL_SHAPE, dtype=np.bool_),
        "voxel_aux_probability_grid": np.full(FULL_SHAPE, 0.25, dtype=np.float32),
        "voxel_final_grid": voxel_final,
    }


def _logits(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float32)
    result = np.empty_like(values)
    result[values <= 0.0] = -80.0
    result[values >= 1.0] = 80.0
    middle = (values > 0.0) & (values < 1.0)
    result[middle] = np.log(values[middle] / (1.0 - values[middle]))
    return result


class _FakeCenteredRuntime:
    """把逐请求测试 fixture 变成正式批量 batch 和完整 wrapper 输出。"""

    def __init__(self, payload_provider):
        self.payload_provider = payload_provider
        self.batch_sizes: list[int] = []

    def batch_builder(self, requests):
        payloads = [self.payload_provider(request) for request in requests]
        batch = {
            "requests": tuple(requests),
            "payloads": payloads,
            "hardmask": np.stack([payload["hardmask"] for payload in payloads]),
            "box_shape_zyx": np.asarray([FULL_SHAPE] * len(payloads), dtype=np.int64),
            "voxel_size_world": np.ones((len(payloads), 3), dtype=np.float32),
        }
        if "A_global_index" in payloads[0]:
            batch["atom_counts"] = np.asarray(
                [payload["A_global_index"].shape[0] for payload in payloads],
                dtype=np.int64,
            )
            batch["atom_global_indices"] = np.concatenate(
                [payload["A_global_index"] for payload in payloads]
            )
            batch["atom_feat"] = np.concatenate(
                [payload["A_feat_L0"] for payload in payloads]
            )
        return batch

    def wrapper(self, batch):
        payloads = batch["payloads"]
        self.batch_sizes.append(len(payloads))
        output = {
            "voxel_logits_ligand": np.stack(
                [_logits(payload["ligand_probability"])[None] for payload in payloads]
            ),
            "voxel_logits_aux": np.stack(
                [
                    _logits(payload["voxel_aux_probability_grid"])[None]
                    for payload in payloads
                ]
            ),
            "voxel_features": {
                "voxel_final": np.stack(
                    [payload["voxel_final_grid"] for payload in payloads]
                ),
            },
        }
        if "A_global_index" not in payloads[0]:
            return output

        atom_counts = np.asarray(
            [payload["A_global_index"].shape[0] for payload in payloads], dtype=np.int64
        )
        anchor_counts = [payload["P_probability"].shape[0] for payload in payloads]
        output.update(
            {
                "atom_counts": atom_counts,
                "atom_global_indices": np.concatenate(
                    [payload["A_global_index"] for payload in payloads]
                ),
                "atom_coord_local_voxel": np.concatenate(
                    [payload["A_coord_local_xyz"] for payload in payloads]
                ),
                "atom_logits": np.concatenate(
                    [_logits(payload["A_probability"]) for payload in payloads]
                ),
                "anchor_batch_index": np.repeat(
                    np.arange(len(payloads), dtype=np.int64), anchor_counts
                ),
                "anchor_coord_local_voxel": np.concatenate(
                    [payload["P_coord_local_xyz"] for payload in payloads]
                ),
                "pseudo_logits": np.concatenate(
                    [_logits(payload["P_probability"]) for payload in payloads]
                ),
            }
        )
        for level in range(1, 4):
            output[f"A_feat_L{level}"] = np.concatenate(
                [payload[f"A_feat_L{level}"] for payload in payloads]
            )
        for level in range(2, 4):
            output[f"P_feat_L{level}"] = np.concatenate(
                [payload[f"P_feat_L{level}"] for payload in payloads]
            )
        return output


def test_f1_and_clg_keep_global_source_identity_and_memberships() -> None:
    oldest = _node(0, [(10, 10, 10), (10, 10, 11)])
    child = _node(1, [(10, 10, 10)], threshold=0.75)
    child.parent = oldest
    oldest.children.append(child)
    clg = CLG(0, 0, seed_node=child, oldest_node=oldest, candidate_nodes=(child, oldest))
    geometry = CenteredGeometry(FULL_SHAPE, np.zeros(3), np.ones(3))
    requests = []

    def payload_provider(request):
        requests.append(request)
        probability = np.zeros(FULL_SHAPE, dtype=np.float32)
        probability[10, 10, 10:12] = 0.8
        return _payload(probability)

    runtime = _FakeCenteredRuntime(payload_provider)
    common = dict(
        geometry=geometry,
        stage1_model_name="unet_c1",
        split="train",
        pdb_id="sample_001",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        wrapper=runtime.wrapper,
        batch_builder=runtime.batch_builder,
        centered_batch_size=12,
    )
    f1_entries = produce_f1_centered_entries(
        nodes=(oldest,),
        f1_threshold_grid_index=oldest.threshold_grid_index,
        **common,
    )
    f1_arrays = pack_centered_entries(f1_entries, "F1_centered")
    assert f1_arrays["source_node_id"].tolist() == [0]
    assert f1_arrays["voxel_offsets"].tolist() == [0, 2]
    assert np.allclose(f1_arrays["centered_probability"], 0.8)

    clg_entries = produce_clg_centered_entries(clgs=(clg,), **common)
    clg_arrays = pack_centered_entries(clg_entries, "CLG_centered")
    assert clg_arrays["candidate_offsets"].tolist() == [0, 2]
    assert clg_arrays["candidate_voxel_offsets"].tolist() == [0, 1, 3]
    assert clg_arrays["candidate_voxel_index"].tolist() == [0, 0, 1]
    assert {(item.stage1_model_name, item.split, item.pdb_id) for item in requests} == {
        ("unet_c1", "train", "sample_001")
    }


def test_selected_status_codes_and_refined_blob_semantics() -> None:
    nodes = [
        _node(0, [(10, 10, 10), (10, 10, 11), (10, 10, 12)]),
        _node(1, [(20, 20, 20)]),
        _node(2, [(30, 30, 30)]),
    ]

    def payload_provider(request):
        probability = np.zeros(FULL_SHAPE, dtype=np.float32)
        if request.source_node_id == 0:
            probability[10, 10, 10] = 0.9
            probability[10, 10, 12] = 0.9
            probability[70, 70, 70] = 0.9
        elif request.source_node_id == 2:
            probability[0, 0, 0] = 0.9
        return _payload(probability)

    runtime = _FakeCenteredRuntime(payload_provider)
    entries = produce_selected_refined_entries(
        selected_nodes=nodes,
        geometry=CenteredGeometry(FULL_SHAPE, np.zeros(3), np.ones(3)),
        stage1_model_name="unet_c1",
        split="validation",
        pdb_id="sample_002",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        wrapper=runtime.wrapper,
        batch_builder=runtime.batch_builder,
        centered_batch_size=12,
    )
    arrays = pack_centered_entries(entries, "Selected_Refined_Centered")
    assert arrays["refine_status"].tolist() == [0, 1, 2]
    assert arrays["voxel_offsets"].tolist() == [0, 1, 1, 1]
    assert "feature_entry_index" not in arrays
    assert arrays["voxel_index_local_zyx"].tolist() == [[10, 10, 10]]


def test_selected_forward_error_propagates() -> None:
    def payload_provider(request):
        raise RuntimeError(f"模拟 forward 失败: {request.source_node_id}")

    runtime = _FakeCenteredRuntime(payload_provider)
    with pytest.raises(RuntimeError, match="模拟 forward 失败"):
        produce_selected_refined_entries(
            selected_nodes=(_node(0, [(10, 10, 10)]),),
            geometry=CenteredGeometry(FULL_SHAPE, np.zeros(3), np.ones(3)),
            stage1_model_name="unet_c1",
            split="validation",
            pdb_id="sample_failed",
            resolve_box_start=lambda centroid, shape: (0, 0, 0),
            wrapper=runtime.wrapper,
            batch_builder=runtime.batch_builder,
            centered_batch_size=12,
        )


def test_find_centered_preserves_probability_and_exports_only_10a_core_atoms() -> None:
    oldest = _node(0, [(10, 10, 10), (10, 10, 11)])
    child = _node(1, [(10, 10, 10)], threshold=0.75)
    child.parent = oldest
    oldest.children.append(child)
    clg = CLG(0, 0, seed_node=child, oldest_node=oldest, candidate_nodes=(child, oldest))

    def payload_provider(request):
        del request
        payload = _payload(np.full(FULL_SHAPE, 0.8, dtype=np.float32))
        payload["hardmask"][10, 10, 10] = True
        payload.update(
            {
                "P_coord_local_xyz": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                "P_probability": np.asarray([0.7], dtype=np.float32),
                "P_feat_L2": np.ones((1, 2), dtype=np.float16),
                "P_feat_L3": np.ones((1, 3), dtype=np.float16),
                "A_global_index": np.asarray([5, 6, 7], dtype=np.int64),
                "A_coord_local_xyz": np.asarray(
                    [[10.5, 10.5, 10.5], [70.5, 70.5, 70.5], [81.0, 10.0, 10.0]],
                    dtype=np.float32,
                ),
                "A_coord_centered_world": np.zeros((3, 3), dtype=np.float32),
                "A_probability": np.asarray([0.9, 0.1, 0.2], dtype=np.float32),
                "A_feat_L0": np.arange(3 * 49, dtype=np.float32).reshape(3, 49),
                "A_feat_L1": np.ones((3, 2), dtype=np.float16),
                "A_feat_L2": np.ones((3, 3), dtype=np.float16),
                "A_feat_L3": np.ones((3, 4), dtype=np.float16),
            }
        )
        return payload

    runtime = _FakeCenteredRuntime(payload_provider)
    entries = produce_clg_centered_entries(
        clgs=(clg,),
        geometry=CenteredGeometry(FULL_SHAPE, np.zeros(3), np.ones(3)),
        stage1_model_name="Find_1",
        split="calibration",
        pdb_id="sample_003",
        resolve_box_start=lambda centroid, shape: (0, 0, 0),
        wrapper=runtime.wrapper,
        batch_builder=runtime.batch_builder,
        centered_batch_size=12,
    )
    arrays = pack_centered_entries(entries, "CLG_centered")
    assert arrays["A_global_index"].tolist() == [5]
    assert arrays["candidate_A_offsets"].tolist() == [0, 1, 2]
    assert arrays["centered_probability"].tolist() == pytest.approx([0.8, 0.8])
    assert arrays["A_feat_L0"].dtype == np.float32
    np.testing.assert_array_equal(
        arrays["A_feat_L0"],
        np.arange(3 * 49, dtype=np.float32).reshape(3, 49)[[0]],
    )


def test_centered_start_delegates_to_shared_centroid_resolver() -> None:
    assert centered_start_from_centroid_zyx(
        centroid_zyx=np.asarray([119.0, 129.0, 139.0]),
        full_shape_zyx=(120, 130, 140),
    ) == (40, 50, 60)


def test_centered_batch_split_uses_dense_batch_and_ragged_a_p_ownership() -> None:
    shape = (2, 2, 2)
    batch = {
        "hardmask": np.zeros((2, *shape), dtype=np.bool_),
        "box_shape_zyx": np.asarray([shape, shape], dtype=np.int64),
        "voxel_size_world": np.ones((2, 3), dtype=np.float32),
        "atom_counts": np.asarray([2, 3], dtype=np.int64),
        "atom_global_indices": np.asarray([11, 10, 21, 22, 20], dtype=np.int64),
        "atom_feat": np.repeat(
            np.asarray([[11.0], [10.0], [21.0], [22.0], [20.0]], dtype=np.float32),
            49,
            axis=1,
        ),
    }
    features = {"voxel_final": np.ones((2, 4, 1, 1, 1), dtype=np.float32)}
    output = {
        "voxel_logits_ligand": np.zeros((2, 1, *shape), dtype=np.float32),
        "voxel_logits_aux": np.zeros((2, 1, *shape), dtype=np.float32),
        "voxel_features": features,
        "atom_counts": np.asarray([1, 2]),
        "atom_global_indices": np.asarray([10, 20, 21]),
        "atom_coord_local_voxel": np.ones((3, 3), dtype=np.float32),
        "atom_logits": np.zeros(3, dtype=np.float32),
        "anchor_batch_index": np.asarray([1, 0, 1]),
        "anchor_coord_local_voxel": np.ones((3, 3), dtype=np.float32),
        "pseudo_logits": np.zeros(3, dtype=np.float32),
        **{f"A_feat_L{level}": np.ones((3, level), dtype=np.float32) for level in range(1, 5)},
        **{f"P_feat_L{level}": np.ones((3, level), dtype=np.float32) for level in range(2, 5)},
    }
    payloads = list(iter_stage1_centered_batch_payloads(output, batch, "Find_1"))
    assert [payload["A_global_index"].tolist() for payload in payloads] == [[10], [20, 21]]
    assert [payload["A_feat_L0"][:, 0].tolist() for payload in payloads] == [[10.0], [20.0, 21.0]]
    assert all(payload["A_feat_L0"].dtype == np.float32 for payload in payloads)
    assert [payload["P_probability"].shape[0] for payload in payloads] == [1, 2]
    assert all("A_feat_L4" not in payload and "P_feat_L4" not in payload for payload in payloads)


def test_centered_scheduler_uses_default_scale_batches() -> None:
    requests = [
        CenteredRequest("unet_c1", "train", "pdb", "F1_centered", i, (0, 0, 0), 0, i, 1)
        for i in range(13)
    ]
    batch_sizes = []

    def batch_builder(items):
        return {
            "count": len(items),
            "hardmask": np.zeros((len(items), 2, 2, 2), dtype=np.bool_),
        }

    def wrapper(batch):
        size = batch["count"]
        batch_sizes.append(size)
        features = {"voxel_final": np.ones((size, 1, 1, 1, 1), dtype=np.float32)}
        return {
            "voxel_logits_ligand": np.zeros((size, 1, 2, 2, 2), dtype=np.float32),
            "voxel_logits_aux": np.zeros((size, 1, 2, 2, 2), dtype=np.float32),
            "voxel_features": features,
        }

    payloads = list(iter_model_centered_payloads(requests, wrapper, batch_builder, 12))
    assert len(payloads) == 13
    assert batch_sizes == [12, 1]


def test_centered_batch_split_bridges_bfloat16_features_to_numpy() -> None:
    shape = (2, 2, 2)
    output = {
        "voxel_logits_ligand": torch.zeros((1, 1, *shape), dtype=torch.bfloat16),
        "voxel_logits_aux": torch.zeros((1, 1, *shape), dtype=torch.bfloat16),
        "voxel_features": {
            "voxel_final": torch.ones((1, 1, 1, 1, 1), dtype=torch.bfloat16)
        },
    }
    batch = {
        "hardmask": torch.zeros((1, *shape), dtype=torch.bool),
        "box_shape_zyx": torch.tensor([shape]),
        "voxel_size_world": torch.ones((1, 3)),
    }
    payload = next(iter_stage1_centered_batch_payloads(output, batch, "unet_c1"))
    assert payload["voxel_final_grid"].dtype == np.float32
