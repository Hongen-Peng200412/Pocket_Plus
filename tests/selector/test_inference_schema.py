"""Selector selection.npz 的门控、精确 MAP 与字段契约测试. """

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import src.selector.inference as selector_inference_module
from src.component_lineage import ComponentForest, ComponentNode, ComponentTree
from src.selector.inference import load_selected_nodes_for_pdb, produce_selection_for_pdb


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """保存一个测试 NPZ; 测试数据无需模拟正式原子发布. """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def test_selection_schema_uses_gate_and_clg_local_candidate_indices(tmp_path: Path) -> None:
    """门控通过时取非空精确 MAP, 失败时写空段且不混用 forest node ID. """
    scores_path = tmp_path / "scores.npz"
    forest_path = tmp_path / "forest.npz"
    clg_path = tmp_path / "clg.npz"
    selection_path = tmp_path / "selection.npz"
    _save_npz(
        scores_path,
        CLG_id=np.asarray([5, 6], dtype=np.int32),
        CLG_logit=np.asarray([2.0, -2.0], dtype=np.float32),
        CLG_valid_probability=np.asarray([0.9, 0.2], dtype=np.float32),
        candidate_offsets=np.asarray([0, 3, 4], dtype=np.int64),
        predicted_max_iou=np.asarray([0.2, 0.8, 0.7, 0.9], dtype=np.float32),
        selection_logit=np.asarray([0.1, 2.0, 1.5, 5.0], dtype=np.float32),
    )
    def node(tree_id: int, node_id: int, values: list[int]) -> ComponentNode:
        """构造用于 schema/恢复测试的最小合法组件节点. """
        return ComponentNode(
            tree_id=tree_id,
            node_id=node_id,
            threshold_grid_index=100,
            threshold_value=100 / 32768,
            voxel_global_linear_index=np.asarray(values, dtype=np.int64),
            bbox_min_zyx=np.asarray([0, 0, 0], dtype=np.int32),
            bbox_max_zyx=np.asarray([1, 1, 1], dtype=np.int32),
            centroid_zyx=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
            probability_mean=0.5,
            probability_max=0.8,
            candidate_eligible=True,
            ineligible_reason_code=0,
        )

    root = node(0, 10, [0, 1, 2])
    left = node(0, 11, [0])
    right = node(0, 12, [1])
    left.parent = root
    right.parent = root
    root.children.extend([left, right])
    other = node(1, 20, [5])
    forest_arrays = ComponentForest(
        (ComponentTree(0, (root, left, right)), ComponentTree(1, (other,)))
    ).to_arrays()
    _save_npz(forest_path, **forest_arrays)
    _save_npz(
        clg_path,
        CLG_id=np.asarray([5, 6], dtype=np.int32),
        tree_id=np.asarray([0, 1], dtype=np.int32),
        candidate_offsets=np.asarray([0, 3, 4], dtype=np.int64),
        candidate_node_id=np.asarray([10, 11, 12, 20], dtype=np.int32),
    )

    produce_selection_for_pdb(
        scores_path=scores_path,
        forest_path=forest_path,
        clg_path=clg_path,
        selection_path=selection_path,
        tau_g=0.5,
        lambda_count=0.05,
    )

    with np.load(selection_path, allow_pickle=False) as selection:
        assert set(selection.files) == {
            "CLG_id",
            "CLG_gate_pass",
            "selected_candidate_offsets",
            "selected_candidate_index",
        }
        np.testing.assert_array_equal(selection["CLG_id"], [5, 6])
        np.testing.assert_array_equal(selection["CLG_gate_pass"], [True, False])
        np.testing.assert_array_equal(selection["selected_candidate_offsets"], [0, 2, 2])
        # 11 与 12 是 sibling, 可同时进入反链; 保存的是本 CLG 局部下标 1/2. 
        np.testing.assert_array_equal(selection["selected_candidate_index"], [1, 2])
        assert selection["selected_candidate_index"].dtype == np.int16

    selected_nodes = load_selected_nodes_for_pdb(selection_path, forest_path, clg_path)
    assert [(value.tree_id, value.node_id) for value in selected_nodes] == [(0, 11), (0, 12)]


def test_selected_node_loader_merges_same_node_across_clgs_in_first_order(
    tmp_path: Path,
) -> None:
    """同一 forest node 可属于多个 CLG; 下游只接收首次出现的一份. """

    node = ComponentNode(
        tree_id=0,
        node_id=7,
        threshold_grid_index=100,
        threshold_value=100 / 32768,
        voxel_global_linear_index=np.asarray([0], dtype=np.int64),
        bbox_min_zyx=np.asarray([0, 0, 0], dtype=np.int32),
        bbox_max_zyx=np.asarray([0, 0, 0], dtype=np.int32),
        centroid_zyx=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        probability_mean=0.8,
        probability_max=0.8,
        candidate_eligible=True,
        ineligible_reason_code=0,
    )
    forest_path = tmp_path / "forest.npz"
    clg_path = tmp_path / "clg.npz"
    selection_path = tmp_path / "selection.npz"
    _save_npz(
        forest_path,
        **ComponentForest((ComponentTree(0, (node,)),)).to_arrays(),
    )
    _save_npz(
        clg_path,
        CLG_id=np.asarray([10, 11], dtype=np.int32),
        tree_id=np.asarray([0, 0], dtype=np.int32),
        candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        candidate_node_id=np.asarray([7, 7], dtype=np.int32),
    )
    _save_npz(
        selection_path,
        CLG_id=np.asarray([10, 11], dtype=np.int32),
        CLG_gate_pass=np.asarray([True, True], dtype=np.bool_),
        selected_candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        selected_candidate_index=np.asarray([0, 0], dtype=np.int16),
    )

    selected = load_selected_nodes_for_pdb(selection_path, forest_path, clg_path)

    assert [(value.tree_id, value.node_id) for value in selected] == [(0, 7)]


def test_produce_scores_publishes_empty_archive_for_zero_clg_pdb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """冻结 inventory 中的零 CLG PDB 也必须得到可消费的空 scores.npz. """

    stage1_root = tmp_path / "stage1"
    clg_path = stage1_root / "unet_c1" / "calibration" / "empty" / "components" / "clg.npz"
    _save_npz(
        clg_path,
        CLG_id=np.empty(0, dtype=np.int32),
        candidate_offsets=np.zeros(1, dtype=np.int64),
    )
    frozen_path = tmp_path / "input_CLG_list.json"
    frozen_path.write_text(
        json.dumps(
            {
                "stage1_model_name": "unet_c1",
                "split_order": ["calibration"],
                "split_counts": {"calibration": 0},
                "split_pdb_counts": {"calibration": 1},
                "pdb_ids_by_split": {"calibration": ["empty"]},
                "items": [],
            }
        ),
        encoding="utf-8",
    )

    class _EmptyDataset:
        """绕过模型输入, 只验证零 CLG PDB 的发布边界. """

        def __init__(self, **_kwargs) -> None:
            pass

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(selector_inference_module, "SelectorDataset", _EmptyDataset)
    monkeypatch.setattr(
        selector_inference_module,
        "load_selector_checkpoint",
        lambda *_args, **_kwargs: (
            SimpleNamespace(model=None),
            {
                "resolved_config": {
                    "stage1_model_name": "unet_c1",
                    "data": {
                        "lambda_count": 0.05,
                        "density_clip_percentile": [0.001, 0.999],
                        "pdb_cache_size": 1,
                    },
                }
            },
        ),
    )

    paths = selector_inference_module.produce_scores(
        checkpoint_path=tmp_path / "unused.ckpt",
        input_clg_list_path=frozen_path,
        stage1_outputs_root=stage1_root,
        upstream_root=tmp_path / "upstream",
        selector_run_dir=tmp_path / "selector_run",
        split="calibration",
        device_name="cpu",
    )

    assert len(paths) == 1
    with np.load(paths[0], allow_pickle=False) as scores:
        assert scores["CLG_id"].shape == (0,)
        assert scores["candidate_offsets"].tolist() == [0]
        assert scores["predicted_max_iou"].shape == (0,)
        assert scores["selection_logit"].shape == (0,)
