from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np

from src.inference.baseline.emap2lig_find import (
    EVALUATION_NAME,
    _aggregate_split,
    _evaluate_one_pdb,
    _load_evaluation,
)


def _write_mrc(path: Path, grid: np.ndarray) -> None:
    """写出标准轴序、单位体素和零原点的测试 MRC。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, data=np.asarray(grid, dtype=np.float32), overwrite=True) as archive:
        archive.voxel_size = (1.0, 1.0, 1.0)
        archive.header.origin = (0.0, 0.0, 0.0)


def test_emap2lig_adapter_preserves_official_instances_and_metrics(tmp_path: Path) -> None:
    """适配器保留官方实例身份，并发布现有 Stage1 聚合字段。"""

    pdb_id = "demo"
    # probability: float32 (3,3,3)，两个官方实例分别取得 0.9 与 0.8 分数。
    probability = np.zeros((3, 3, 3), dtype=np.float32)
    probability[1, 1, 1] = 0.9
    probability[1, 1, 2] = 0.8
    # first_mask/second_mask: float32 (3,3,3)，两个保持独立身份的官方实例。
    first_mask = np.zeros_like(probability)
    second_mask = np.zeros_like(probability)
    first_mask[1, 1, 1] = 1.0
    second_mask[1, 1, 2] = 1.0

    raw_root = tmp_path / "held_out_test_0" / "raw_find_outputs"
    _write_mrc(raw_root / pdb_id / "find_maps" / "ligand.mrc", probability)
    _write_mrc(raw_root / pdb_id / "find_blobs" / "mask_1.mrc", first_mask)
    _write_mrc(raw_root / pdb_id / "find_blobs" / "mask_2.mrc", second_mask)

    density_root = tmp_path / "data" / "density" / pdb_id
    density_root.mkdir(parents=True)
    np.savez_compressed(
        density_root / "ligand_area.npz",
        grid_shape_zyx=np.asarray([3, 3, 3], dtype=np.int64),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
        origin_xyz=np.zeros(3, dtype=np.float32),
        mask_0=np.asarray([[1, 1, 1]], dtype=np.int32),
        mask_1=np.asarray([[1, 1, 2]], dtype=np.int32),
    )
    # union_mask: bool (1,3,3,3)，与两个真实 occurrence 对齐。
    union_mask = np.zeros((1, 3, 3, 3), dtype=np.bool_)
    union_mask[0, 1, 1, 1:3] = True
    np.save(density_root / "union_mask.npy", union_mask)

    split_root = tmp_path / "held_out_test_0"
    assert (
        _evaluate_one_pdb(
            pdb_id,
            str(raw_root),
            str(tmp_path / "data"),
            str(split_root),
            2,
        )
        == pdb_id
    )
    evaluation, histogram = _load_evaluation(
        split_root / "evaluation" / "per_pdb" / f"{pdb_id}.npz",
        pdb_id,
    )
    assert evaluation.source_blob_index.tolist() == [1, 2]
    assert np.allclose(evaluation.candidate_score, [0.9, 0.8])
    assert evaluation.intersections.tolist() == [[1, 0], [0, 1]]
    assert histogram.shape == (2, 1024)

    metrics_path, jsonl_path = _aggregate_split(
        [pdb_id],
        split_root,
        split_root,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["semantic_micro_f1"] == 1.0
    assert metrics["coverage_micro_f1_0p3"] == 1.0
    assert metrics["one_to_one_macro_recall_0p6"] == 1.0
    assert metrics["top3_success_count_0p5"] == 1
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["pdb_id"] == pdb_id
    assert metrics_path.name == f"{EVALUATION_NAME}.metrics.json"

