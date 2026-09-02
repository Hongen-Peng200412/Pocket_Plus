# -*- coding: utf-8 -*-
"""验证 Stage1 调参外层并发与串行科学结果完全一致."""

from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np
from omegaconf import OmegaConf
import pytest

import src.inference.calibration as calibration_module
import src.inference.pipeline as pipeline_module
from src.inference.artifacts import Stage1ArtifactPaths, publish_stage1_artifact
from src.inference.calibration import tune_centered_selection
from src.inference.pipeline import run_tune_stage


def _build_parallel_tuning_case() -> tuple[
    tuple[tuple[str, dict[str, np.ndarray]], ...],
    dict[str, tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]],
]:
    """构造同时适用于 basic 与 Gaussian 的两个 PDB 校准事实."""
    first = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.85, 0.35], dtype=np.float32),
        "source_voxel_count": np.asarray([2, 1], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[0, 0, 0], [0, 0, 1], [2, 2, 2]],
            dtype=np.int16,
        ),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
        "voxel_size_world": np.ones((2, 3), dtype=np.float32),
        "A_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "A_coord_local_xyz": np.asarray(
            [[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]],
            dtype=np.float32,
        ),
        "A_probability": np.asarray([0.9, 0.1], dtype=np.float32),
    }
    second = {
        "source_blob_index": np.asarray([0], dtype=np.int32),
        "source_probability_mean": np.asarray([0.65], dtype=np.float32),
        "source_voxel_count": np.asarray([2], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[1, 1, 1], [1, 1, 2]],
            dtype=np.int16,
        ),
        "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
        "voxel_size_world": np.ones((1, 3), dtype=np.float32),
        "A_offsets": np.asarray([0, 2], dtype=np.int64),
        "A_coord_local_xyz": np.asarray(
            [[1.5, 1.5, 1.5], [2.5, 1.5, 1.5]],
            dtype=np.float32,
        ),
        "A_probability": np.asarray([0.75, 0.25], dtype=np.float32),
    }
    ground_truth = {
        "first": (
            np.asarray([11], dtype=np.int32),
            (np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int32),),
            (4, 4, 4),
        ),
        "second": (
            np.asarray([21], dtype=np.int32),
            (np.asarray([[1, 1, 1], [1, 1, 2]], dtype=np.int32),),
            (4, 4, 4),
        ),
    }
    return (("first", first), ("second", second)), ground_truth


def _build_imbalanced_macro_tuning_case() -> tuple[
    tuple[tuple[str, dict[str, np.ndarray]], ...],
    dict[str, tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]],
]:
    """构造一个大 PDB 与一个小 PDB 对最佳 basic 阈值意见相反的事实.

    large 的高分候选命中 100 个真实体素, 低分候选增加 900 个假阳性体素;
    small 只有一个低分真阳性候选. 返回的候选同时含空 A 表, 因此 basic 与
    Gaussian 可以在完全相同的候选分数上比较 macro 目标.
    """

    # int16, (100, 3), large 高分候选与唯一真实 occurrence 完全重合的 ZYX 体素.
    large_true = np.asarray(
        [[0, 0, index] for index in range(100)],
        dtype=np.int16,
    )
    # int16, (900, 3), large 低分候选中全部位于真实 occurrence 外的 ZYX 体素.
    large_false = np.asarray(
        [[0, 0, index] for index in range(100, 1000)],
        dtype=np.int16,
    )
    # large 含一个 0.9 真阳性候选和一个 0.8 假阳性候选.
    large = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_voxel_count": np.asarray([100, 900], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 100, 1000], dtype=np.int64),
        "voxel_index_local_zyx": np.concatenate((large_true, large_false)),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
        "voxel_size_world": np.ones((2, 3), dtype=np.float32),
        "A_offsets": np.zeros(3, dtype=np.int64),
        "A_coord_local_xyz": np.empty((0, 3), dtype=np.float32),
        "A_probability": np.empty(0, dtype=np.float32),
    }
    # small 含一个 0.8 真阳性候选, 对低阈值方案投出与 large 等权的一票.
    small = {
        "source_blob_index": np.asarray([0], dtype=np.int32),
        "source_probability_mean": np.asarray([0.8], dtype=np.float32),
        "source_voxel_count": np.asarray([1], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray([[0, 0, 0]], dtype=np.int16),
        "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
        "voxel_size_world": np.ones((1, 3), dtype=np.float32),
        "A_offsets": np.zeros(2, dtype=np.int64),
        "A_coord_local_xyz": np.empty((0, 3), dtype=np.float32),
        "A_probability": np.empty(0, dtype=np.float32),
    }
    # ground_truth 为两个 PDB 各保存一个 ligand occurrence, 但体素规模相差 100 倍.
    ground_truth = {
        "large": (
            np.asarray([1], dtype=np.int32),
            (large_true.astype(np.int32),),
            (1, 1, 1000),
        ),
        "small": (
            np.asarray([1], dtype=np.int32),
            (np.asarray([[0, 0, 0]], dtype=np.int32),),
            (1, 1, 1),
        ),
    }
    return (("large", large), ("small", small)), ground_truth


def test_basic_parallel_result_is_exact_and_uses_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """basic 最终体素门槛并发必须逐字段等于串行结果, 且任务确实重叠."""
    centered_items, ground_truth = _build_parallel_tuning_case()
    arguments = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "score_mode": "basic",
        "score_parameter_grid": None,
        "refinement_multipliers": None,
        "prefiltered_min_voxel": 1,
        "min_voxel_values": [1, 2, 3, 4],
        "objective_beta": 1.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
    }
    serial = tune_centered_selection(**arguments, workers=1)

    original_objective = calibration_module._selection_objective
    first_two_tasks = threading.Barrier(2)
    worker_threads: set[int] = set()
    call_count = 0
    state_lock = threading.Lock()

    def synchronized_objective(*args: object, **kwargs: object) -> float:
        """让首两个最小体素数组合并发会合后调用正式目标函数."""

        nonlocal call_count
        with state_lock:
            current_call = call_count
            call_count += 1
            worker_threads.add(threading.get_ident())
        if current_call < 2:
            first_two_tasks.wait(timeout=5.0)
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(
        calibration_module, "_selection_objective", synchronized_objective
    )
    parallel = tune_centered_selection(**arguments, workers=4)

    assert parallel == serial
    assert len(worker_threads) >= 2


def test_gaussian_parallel_result_is_exact() -> None:
    """Gaussian 粗搜, 细搜和最终体素门槛的并发结果必须逐字段等于串行结果."""
    centered_items, ground_truth = _build_parallel_tuning_case()
    arguments = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "score_mode": "gaussian",
        "score_parameter_grid": {
            "tau_angstrom": [0.75, 1.0],
            "lambda_positive": [0.1, 0.2],
            "lambda_negative": [0.05],
        },
        "refinement_multipliers": {
            "lambda": [0.8, 1.2],
        },
        "prefiltered_min_voxel": 1,
        "min_voxel_values": [1, 2, 3],
        "objective_beta": 2.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
    }

    serial = tune_centered_selection(**arguments, workers=1)
    parallel = tune_centered_selection(**arguments, workers=4)

    assert parallel == serial


def test_basic_and_gaussian_searches_use_same_macro_objective() -> None:
    """basic 与 Gaussian 的全部阶段必须选择同一 PDB 等权 macro 三项目标."""

    centered_items, ground_truth = _build_imbalanced_macro_tuning_case()
    common = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "prefiltered_min_voxel": 1,
        "min_voxel_values": [1],
        "objective_beta": 1.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
        "workers": 1,
    }
    # large 的三项 F1 为 2/11, 2/3, 2/3; small 的三项 F1 均为 1.
    expected_objective = ((2.0 / 11.0) + (2.0 / 3.0) * 2.0 + 3.0) / 2.0
    basic = tune_centered_selection(
        **common,
        score_mode="basic",
        score_parameter_grid=None,
        refinement_multipliers=None,
    )
    gaussian = tune_centered_selection(
        **common,
        score_mode="gaussian",
        score_parameter_grid={
            "tau_angstrom": [1.0],
            "lambda_positive": [0.0],
            "lambda_negative": [0.0],
        },
        refinement_multipliers={
            "lambda": [1.0],
        },
    )

    assert basic["score_threshold"] == pytest.approx(0.8)
    assert basic["stages"]["score_threshold"]["objective"] == pytest.approx(
        expected_objective
    )
    assert basic["stages"]["min_voxels"]["objective"] == pytest.approx(
        expected_objective
    )
    assert gaussian["score_threshold"] == pytest.approx(0.8)
    for stage in ("coarse", "refined", "min_voxels"):
        assert gaussian["stages"][stage]["objective"] == pytest.approx(
            expected_objective
        )


def test_basic_exact_objective_tie_keeps_higher_score_threshold() -> None:
    """数学上严格并列的 macro 目标必须保留先遇到的高分阈值."""

    # int16, (7, 3), first 的真实 ligand occurrence 体素.
    first_ground_truth = np.asarray(
        [[0, 0, index] for index in range(7)],
        dtype=np.int16,
    )
    # first 的高分候选含 7 TP 和 28 FP, 低分候选再增加 42 FP.
    first = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_voxel_count": np.asarray([35, 42], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 35, 77], dtype=np.int64),
        "voxel_index_local_zyx": np.asarray(
            [[0, 0, index] for index in range(77)],
            dtype=np.int16,
        ),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
    }
    # int16, (17, 3), second 的真实 ligand occurrence 体素.
    second_ground_truth = np.asarray(
        [[0, 0, index] for index in range(17)],
        dtype=np.int16,
    )
    # second 的高分候选含 3 TP 和 34 FP, 低分候选再增加 12 TP 和 42 FP.
    second_high = np.asarray(
        [[0, 0, index] for index in range(3)]
        + [[0, 0, index] for index in range(17, 51)],
        dtype=np.int16,
    )
    second_low = np.asarray(
        [[0, 0, index] for index in range(3, 15)]
        + [[0, 0, index] for index in range(51, 93)],
        dtype=np.int16,
    )
    second = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_voxel_count": np.asarray([37, 54], dtype=np.int32),
        "voxel_offsets": np.asarray([0, 37, 91], dtype=np.int64),
        "voxel_index_local_zyx": np.concatenate((second_high, second_low)),
        "box_start_zyx": np.zeros((2, 3), dtype=np.int32),
    }
    # result 应保留先扫描到的 0.9, 两个阈值的精确 macro 目标均为 2/9.
    result = tune_centered_selection(
        centered_items=(("first", first), ("second", second)),
        ground_truth_by_pdb={
            "first": (
                np.asarray([1], dtype=np.int32),
                (first_ground_truth.astype(np.int32),),
                (1, 1, 100),
            ),
            "second": (
                np.asarray([1], dtype=np.int32),
                (second_ground_truth.astype(np.int32),),
                (1, 1, 100),
            ),
        },
        score_mode="basic",
        score_parameter_grid=None,
        refinement_multipliers=None,
        prefiltered_min_voxel=1,
        min_voxel_values=[1],
        objective_beta=1.0,
        coverage_thresholds=[0.3, 0.5, 0.6],
        topk_values=[3, 4, 5],
        workers=1,
    )

    assert result["score_threshold"] == pytest.approx(0.9)
    assert result["stages"]["score_threshold"]["objective"] == pytest.approx(2.0 / 9.0)


def test_run_tune_stage_parallel_loads_both_modes_and_publishes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式 tune 入口必须并行读取输入, 保持 PDB 顺序并发布 basic/Gaussian JSON."""
    config = OmegaConf.create(
        {
            "calibration": {
                "workers": 2,
                "min_voxel_values": [1, 2],
                "gaussian_grid": {
                    "tau_angstrom": [1.0],
                    "lambda_positive": [0.2],
                    "lambda_negative": [0.1],
                },
                "gaussian_refinement": {
                    "lambda": [1.0],
                },
            },
            "evaluation": {
                "coverage_thresholds": [0.3, 0.5, 0.6],
                "topk_values": [3, 4, 5],
            },
        }
    )
    data_root = tmp_path / "data"
    output_root = tmp_path / "artifacts"
    pdb_ids = ("second", "first")
    voxel_rows = {
        "second": np.asarray([[1, 1, 1], [1, 1, 2]], dtype=np.int32),
        "first": np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int32),
    }
    for index, pdb_id in enumerate(pdb_ids):
        density_root = data_root / "density" / pdb_id
        density_root.mkdir(parents=True)
        np.savez(
            density_root / "ligand_area.npz",
            grid_shape_zyx=np.asarray([4, 4, 4], dtype=np.int32),
            mask_7=voxel_rows[pdb_id],
        )
        basic_paths = Stage1ArtifactPaths(
            output_root,
            "unet_c1",
            "calibration",
            pdb_id,
        )
        publish_stage1_artifact(
            basic_paths.artifact("F1_blobs"),
            {
                "blob_index": np.asarray([index], dtype=np.int32),
                "source_probability_mean": np.asarray(
                    [0.8 - 0.2 * index], dtype=np.float32
                ),
                "voxel_count": np.asarray([2], dtype=np.int32),
                "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
                "voxel_index_global_zyx": voxel_rows[pdb_id],
            },
            None,
        )
        gaussian_paths = Stage1ArtifactPaths(
            output_root,
            "Find_0",
            "calibration",
            pdb_id,
        )
        publish_stage1_artifact(
            gaussian_paths.artifact("F2_blobs"),
            {
                "blob_index": np.asarray([0], dtype=np.int32),
                "voxel_count": np.asarray([2], dtype=np.int32),
                "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
                "voxel_index_global_zyx": voxel_rows[pdb_id],
            },
            None,
        )
        publish_stage1_artifact(
            gaussian_paths.artifact("F2_centered"),
            {
                "source_blob_index": np.asarray([0], dtype=np.int32),
                "source_probability_mean": np.asarray(
                    [0.8 - 0.2 * index], dtype=np.float32
                ),
                "source_voxel_count": np.asarray([2], dtype=np.int32),
                "voxel_offsets": np.asarray([0, 2], dtype=np.int64),
                "voxel_index_local_zyx": voxel_rows[pdb_id].astype(np.int16),
                "box_start_zyx": np.zeros((1, 3), dtype=np.int32),
                "A_offsets": np.asarray([0, 1], dtype=np.int64),
                "A_coord_local_xyz": voxel_rows[pdb_id][:1, ::-1].astype(np.float32)
                + 0.5,
                "A_probability": np.asarray([0.75], dtype=np.float32),
                "voxel_size_world": np.ones((1, 3), dtype=np.float32),
            },
            None,
        )

    original_load_npz = pipeline_module.load_stage1_npz
    original_load_occurrence = pipeline_module.load_occurrence_voxels
    original_tune = pipeline_module.tune_centered_selection
    first_two_loads = threading.Barrier(2)
    loader_threads: set[int] = set()
    load_count = 0
    state_lock = threading.Lock()

    def synchronized_npz_load(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        """记录 NPZ 加载线程并让最先两个加载任务并发会合."""

        nonlocal load_count
        with state_lock:
            current_load = load_count
            load_count += 1
            loader_threads.add(threading.get_ident())
        if current_load < 2:
            first_two_loads.wait(timeout=5.0)
        return original_load_npz(*args, **kwargs)

    def synchronized_occurrence_load(
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]:
        """记录 occurrence 加载线程并复用同一并发屏障."""

        nonlocal load_count
        with state_lock:
            current_load = load_count
            load_count += 1
            loader_threads.add(threading.get_ident())
        if current_load < 2:
            first_two_loads.wait(timeout=5.0)
        return original_load_occurrence(*args, **kwargs)

    observed_calls: list[dict[str, object]] = []

    def recording_tune(**kwargs: object) -> dict[str, object]:
        """记录 pipeline 传入的候选字段与 PDB 顺序后调用正式调参."""

        items = tuple(kwargs["centered_items"])
        kwargs["centered_items"] = items
        observed_calls.append(
            {
                "score_mode": kwargs["score_mode"],
                "workers": kwargs["workers"],
                "pdb_ids": tuple(pdb_id for pdb_id, _ in items),
                "first_fields": tuple(items[0][1]),
            }
        )
        return original_tune(**kwargs)

    monkeypatch.setattr(pipeline_module, "load_stage1_npz", synchronized_npz_load)
    monkeypatch.setattr(
        pipeline_module,
        "load_occurrence_voxels",
        synchronized_occurrence_load,
    )
    monkeypatch.setattr(pipeline_module, "tune_centered_selection", recording_tune)

    basic = run_tune_stage(
        config=config,
        data_root=data_root,
        producer="unet_c1",
        split="calibration",
        pdb_ids=pdb_ids,
        output_root=output_root,
        alpha=1.0,
        score_mode="basic",
        objective_beta=1.0,
        prefiltered_min_voxel=1,
    )
    gaussian = run_tune_stage(
        config=config,
        data_root=data_root,
        producer="Find_0",
        split="calibration",
        pdb_ids=pdb_ids,
        output_root=output_root,
        alpha=2.0,
        score_mode="gaussian",
        objective_beta=2.0,
        prefiltered_min_voxel=1,
    )

    assert len(loader_threads) >= 2
    assert observed_calls[0]["pdb_ids"] == pdb_ids
    assert observed_calls[0]["workers"] == 2
    assert "source_blob_index" in observed_calls[0]["first_fields"]
    assert "blob_index" not in observed_calls[0]["first_fields"]
    assert observed_calls[1]["pdb_ids"] == pdb_ids
    assert observed_calls[1]["workers"] == 2
    assert "A_offsets" in observed_calls[1]["first_fields"]
    basic_json = output_root / "unet_c1" / "tuning" / "F1_basic.json"
    gaussian_json = output_root / "Find_0" / "tuning" / "F2_gaussian.json"
    assert json.loads(basic_json.read_text(encoding="utf-8")) == basic
    assert json.loads(gaussian_json.read_text(encoding="utf-8")) == gaussian
    assert not (output_root / "unet_c1" / "calibration" / "F1_basic.json").exists()
    assert not (output_root / "Find_0" / "calibration" / "F2_gaussian.json").exists()


def test_gaussian_out_of_order_completion_keeps_first_tied_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任务乱序完成且全部目标并列时, 三个搜索阶段仍保留配置中的首项."""
    centered_items, ground_truth = _build_parallel_tuning_case()
    arguments = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "score_mode": "gaussian",
        "score_parameter_grid": {
            "tau_angstrom": [1.0, 2.0],
            "lambda_positive": [0.2, 0.4],
            "lambda_negative": [0.1],
        },
        "refinement_multipliers": {
            "lambda": [0.8, 1.0],
        },
        "prefiltered_min_voxel": 100,
        "min_voxel_values": [1, 2],
        "objective_beta": 2.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
    }
    serial = tune_centered_selection(**arguments, workers=1)

    original_atom_terms = calibration_module.sum_gaussian_atom_terms
    original_score_scan = calibration_module._scan_actual_score_thresholds
    original_selection_objective = calibration_module._selection_objective
    atom_barrier = threading.Barrier(2)
    scan_barriers = {
        "coarse": threading.Barrier(2),
        "refined": threading.Barrier(2),
    }
    scan_release = {stage: threading.Event() for stage in scan_barriers}
    minimum_barrier = threading.Barrier(2)
    minimum_release = threading.Event()
    atom_threads: set[int] = set()
    scan_threads = {stage: set() for stage in scan_barriers}
    minimum_threads: set[int] = set()
    completion_order: list[tuple[str, int]] = []
    atom_call_count = 0
    scan_call_count = 0
    minimum_call_count = 0
    state_lock = threading.Lock()

    def synchronized_atom_terms(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """让首两个 Gaussian 原子项任务并发会合并记录工作线程."""

        nonlocal atom_call_count
        with state_lock:
            current_call = atom_call_count
            atom_call_count += 1
            atom_threads.add(threading.get_ident())
        if current_call < 2:
            atom_barrier.wait(timeout=5.0)
        return original_atom_terms(*args, **kwargs)

    def delayed_score_scan(
        facts_by_pdb: object,
        scores_by_pdb: object,
        min_voxels: int,
        beta: float,
    ) -> dict[str, float]:
        """反转同阶段前两个分数扫描的完成顺序并记录工作线程."""

        nonlocal scan_call_count
        with state_lock:
            current_call = scan_call_count
            scan_call_count += 1
        if current_call < 4:
            stage = "coarse"
            stage_offset = current_call
        else:
            stage = "refined"
            stage_offset = current_call - 4
        with state_lock:
            scan_threads[stage].add(threading.get_ident())
        if stage_offset < 2:
            scan_barriers[stage].wait(timeout=5.0)
        if stage_offset == 0:
            if not scan_release[stage].wait(timeout=5.0):
                raise RuntimeError(f"{stage} 首个配置没有等到另一个任务先完成")
        result = original_score_scan(
            facts_by_pdb,
            scores_by_pdb,
            min_voxels,
            beta,
        )
        with state_lock:
            completion_order.append((stage, stage_offset))
        if stage_offset != 0:
            scan_release[stage].set()
        return result

    def delayed_minimum_objective(*args: object, **kwargs: object) -> float:
        """反转前两个最小体素数任务的完成顺序并记录工作线程."""

        nonlocal minimum_call_count
        with state_lock:
            current_call = minimum_call_count
            minimum_call_count += 1
            minimum_threads.add(threading.get_ident())
        if current_call < 2:
            minimum_barrier.wait(timeout=5.0)
        if current_call == 0:
            if not minimum_release.wait(timeout=5.0):
                raise RuntimeError("min_voxels 首个配置没有等到另一个任务先完成")
        objective = original_selection_objective(*args, **kwargs)
        with state_lock:
            completion_order.append(("min_voxels", current_call))
        if current_call != 0:
            minimum_release.set()
        return objective

    monkeypatch.setattr(
        calibration_module, "sum_gaussian_atom_terms", synchronized_atom_terms
    )
    monkeypatch.setattr(
        calibration_module, "_scan_actual_score_thresholds", delayed_score_scan
    )
    monkeypatch.setattr(
        calibration_module, "_selection_objective", delayed_minimum_objective
    )
    parallel = tune_centered_selection(**arguments, workers=4)

    assert parallel == serial
    assert len(atom_threads) >= 2
    assert all(len(thread_ids) >= 2 for thread_ids in scan_threads.values())
    assert len(minimum_threads) >= 2
    for stage in ("coarse", "refined", "min_voxels"):
        first_completed = next(entry for entry in completion_order if entry[0] == stage)
        assert first_completed[1] != 0
    assert parallel["stages"]["coarse"] == {
        "objective": 0.0,
        "tau_angstrom": 1.0,
        "lambda_positive": 0.2,
        "lambda_negative": 0.1,
        "score_threshold": 0.0,
    }
    assert parallel["stages"]["refined"]["objective"] == 0.0
    assert parallel["stages"]["refined"]["tau_angstrom"] == 1.0
    assert parallel["stages"]["refined"]["lambda_positive"] == pytest.approx(0.16)
    assert parallel["stages"]["refined"]["lambda_negative"] == pytest.approx(0.08)
    assert parallel["stages"]["refined"]["score_threshold"] == 0.0
    assert parallel["stages"]["min_voxels"] == {
        "objective": 0.0,
        "min_voxels": 1,
    }
