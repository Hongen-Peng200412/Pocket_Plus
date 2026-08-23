# -*- coding: utf-8 -*-
"""验证 Stage1 调参外层并发与串行科学结果完全一致."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import src.inference.calibration as calibration_module
from src.inference.calibration import tune_centered_selection


def _build_parallel_tuning_case() -> tuple[
    tuple[tuple[str, dict[str, np.ndarray]], ...],
    dict[str, tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, int, int]]],
]:
    """构造同时适用于 basic 与 Gaussian 的两个 PDB 校准事实."""
    first = {
        "source_blob_index": np.asarray([0, 1], dtype=np.int32),
        "source_probability_mean": np.asarray([0.85, 0.35], dtype=np.float32),
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
        nonlocal call_count
        with state_lock:
            current_call = call_count
            call_count += 1
            worker_threads.add(threading.get_ident())
        if current_call < 2:
            first_two_tasks.wait(timeout=5.0)
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "_selection_objective", synchronized_objective)
    parallel = tune_centered_selection(**arguments, workers=4)

    assert parallel == serial
    assert len(worker_threads) >= 2


def test_gaussian_parallel_result_is_exact() -> None:
    """Gaussian 粗搜、细搜和最终体素门槛的并发结果必须逐字段等于串行结果."""
    centered_items, ground_truth = _build_parallel_tuning_case()
    arguments = {
        "centered_items": centered_items,
        "ground_truth_by_pdb": ground_truth,
        "score_mode": "gaussian",
        "score_parameter_grid": {
            "tau_angstrom": [0.75, 1.0],
            "lambda_positive": [0.1, 0.2],
            "lambda_negative": [0.05],
            "gauss_score_min": [0.2, 0.6],
        },
        "refinement_multipliers": {
            "lambda": [0.8, 1.2],
            "score_threshold": [0.8, 1.2],
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
            "gauss_score_min": [0.5, 0.75],
        },
        "refinement_multipliers": {
            "lambda": [0.8, 1.0],
            "score_threshold": [0.5, 1.0],
        },
        "prefiltered_min_voxel": 100,
        "min_voxel_values": [1, 2],
        "objective_beta": 2.0,
        "coverage_thresholds": [0.3, 0.5, 0.6],
        "topk_values": [3, 4, 5],
    }
    serial = tune_centered_selection(**arguments, workers=1)

    original_objective = calibration_module._selection_objective
    completion_order: list[int] = []
    next_call = 0
    state_lock = threading.Lock()

    def delayed_first_objective(*args: object, **kwargs: object) -> float:
        nonlocal next_call
        with state_lock:
            current_call = next_call
            next_call += 1
        if current_call == 0:
            time.sleep(0.1)
        objective = original_objective(*args, **kwargs)
        with state_lock:
            completion_order.append(current_call)
        return objective

    monkeypatch.setattr(calibration_module, "_selection_objective", delayed_first_objective)
    parallel = tune_centered_selection(**arguments, workers=4)

    assert parallel == serial
    assert completion_order[0] != 0
    assert parallel["stages"]["coarse"] == {
        "objective": 0.0,
        "tau_angstrom": 1.0,
        "lambda_positive": 0.2,
        "lambda_negative": 0.1,
        "score_threshold": 0.5,
    }
    assert parallel["stages"]["refined"]["objective"] == 0.0
    assert parallel["stages"]["refined"]["tau_angstrom"] == 1.0
    assert parallel["stages"]["refined"]["lambda_positive"] == pytest.approx(0.16)
    assert parallel["stages"]["refined"]["lambda_negative"] == pytest.approx(0.08)
    assert parallel["stages"]["refined"]["score_threshold"] == pytest.approx(0.25)
    assert parallel["stages"]["min_voxels"] == {
        "objective": 0.0,
        "min_voxels": 1,
    }
