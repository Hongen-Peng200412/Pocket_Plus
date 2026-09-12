# -*- coding: utf-8 -*-
"""验证 Find_1 真实受体正式入口与 test_1 派生边界."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_find1_shell_uses_official_stages_and_two_gpu_resource_profile() -> None:
    """正式入口应固定 checkpoint、双 GPU 分片、两套评分和 64 CPU 资源值."""

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_evaluation.sh"
    )
    shell_text = shell_path.read_text(encoding="utf-8")
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "stage1_v3.yaml")

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert 'CUDA_VISIBLE_DEVICES=0 stage1 "$@" --shard-count 2 --shard-index 0' in shell_text
    assert 'CUDA_VISIBLE_DEVICES=1 stage1 "$@" --shard-count 2 --shard-index 1' in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "run_tune 1 basic" in shell_text
    assert "run_tune 2 gaussian" in shell_text
    assert "f1_blobs_basic_macro_selected" in shell_text
    assert "f2_centered_gaussian_macro_selected" in shell_text
    assert "derive_test1.py" not in shell_text
    assert int(config.window.batch_size) == 18
    assert int(config.window.workers) == 26
    assert int(config.window.prefetch_batches) == 26
    assert int(config.centered.batch_size) == 12
    assert int(config.centered.workers) == 26
    assert int(config.blob_workers) == 56
    assert int(config.calibration.workers) == 56


def test_find1_scored_centered_shell_reuses_frozen_parameters() -> None:
    """Stage2/Stage3 入口应只产生冻结 F2 Gaussian scored-centered 产物."""

    shell_path = (
        PROJECT_ROOT
        / "训练与运行"
        / "sh"
        / "infer"
        / "find1_real_receptor_scored_centered.sh"
    )
    shell_text = shell_path.read_text(encoding="utf-8")
    config = OmegaConf.load(PROJECT_ROOT / "configs" / "inference" / "stage1_v3.yaml")

    calibration_score = 'run_score_only "${CALIBRATION_JSON}" calibration'
    validation_probability = 'run_probability "${VALIDATION_JSON}" validation'
    validation_blobs = 'run_f2_blobs "${VALIDATION_JSON}" validation'
    validation_centered = 'run_centered "${VALIDATION_JSON}" validation'
    validation_score = 'run_score_only "${VALIDATION_JSON}" validation'

    assert "训练与运行/sh/infer/stage1_v3.sh" in shell_text
    assert "TOP_epoch_04_score_0.6654.ckpt" in shell_text
    assert "stage1_preparation_box_pool_3/split/pdb_split/validation.json" in shell_text
    assert "F2_semantic.json" in shell_text
    assert "F2_gaussian.json" in shell_text
    assert "--semantic-parameters" in shell_text
    assert "--selection-parameters" in shell_text
    assert "--score-only" in shell_text
    assert "--forward-min-voxels 8" in shell_text
    assert "--continue-on-blob-exceed" in shell_text
    assert "--fit-semantic" not in shell_text
    assert " tune " not in shell_text
    assert " evaluate " not in shell_text
    assert shell_text.index(calibration_score) < shell_text.index(validation_probability)
    assert shell_text.index(validation_probability) < shell_text.index(validation_blobs)
    assert shell_text.index(validation_blobs) < shell_text.index(validation_centered)
    assert shell_text.index(validation_centered) < shell_text.index(validation_score)
    assert int(config.window.batch_size) == 18
    assert int(config.window.workers) == 26
    assert int(config.centered.batch_size) == 12
    assert int(config.centered.workers) == 26
