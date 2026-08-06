"""Li 逐图阈值、量化和独立 centered 归档测试。"""

from __future__ import annotations

import numpy as np

from src.artifacts.io import pack_centered_entries, validate_centered_archive
from src.inference.Gauss_Scorer import GaussScorerParameters, score_li_centered_entries
from src.inference.li_centered import li_threshold, quantize_li_threshold


def test_li_threshold_is_finite_and_quantized_upward() -> None:
    probability = np.asarray(
        [[[0.0, 0.05, 0.1, 0.7, 0.8, 0.9]]], dtype=np.float32
    )
    raw = li_threshold(probability)
    grid_index, applied = quantize_li_threshold(raw, denominator=32768)

    assert 0.1 < raw < 0.8
    assert applied >= raw
    assert applied - raw < 1.0 / 32768.0
    assert applied == grid_index / 32768.0


def test_empty_li_centered_keeps_centered_contract_and_li_metadata() -> None:
    arrays = pack_centered_entries(
        (),
        "Li_centered",
        role_metadata={
            "li_threshold_raw": 0.25,
            "li_threshold_grid_index": 8192,
            "li_threshold_applied": 0.25,
            "threshold_denominator": 32768,
        },
    )
    validate_centered_archive(arrays, "Li_centered")

    assert arrays["centered_box_index"].shape == (0,)
    assert arrays["source_probability_mean"].shape == (0,)
    assert arrays["li_threshold_grid_index"].tolist() == [8192]
    assert "candidate_eligible" not in arrays
    assert "CLG_id" not in arrays

    score, selected = score_li_centered_entries(
        arrays,
        GaussScorerParameters(0.1, 0.001, 1.0, 1.0),
    )
    assert score.shape == (0,)
    assert selected.shape == (0,)
