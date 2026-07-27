"""Stage1 80³/stride40 完整图融合与 hardmask 规则测试. """

from __future__ import annotations

import numpy as np

from src.inference.full_map import (
    gaussian_window_weight,
    infer_full_map,
    window_starts_1d,
)
from src.artifacts.paths import Stage1ArtifactPaths
from src.inference.runner import (
    FullMapTaskInput,
    ProductionTask,
    Stage1ProductionRunner,
    make_probability_role_producer,
)


class _NumpyVoxelModel:
    """把 batch 中预先构造的 logits 原样作为 voxel-only 输出. """

    def forward_voxel_probability(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """返回 (B,1,80,80,80) 的测试 logits. """
        return batch["logits"]


def test_window_starts_force_real_last_window() -> None:
    assert window_starts_1d(80, 80, 40) == (0,)
    assert window_starts_1d(121, 80, 40) == (0, 40, 41)


def test_gaussian_weight_matches_frozen_formula() -> None:
    weight = gaussian_window_weight((80, 80, 80), sigma=0.5)
    assert weight.dtype == np.float32
    assert weight.shape == (80, 80, 80)
    expected_corner = np.exp(-3.0 / (2.0 * 0.5**2))
    assert np.isclose(float(weight[0, 0, 0]), expected_corner, rtol=1e-6)
    assert float(weight[39, 39, 39]) > float(weight[0, 0, 0]) > 0.0


def test_full_map_fusion_covers_edges_and_masks_only_find() -> None:
    full_shape = (81, 82, 83)
    model = _NumpyVoxelModel()

    def build_batch(starts: tuple[tuple[int, int, int], ...]) -> dict[str, np.ndarray]:
        logits = np.zeros((len(starts), 1, 80, 80, 80), dtype=np.float32)
        for row, start in enumerate(starts):
            logits[row] = np.float32(sum(start) / 100.0)
        return {"logits": logits}

    hardmask = np.zeros(full_shape, dtype=np.bool_)
    hardmask[0, 0, 0] = True
    find_result = infer_full_map(
        model=model,
        full_shape_zyx=full_shape,
        window_batch_builder=build_batch,
        stage1_model_name="Find_0",
        receptor_hardmask_full=hardmask,
        window_batch_size=3,
    )
    assert find_result.probability_map.dtype == np.float32
    assert find_result.weight_sum.dtype == np.float32
    assert np.all(find_result.weight_sum > 0.0)
    assert find_result.probability_map[0, 0, 0] == 0.0
    assert find_result.probability_map[-1, -1, -1] > 0.5

    unet_result = infer_full_map(
        model=model,
        full_shape_zyx=full_shape,
        window_batch_builder=build_batch,
        stage1_model_name="unet_c1",
        receptor_hardmask_full=None,
        window_batch_size=4,
    )
    assert unet_result.probability_map[0, 0, 0] == 0.5


def test_probability_role_factory_is_directly_runnable_and_resumable(tmp_path) -> None:
    model = _NumpyVoxelModel()

    def input_provider(task: ProductionTask) -> FullMapTaskInput:
        assert task == ProductionTask("unet_c1", "calibration", "cal_001")
        return FullMapTaskInput(
            model=model,
            full_shape_zyx=(80, 80, 80),
            window_batch_builder=lambda starts: {
                "logits": np.zeros((len(starts), 1, 80, 80, 80), dtype=np.float32)
            },
            receptor_hardmask_full=None,
            window_batch_size=1,
            origin_xyz=(1.0, 2.0, 3.0),
            voxel_size_xyz=(1.0, 1.0, 1.0),
        )

    runner = Stage1ProductionRunner(
        output_root=str(tmp_path),
        role_producers={"probability": make_probability_role_producer(input_provider)},
        owner_token="probability-worker",
    )
    task = ProductionTask("unet_c1", "calibration", "cal_001")
    first = runner.run_calibration_probability((task,))[0]
    second = runner.run_calibration_probability((task,))[0]
    paths = Stage1ArtifactPaths(tmp_path, "unet_c1", "calibration", "cal_001")
    assert first.status == "completed"
    assert second.status == "skipped_complete"
    assert paths.probability_npz.is_file()
    assert paths.role_complete_path("probability").is_file()
