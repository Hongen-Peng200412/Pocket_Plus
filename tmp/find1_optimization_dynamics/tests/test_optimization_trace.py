from __future__ import annotations

import torch
from omegaconf import OmegaConf

from tmp.find1_optimization_dynamics import optimization_trace


class _DatasetStub:
    def __init__(self) -> None:
        self.epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_sample_positions_remain_valid_above_float32_integer_range() -> None:
    numel = (1 << 24) + 3

    positions = optimization_trace._sample_positions(numel)

    assert optimization_trace._sample_positions(0) == ()
    assert optimization_trace._sample_positions(1) == (0,)
    assert optimization_trace._sample_positions(4) == (0, 1, 2, 3)
    assert len(positions) == 16
    assert positions[0] == 0
    assert positions[-1] == numel - 1
    assert tuple(sorted(set(positions))) == positions


def test_tensor_signature_preserves_scalar_raw_bytes() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        tensor = torch.tensor(1.25, dtype=dtype)

        signature = optimization_trace.tensor_signature(tensor)

        assert signature["shape"] == []
        assert signature["dtype"] == str(dtype)
        assert signature["numel"] == 1
        assert signature["all_finite"] is True
        assert len(signature["sha256"]) == 64
        assert signature["samples"] == [1.25]


def test_build_dataset_resolves_root_scoped_interpolations(monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "dataset": {
                "_target_": "tests.DatasetStub",
                "box_pool_root": "/storage/box_pool",
                "split_train": "${dataset.box_pool_root}/split_train.npz",
                "all_data_path": "/storage/all_data",
                "enable_random_rotation": True,
            }
        }
    )
    captured: dict[str, object] = {}
    dataset = _DatasetStub()

    def instantiate_stub(dataset_config, **arguments):
        captured["dataset_config"] = dataset_config
        captured["arguments"] = arguments
        return dataset

    monkeypatch.setattr(optimization_trace, "instantiate", instantiate_stub)

    result = optimization_trace._build_dataset(config)

    assert result is dataset
    assert dataset.epoch == 0
    assert captured["arguments"] == {
        "split_file": "/storage/box_pool/split_train.npz",
        "all_data_path": "/storage/all_data",
        "mode": "train",
    }
    assert captured["dataset_config"].enable_random_rotation is False
