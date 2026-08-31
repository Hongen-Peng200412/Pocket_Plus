from __future__ import annotations

from omegaconf import OmegaConf

from tmp.find1_optimization_dynamics import optimization_trace


class _DatasetStub:
    def __init__(self) -> None:
        self.epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


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
