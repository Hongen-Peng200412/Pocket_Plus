from __future__ import annotations

from pathlib import Path

import json
import torch
import numpy as np
from omegaconf import OmegaConf

from src.inference.get_pred import load_model
import src.inference.parse_input as parse_input
from src.inference.utils.utils import generate_param_grid
from src.inference.utils.yield_json_from_raw_sample import load_raw_pairs


def test_generate_param_grid_supports_dict_style_config() -> None:
    param_cfg = {
        "threshold": {"min": 0.30, "max": 0.50, "step": 0.10},
        "core_decay_mode": ["hard", "linear"],
    }

    param_grid = generate_param_grid(param_cfg)

    assert len(param_grid) == 6
    assert param_grid[0] == {"threshold": 0.3, "core_decay_mode": "hard"}
    assert param_grid[-1] == {"threshold": 0.5, "core_decay_mode": "linear"}


def test_load_model_falls_back_to_training_snapshot_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp_run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    ckpt_path = ckpt_dir / "last.ckpt"
    torch.save(
        {
            "state_dict": {},
            "hyper_parameters": {},
        },
        ckpt_path,
    )

    config_path = run_dir / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "backbone": {
                        "_target_": "torch.nn.Identity",
                    }
                }
            }
        ),
        config_path,
    )

    model = load_model(str(ckpt_path), device="cpu")

    assert isinstance(model, torch.nn.Identity)


def test_load_raw_pairs_supports_json_file(tmp_path: Path) -> None:
    raw_pairs_path = tmp_path / "raw_pairs.json"
    raw_pairs_path.write_text(
        json.dumps(
            [
                {
                    "cif_path": "/tmp/sample.cif",
                    "map_path": "/tmp/sample.map",
                    "cif_gt_path": "/tmp/sample_gt.cif",
                }
            ]
        ),
        encoding="utf-8",
    )

    pairs = load_raw_pairs(str(raw_pairs_path))

    assert pairs == [
        ("/tmp/sample.cif", "/tmp/sample.map", "/tmp/sample_gt.cif")
    ]


def test_load_raw_pairs_supports_utf8_bom_json(tmp_path: Path) -> None:
    raw_pairs_path = tmp_path / "raw_pairs_bom.json"
    raw_pairs_path.write_bytes(
        json.dumps(
            [
                {
                    "cif_path": "/tmp/sample.cif",
                    "map_path": "/tmp/sample.map",
                }
            ]
        ).encode("utf-8-sig")
    )

    pairs = load_raw_pairs(str(raw_pairs_path))

    assert pairs == [
        ("/tmp/sample.cif", "/tmp/sample.map", None)
    ]


def test_load_from_raw_cif_uses_single_preparsed_atom_info(monkeypatch) -> None:
    import Make_Data.process_and_label as process_and_label
    import Pocket.utils.mrc_tools as mrc_tools
    import processedPDB_EMDB_binder.bind as binder

    calls = {"parse": 0, "bind": 0}
    atom_info = {
        "coords": np.array([[0.5, 1.5, 1.5], [1.5, 1.5, 1.5]], dtype=np.float32),
        "features": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    def fake_get_features_when_infer(**kwargs):
        calls["parse"] += 1
        return atom_info, {}, {}

    def fake_load_map(path):
        assert path == "fake.map"
        return np.arange(27, dtype=np.float32).reshape(3, 3, 3), np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def fake_make_model_grid(grid, voxel_size, origin, target_voxel_size):
        assert target_voxel_size == 1.0
        return grid, voxel_size, origin

    def fake_bind_atoms_feature_to_emdb(**kwargs):
        calls["bind"] += 1
        assert kwargs["pre_parsed_atom_info"] is atom_info
        assert kwargs["emdb_path"] == "fake.map"
        assert kwargs["return_atom_pos_array"] is True
        feature_grid = np.zeros((2, 3, 3, 3), dtype=np.float32)
        feature_grid[:, 1, 1, 1] = np.array([10.0, 20.0], dtype=np.float32)
        return feature_grid, atom_info["coords"]

    monkeypatch.setattr(process_and_label, "get_features_when_infer", fake_get_features_when_infer)
    monkeypatch.setattr(mrc_tools, "load_map", fake_load_map)
    monkeypatch.setattr(mrc_tools, "make_model_grid", fake_make_model_grid)
    monkeypatch.setattr(binder, "bind_AtomsFeature_to_EMDB", fake_bind_atoms_feature_to_emdb)
    monkeypatch.setattr(parse_input.os.path, "exists", lambda path: path == "fake.map")

    result = parse_input.load_from_raw_cif(
        cif_path="fake.cif",
        map_path="fake.map",
        target_voxel_size=1.0,
        compute_density=True,
        select_first_model=False,
        error_dir="fake_error_dir",
    )

    assert calls == {"parse": 1, "bind": 1}
    assert np.array_equal(result["atom_coords"], atom_info["coords"])
    assert np.array_equal(result["atom_feat"], atom_info["features"])
    assert result["grid"].shape == (3, 3, 3, 3)
    assert result["emdb_channels"] == 1
    assert np.isclose(result["grid"][0].mean(), 0.0, atol=1e-6)
    assert np.isclose(result["grid"][0].std(), 1.0, atol=1e-6)
    assert np.array_equal(result["grid"][1:, 1, 1, 1], np.array([10.0, 20.0], dtype=np.float32))
