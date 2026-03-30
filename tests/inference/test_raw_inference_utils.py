from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.inference.get_pred import load_model
from src.inference.utils.utils import generate_param_grid


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
