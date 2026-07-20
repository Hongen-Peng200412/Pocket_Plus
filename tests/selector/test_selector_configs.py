"""Stage1 Selector 配置的 producer 边界与可实例化性测试. """

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.selector.wrapper import build_selector_wrapper_from_config
from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "selector"


def _load_config(name: str) -> dict:
    """把一份独立 YAML 配置解析为普通容器. """
    config = OmegaConf.load(CONFIG_ROOT / name)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def test_find_configs_build_real_vpa_ccln() -> None:
    """全部 Find 必须使用 V/P/A，并只接收需要落盘的 A/P 特征层。"""
    source_dimensions = {
        "A": {
            "A_feat_L0": 49,
            "A_feat_L1": 8,
            "A_feat_L2": 16,
            "A_feat_L3": 24,
        },
        "P": {"P_feat_L2": 16, "P_feat_L3": 24},
    }
    for producer in FIND_MODEL_NAMES:
        name = f"{producer}.yaml"
        config = _load_config(name)
        wrapper = build_selector_wrapper_from_config(config, source_dimensions)
        assert config["stage1_model_name"] == producer
        assert config["selector_variant"] == "ccln_v48d_oracle"
        assert "V_grid_source_order" not in config["model"]
        assert "use_V_multiscale" not in config["model"]
        assert wrapper.model.modalities == ("V", "P", "A")
        assert tuple(wrapper.model.a_fusion.source_order) == (
            "A_feat_L0",
            "A_feat_L1",
            "A_feat_L2",
            "A_feat_L3",
        )
        assert tuple(wrapper.model.p_fusion.source_order) == ("P_feat_L2", "P_feat_L3")


def test_unet_config_builds_v_only_ccln() -> None:
    """unet_c1 的 Selector 不得实例化不存在的 P/A 模态. """
    config = _load_config("unet_c1.yaml")
    wrapper = build_selector_wrapper_from_config(config, {"A": {}, "P": {}})
    assert config["stage1_model_name"] == "unet_c1"
    assert config["selector_variant"] == "ccln_v48d_oracle"
    assert "V_grid_source_order" not in config["model"]
    assert "use_V_multiscale" not in config["model"]
    assert wrapper.model.modalities == ("V",)
    assert wrapper.model.a_fusion is None
    assert wrapper.model.p_fusion is None


def test_all_configs_keep_formal_density_munet_shape_and_channels() -> None:
    """全部 producer 的正式 DensityMUNetLite 统一为 80³ 与 [32,64,64,128]. """
    for producer in STAGE1_MODEL_NAMES:
        name = f"{producer}.yaml"
        config = _load_config(name)
        assert config["model"]["density_input_shape_zyx"] == [80, 80, 80]
        assert config["model"]["density_channels"] == [32, 64, 64, 128]
        assert config["model"]["density_bottleneck_heads"] == 4
        assert config["model"]["density_bottleneck_layers"] == 1
