from __future__ import annotations

from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
EXPERIMENT_ROOT = CONFIG_ROOT / "experiment"

CPC1_NAMES = {
    "trunk_main",
    "trunk_no_real_density",
    "trunk_hard_scatter",
    "trunk_decoder_fusion",
}
CPC2_NAMES = {
    "heads_trunk_main",
    "heads_trunk_no_real_density",
    "heads_trunk_hard_scatter",
    "heads_trunk_decoder_fusion",
}
CPC3_NAMES = {
    f"refine_{loss}_{ratio}"
    for loss in ("tversky", "tversky_bce", "tversky_rank")
    for ratio in ("73", "82", "91")
}
ADALIGAND_CPC1_NAMES = {"Find_0", "Find_1"}
ADALIGAND_CPC2_NAMES = {"Find_0", "Find_1"}


def _yaml_files(group: str) -> list[Path]:
    """
    列出指定 CPC experiment 目录下的实体 YAML 文件。

    输入参数:
        - group: str, experiment 子目录名, 例如 CPC1/CPC2/CPC3

    输出:
        - files: list[Path], 按文件名排序后的 YAML 文件路径列表
    """
    return sorted((EXPERIMENT_ROOT / group).glob("*.yaml"))


def _load_yaml(path: Path) -> dict:
    """
    读取 YAML 配置为 dict。

    输入参数:
        - path: Path, YAML 文件路径

    输出:
        - data: dict, YAML 顶层映射
    """
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _defaults(data: dict) -> list:
    """
    返回 Hydra defaults 列表。

    输入参数:
        - data: dict, YAML 顶层映射

    输出:
        - defaults: list, 原始 defaults 列表
    """
    defaults = data["defaults"]
    assert isinstance(defaults, list)
    return defaults


def _flat_default_strings(defaults: list) -> set[str]:
    """
    将 defaults 条目转换成便于断言的字符串集合。

    输入参数:
        - defaults: list, Hydra defaults 原始列表

    输出:
        - values: set[str], 包含 "key=value" 或普通字符串条目
    """
    values: set[str] = set()
    for item in defaults:
        if isinstance(item, str):
            values.add(item)
        elif isinstance(item, dict):
            for key, value in item.items():
                values.add(f"{key}={value}")
        else:
            raise TypeError(f"未知 defaults 条目类型: {type(item)!r}")
    return values


def test_cpc_v3_config_files_are_materialized() -> None:
    """
    验证 CPC1/CPC2/CPC3 按 v3 计划实体落盘为 4/4/9 个 YAML。
    """
    assert (CPC1_NAMES | ADALIGAND_CPC1_NAMES).issubset({path.stem for path in _yaml_files("CPC1")})
    assert (CPC2_NAMES | ADALIGAND_CPC2_NAMES).issubset({path.stem for path in _yaml_files("CPC2")})
    assert CPC3_NAMES.issubset({path.stem for path in _yaml_files("CPC3")})


def test_cpc_v3_files_start_with_launch_command() -> None:
    """
    验证每个 CPC v3 YAML 顶部都写有自己的启动命令。
    """
    for group in ("CPC1", "CPC2", "CPC3"):
        for path in _yaml_files(group):
            expected_names = {"CPC1": CPC1_NAMES, "CPC2": CPC2_NAMES, "CPC3": CPC3_NAMES}[group]
            if path.stem not in expected_names:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            header = "\n".join(lines[:8])
            assert f"+experiment={group}/{path.stem}" in header
            assert "src/train.py" in header


def test_cpc1_trunk_main_is_final_main_entry() -> None:
    """
    验证最终 `_main` 入口固定为 CPC1/trunk_main, 不再回到旧 CPC_main。
    """
    data = _load_yaml(EXPERIMENT_ROOT / "CPC1" / "trunk_main.yaml")
    defaults = _flat_default_strings(_defaults(data))
    assert data["name"] == "trunk_main"
    assert data["experiment_group"] == "CPC1"
    assert "override /frozen_module=stage1" in defaults
    assert "CPC_main" not in data["tag"]


def test_cpc2_and_cpc3_encode_stage_boundaries() -> None:
    """
    验证阶段二/三配置显式包含 init_from、small_increment 与冻结阶段。
    """
    for path in _yaml_files("CPC2"):
        if path.stem in ADALIGAND_CPC2_NAMES:
            continue
        data = _load_yaml(path)
        defaults = _flat_default_strings(_defaults(data))
        assert data["init_from"] == "***"
        assert data["experiment_group"] == "CPC2"
        assert "override /train=small_increment" in defaults
        assert "override /frozen_module=stage2" in defaults
        assert any(item.startswith("/experiment/CPC1/trunk_") for item in defaults)

    for path in _yaml_files("CPC3"):
        data = _load_yaml(path)
        defaults = _flat_default_strings(_defaults(data))
        assert data.get("init_from") == "***" or any(
            item.startswith("/experiment/CPC3/refine_tversky") for item in defaults
        )
        assert data.get("experiment_group") == "CPC3" or any(
            item.startswith("/experiment/CPC3/refine_tversky") for item in defaults
        )
        assert "override /frozen_module=stage3" in defaults or any(
            item.startswith("/experiment/CPC3/refine_tversky") for item in defaults
        )


def test_cpc3_refine_grid_ratios_and_losses() -> None:
    """
    验证 CPC3 的 3x3 refine 网格表达了 ratio 与 loss 轴。
    """
    for path in _yaml_files("CPC3"):
        data = _load_yaml(path)
        text = path.read_text(encoding="utf-8")
        if path.stem.endswith("_73"):
            assert "tversky_alpha: 0.3" in text
            assert "tversky_beta: 0.7" in text
        if path.stem.endswith("_82"):
            assert "tversky_alpha: 0.2" in text
            assert "tversky_beta: 0.8" in text
        if path.stem.endswith("_91"):
            assert "tversky_alpha: 0.1" in text
            assert "tversky_beta: 0.9" in text
        if "rank" in path.stem:
            if path.stem.endswith("_73"):
                assert data["model"]["ligand_sparse_refine_w_rank"] == 0.3
            else:
                assert "/experiment/CPC3/refine_tversky_rank_73" in _flat_default_strings(_defaults(data))
        if "bce" in path.stem:
            if path.stem.endswith("_73"):
                assert data["model"]["ligand_sparse_refine_loss"]["focal_gamma"] == 0.0
            else:
                assert "/experiment/CPC3/refine_tversky_bce_73" in _flat_default_strings(_defaults(data))


def test_cpc_v3_hydra_compose() -> None:
    """
    在安装 Hydra 的环境中验证所有 CPC v3 配置可 compose。
    """
    hydra = pytest.importorskip("hydra")
    initialize_config_dir = hydra.initialize_config_dir
    compose = hydra.compose

    experiments = [
        *(f"CPC1/{name}" for name in sorted(CPC1_NAMES)),
        *(f"CPC2/{name}" for name in sorted(CPC2_NAMES)),
        *(f"CPC3/{name}" for name in sorted(CPC3_NAMES)),
    ]
    for experiment in experiments:
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
            cfg = compose(config_name="base", overrides=[f"+experiment={experiment}"])
        assert cfg.name == Path(experiment).name
        assert cfg.experiment_group in {"CPC1", "CPC2", "CPC3"}
