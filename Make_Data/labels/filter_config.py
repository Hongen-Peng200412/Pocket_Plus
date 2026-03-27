# -*- coding: utf-8 -*-
"""
================================================================================
配体筛选预设集中配置 / Single Source of Truth for Ligand Filter Presets
================================================================================

新增预设时只需要在本文件新增一个常量:
1. 常量名必须是 `<YOUR_NAME>_PRESET`, 也就是说必须以  _PRESET 作为识别后缀
2. 值必须是 `LigandFilterConfig(...)`
3. CLI/sbatch 中使用的小写名称会自动从常量名推导:
   - 例如: `MY_NEW_PRESET` -> `my_new`
"""

from typing import Dict, List, Optional

from .ligand_filter import LigandFilterConfig, PocketClassRule


# ============================================================================
# 预设定义 / Presets
# ============================================================================

BINARY_PRESET = LigandFilterConfig(
    rules=[
        PocketClassRule(
            class_id=1,
            class_name="pocket",
            binding_threshold=4.0,
            require_metal_ion=False,
            require_peptide_like=False,
            require_nucleotide_like=False,
        ),
    ]
)

FIVE_CLASS_PRESET = LigandFilterConfig(
    rules=[
        PocketClassRule(  # 优先使用前面的
            class_id=1,
            class_name="metal_ion",
            binding_threshold=4.0,
            require_metal_ion=True,
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=2,
            class_name="peptide",
            binding_threshold=4.0,
            require_peptide_like=True,
            max_peptide_length=30, 
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=3,
            class_name="nucleic",
            binding_threshold=4.0,
            require_nucleotide_like=True,
            max_nucleic_length=10, 
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=4,
            class_name="small_molecule",
            binding_threshold=4.0,
            require_metal_ion=False,
            require_peptide_like=False,
            require_nucleotide_like=False,
            require_covalent=False,
        ),
    ]
)

FIVE_CLASS_5_PRESET = LigandFilterConfig(
    rules=[
        PocketClassRule(  # 优先使用前面的
            class_id=1,
            class_name="metal_ion",
            binding_threshold=5.0,
            require_metal_ion=True,
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=2,
            class_name="peptide",
            binding_threshold=5.0,
            require_peptide_like=True,
            max_peptide_length=30, 
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=3,
            class_name="nucleic",
            binding_threshold=5.0,
            require_nucleotide_like=True,
            max_nucleic_length=10, 
            require_covalent=False,
        ),
        PocketClassRule(
            class_id=4,
            class_name="small_molecule",
            binding_threshold=5.0,
            require_metal_ion=False,
            require_peptide_like=False,
            require_nucleotide_like=False,
            require_covalent=False,
        ),
    ]
)

THREE_CLASS_PRESET = LigandFilterConfig(   # 优先使用前面的
    rules=[
        PocketClassRule(                
            class_id=1,
            class_name="molecule",
            binding_threshold=5.0,
            require_metal_ion=False,
            require_covalent=False, 
            min_contact_residues=2
        ),
        PocketClassRule(
            class_id=2,
            class_name="metal",
            binding_threshold=5.0,
            require_metal_ion=True,
            require_covalent=False,
            min_contact_residues=2
        ),
    ]
)



# ============================================================================
# 自动发现工具 / Discovery helpers
# ============================================================================

_PRESET_SUFFIX = "_PRESET"


def _symbol_to_preset_name(symbol: str) -> str:
    """
    将模块常量名转换为 CLI 可用名。

    例如:
        - "THREE_CLASS_PRESET" -> "three_class"
    """
    return symbol[: -len(_PRESET_SUFFIX)].lower()


def get_filter_presets() -> Dict[str, LigandFilterConfig]:
    """
    自动发现本模块中所有可用预设。

    # 识别规则:
        - 变量名匹配 `*_PRESET`
        - 值类型是 `LigandFilterConfig`

    # 输出:
        - presets: dict[str, LigandFilterConfig], 键是 CLI/sbatch 使用的预设名
    """
    # presets: dict[str, LigandFilterConfig], 预设名称 -> 配置
    presets: Dict[str, LigandFilterConfig] = {}
    for symbol, value in globals().items():
        if not symbol.endswith(_PRESET_SUFFIX):
            continue
        if not isinstance(value, LigandFilterConfig):
            continue

        preset_name = _symbol_to_preset_name(symbol)
        if not preset_name:
            continue
        if preset_name in presets and presets[preset_name] is not value:
            raise ValueError(
                f"Duplicate preset name '{preset_name}' derived from symbol '{symbol}'."
            )
        presets[preset_name] = value
    return presets


def list_filter_preset_names() -> List[str]:
    """
    返回当前全部可用预设名（升序）。
    """
    return sorted(get_filter_presets().keys())


def get_filter_preset(name: str) -> Optional[LigandFilterConfig]:
    """
    按预设名读取配置；不存在时返回 None。
    """
    return get_filter_presets().get(name)


def get_default_filter_preset_name() -> str:
    """
    获取默认预设名。

    # 规则:
        1. 若存在 `binary`，优先使用 `binary`
        2. 否则使用字典序第一个
    """
    preset_names = list_filter_preset_names()
    if len(preset_names) == 0:
        raise ValueError("No filter presets found in labels/filter_config.py.")
    if "binary" in preset_names:
        return "binary"
    return preset_names[0]


__all__ = [
    "LigandFilterConfig",
    "PocketClassRule",
    "BINARY_PRESET",
    "THREE_CLASS_PRESET",
    "get_filter_presets",
    "list_filter_preset_names",
    "get_filter_preset",
    "get_default_filter_preset_name",
]
