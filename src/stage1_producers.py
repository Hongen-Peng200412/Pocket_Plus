"""AdaLigand Stage1 producer 名单的单一来源。"""

from __future__ import annotations


STAGE1_MODEL_NAMES: tuple[str, ...] = ("Find_0", "Find_1", "Find_2", "unet_c1")
FIND_MODEL_NAMES: tuple[str, ...] = tuple(
    name for name in STAGE1_MODEL_NAMES if name.startswith("Find_")
)
