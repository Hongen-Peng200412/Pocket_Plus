"""Stage1 producer 单一名单与 Find_2 下游语义测试。"""

from __future__ import annotations

import numpy as np

from src.artifacts.paths import STAGE1_MODEL_NAMES as ARTIFACT_MODEL_NAMES
from src.artifacts.paths import Stage1ArtifactPaths
from src.inference.probability import (
    FIND_MODEL_NAMES as INFERENCE_FIND_MODEL_NAMES,
    STAGE1_MODEL_NAMES as INFERENCE_MODEL_NAMES,
    postprocess_ligand_probability,
)
from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES


def test_stage1_producer_names_have_one_canonical_source() -> None:
    """验证 artifact 与 inference 兼容导出复用同一四模型名单。"""
    assert STAGE1_MODEL_NAMES == ("Find_0", "Find_1", "Find_2", "unet_c1")
    assert FIND_MODEL_NAMES == ("Find_0", "Find_1", "Find_2")
    assert ARTIFACT_MODEL_NAMES is STAGE1_MODEL_NAMES
    assert INFERENCE_MODEL_NAMES is STAGE1_MODEL_NAMES
    assert INFERENCE_FIND_MODEL_NAMES is FIND_MODEL_NAMES


def test_find2_artifact_path_and_probability_use_find_semantics(tmp_path) -> None:
    """验证 Find_2 可寻址正式产物，并按 Find 规则清零 receptor home voxel。"""
    paths = Stage1ArtifactPaths(tmp_path, "Find_2", "validation", "1abc")
    assert paths.pdb_root == tmp_path / "Find_2" / "validation" / "1abc"

    probability = np.ones((2, 2, 2), dtype=np.float32)
    hardmask = np.zeros((2, 2, 2), dtype=np.bool_)
    hardmask[1, 0, 1] = True
    processed = postprocess_ligand_probability(probability, "Find_2", hardmask)
    assert processed[1, 0, 1] == 0.0
    assert int(np.count_nonzero(processed)) == 7
