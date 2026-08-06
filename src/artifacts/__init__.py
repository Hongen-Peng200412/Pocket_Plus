"""统一导出 AdaLigand Stage1 产物的路径、发布状态和数值归档接口。

主要入口:
    - `Stage1ArtifactPaths`: 从 `producer/split/pdb_id` 身份解析全部正式产物路径。
    - `PdbRunningLease`: 以 PDB 级 `_RUNNING` 目录协调互斥生产。
    - `atomic_write_json`、`atomic_savez_compressed`: 校验临时文件后原子发布 JSON 或纯数值 NPZ。
    - `pack_centered_entries`、`validate_centered_archive`: 构造并校验各类 centered 聚合归档。

本模块只汇总公共符号，不读取或写入任何产物。
"""

from .io import (
    atomic_savez_compressed,
    atomic_write_json,
    load_npz_strict,
    pack_centered_entries,
    validate_centered_archive,
    validate_offsets,
)
from .paths import (
    CENTERED_ROLES,
    F_ALPHA_CENTERED_ROLE_BY_FRACTION,
    F_ALPHA_CENTERED_ROLES,
    OUTPUT_ROLES,
    STAGE1_MODEL_NAMES,
    Stage1ArtifactPaths,
)
from .states import (
    PdbRunningLease,
    is_role_complete,
    mark_blob_exceed,
    mark_role_complete,
    pdb_is_consumable,
)

__all__ = [
    "OUTPUT_ROLES",
    "CENTERED_ROLES",
    "F_ALPHA_CENTERED_ROLE_BY_FRACTION",
    "F_ALPHA_CENTERED_ROLES",
    "STAGE1_MODEL_NAMES",
    "PdbRunningLease",
    "Stage1ArtifactPaths",
    "atomic_savez_compressed",
    "atomic_write_json",
    "is_role_complete",
    "load_npz_strict",
    "mark_blob_exceed",
    "mark_role_complete",
    "pack_centered_entries",
    "pdb_is_consumable",
    "validate_centered_archive",
    "validate_offsets",
]
