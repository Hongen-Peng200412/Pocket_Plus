#!/usr/bin/env bash
set -euo pipefail

# 本脚本只冻结正式推理输入，不运行模型，也不读取隔离 smoke 目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 已经通过 src/datasets/ops 过滤的 Stage1 集合与总清单。
preparation_root="/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation"
split_root="${preparation_root}/split"
keep_list="${preparation_root}/final_keep_list.jsonl"

# 正式模型文件保持在原训练运行目录；本脚本只记录路径与哈希，不复制 1.5 GB checkpoint。
run_root="/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1"
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"
config="${run_root}/config.yaml"

# 所有正式输入只写入本模型专属目录；artifacts/ 由后续推理脚本创建。
formal_root="/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477"
input_root="${formal_root}/inputs"
mkdir -p "${input_root}"

python -u - \
    "${split_root}" \
    "${keep_list}" \
    "${checkpoint}" \
    "${config}" \
    "${input_root}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from src.datasets.ops.stage1_box_pool import _load_split_pdb_ids


split_root, keep_list, checkpoint, config, input_root = map(Path, sys.argv[1:])

expected_hashes = {
    split_root / "calibration.json": "1a3c6c96d64d3e2442e13678d48228ad41f3605f3f832b9d932d131e18e41df1",
    split_root / "validation.json": "41397543f14b260a4f85190728b5583f7ad917e557b84d9c7ca511dc65ad15bd",
    split_root / "train.json": "73bc8604260ac5e36d6d043c0b40cb6fed515699fdfd778f8bc6047f8e760605",
    keep_list: "9d61eefec559504960a86a2a1091fc05158e314aafe75c8cdc0c1bc808e669b4",
    checkpoint: "87ec6080809a14e2558e10c0740c16371b363fed0f672081b389f46c64928b44",
    config: "3eaae2769bdc7df4f0ddfcd40ac5fd34d4d71a9f5119a1f300dd24ea11b82fcf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_exact(path: Path, content: str) -> None:
    """首次原子写入；同名文件已存在时只接受逐字相同的内容。"""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"正式输入已存在但内容不同，拒绝覆盖: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


actual_hashes: dict[str, str] = {}
for path, expected in expected_hashes.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"源文件 SHA-256 不符合冻结契约: {path}: {actual} != {expected}")
    actual_hashes[str(path)] = actual

# 与 Stage1 BOX pool 的正式读取规则相同：取 pdb_id、小写化、去重并按名称排序。
split_ids = {
    split: _load_split_pdb_ids(split_root / f"{split}.json")
    for split in ("calibration", "validation", "train")
}

keep_ids: set[str] = set()
with keep_list.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            keep_ids.add(str(json.loads(line)["pdb_id"]).strip().lower())
for split, pdb_ids in split_ids.items():
    missing = sorted(set(pdb_ids) - keep_ids)
    if missing:
        raise RuntimeError(f"{split} 包含不在 final_keep_list.jsonl 中的 PDB: {missing[:10]}")

derived: dict[str, dict[str, object]] = {}
for split, pdb_ids in split_ids.items():
    destination = input_root / f"{split}_pdb_ids.json"
    content = json.dumps(list(pdb_ids), ensure_ascii=False, indent=2) + "\n"
    publish_exact(destination, content)
    derived[split] = {
        "path": str(destination),
        "pdb_count": len(pdb_ids),
        "sha256": sha256(destination),
    }

manifest = {
    "stage1_model_name": "Find_0",
    "checkpoint": str(checkpoint),
    "config": str(config),
    "selection_metric": "val_score/global/voxel_ligand_PRAUC",
    "selection_metric_value": 0.6754766702651978,
    "source_hashes": actual_hashes,
    "pdb_extraction": "读取 pdb_id，小写化、去重并按名称排序；与 src.datasets.ops.stage1_box_pool._load_split_pdb_ids 相同",
    "derived_lists": derived,
}
publish_exact(
    input_root / "manifest.json",
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[Find_0 inputs] 正式输入已冻结：${input_root}"
