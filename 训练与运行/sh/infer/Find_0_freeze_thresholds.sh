#!/usr/bin/env bash
set -euo pipefail

# 本脚本在全部 calibration probability 分片完成后运行；它只使用 CPU 和已经落盘的概率图。
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

data_root="/storage/penghongen/AdaLigand/Ori_Data"  # occurrence 真值来自 A–G 正式 ligand_area.npz。
formal_root="/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${formal_root}/inputs/calibration_pdb_ids.json"  # 必须覆盖完整 calibration 集合，不能传入单个分片。
output_root="${formal_root}/artifacts"
min_voxels=15       # 连通组件少于 15 个体素时不作为实例候选。
max_voxels=2046     # 连通组件体素数上限，沿用冻结的 Stage1 calibration 契约。
denominator=32768   # 阈值网格包含 0/32768 至 32768/32768，共 32769 个位置。

[[ -f "${formal_root}/inputs/manifest.json" && -f "${pdb_list}" ]] || { echo "缺少冻结 calibration 输入" >&2; exit 2; }

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli freeze-thresholds \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --min-voxels "${min_voxels}" \
    --max-voxels "${max_voxels}" \
    --denominator "${denominator}"
