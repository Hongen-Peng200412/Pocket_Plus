#!/usr/bin/env bash

# Emap2lig v0.3.4 官方 Find 的 held-out test_0 前向与 Pocket Plus 标准评估入口。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
EMAP2LIG_PROJECT_ROOT="${EMAP2LIG_PROJECT_ROOT:-/home/penghongen/My_Project/Map_Ligand/Emap2lig}"
EMAP2LIG_PYTHON="/home/penghongen/anaconda3/envs/Emap2lig_find_py310_cu121/bin/python"
POCKET_PYTHON="/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python"
DATA_ROOT="/storage/penghongen/AdaLigand/Ori_Data"
PAIR_LIST="${DATA_ROOT}/raw/pair_list.jsonl"
MAP_ROOT="${DATA_ROOT}/raw/emdb_maps"
TEST_ROOT="/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain"
OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHONPATH="${EMAP2LIG_PROJECT_ROOT}/src" "${EMAP2LIG_PYTHON}" -u \
    "${EMAP2LIG_PROJECT_ROOT}/复现Find/run_emap2lig_find_held_out.py" \
    --pdb-json "${TEST_ROOT}/test_0.json" \
    --pair-list-jsonl "${PAIR_LIST}" \
    --map-root "${MAP_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu 0 \
    --detection-batch-size 16

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}" "${POCKET_PYTHON}" -u \
    -m src.inference.baseline.emap2lig_find \
    --test0-json "${TEST_ROOT}/test_0.json" \
    --test1-json "${TEST_ROOT}/test_1.json" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --workers 4 \
    --chunk-depth 8
