#!/usr/bin/env bash

# 使用冻结的 Find_1 checkpoint 完成 CryoAtom2 最终受体 calibration 与完整 test_0 评估.
# 基础路径选择 F1 语义阈值并使用 source_probability_mean; Gaussian 路径复用 probability, 选择 F2 语义阈值并执行 centered 前向.
# calibration 的 F2 centered 在 Gaussian 参数冻结后以 score-only 原子回填 score/selected; test_1 由任务临时命令保序派生.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"
CALIBRATION_DATA_ROOT="/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/Ori_Data"
TEST_DATA_ROOT="/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data"
CALIBRATION_JSON="/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json"
TEST_0_JSON="/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json"
RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume"
CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_04_score_0.6654.ckpt"
RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts"
PRODUCER="Find_1"
TEST_SPLIT="held_out_test_0"
F2_GAUSSIAN_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_gaussian.json"
export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3.yaml"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

stage1() {
    # 所有生产阶段只通过 Stage1 V3 官方入口进入 Python CLI.
    bash "${STAGE1_ENTRY}" "$@"
}

run_two_gpu_shards() {
    # 两张 H100 使用同一有序 PDB 清单的两个互斥分片; 第一个参数是受体数据根, 其余参数传给官方入口, 任一分片失败均返回非零值.
    local data_root="$1"
    shift
    local first_exit=0
    local second_exit=0
    ADALIGAND_DATA_ROOT="${data_root}" CUDA_VISIBLE_DEVICES=0 bash "${STAGE1_ENTRY}" "$@" --shard-count 2 --shard-index 0 &
    local first_pid=$!
    ADALIGAND_DATA_ROOT="${data_root}" CUDA_VISIBLE_DEVICES=1 bash "${STAGE1_ENTRY}" "$@" --shard-count 2 --shard-index 1 &
    local second_pid=$!
    wait "${first_pid}" || first_exit=$?
    wait "${second_pid}" || second_exit=$?
    if ((first_exit != 0 || second_exit != 0)); then
        return 1
    fi
}

run_probability() {
    # 同一划分只生成一次完整图 probability, F1 与 F2 blobs 共同复用.
    local pdb_json="$1"
    local split="$2"
    local data_root="$3"
    run_two_gpu_shards "${data_root}" probability \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}"
}

run_blobs() {
    # 在完整 PDB 清单上拟合或应用给定 F-alpha 的语义阈值.
    local pdb_json="$1"
    local split="$2"
    local alpha="$3"
    shift 3
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha "${alpha}" \
        "$@"
}

run_centered() {
    # 体素数至少为 8 的 F2 blob 进入 centered 前向; 候选数超过 1,000 时仍保留该 PDB.
    local pdb_json="$1"
    local split="$2"
    local data_root="$3"
    run_two_gpu_shards "${data_root}" centered \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --forward-min-voxels 8 \
        --continue-on-blob-exceed
}

run_tune() {
    # 两种评分都最大化三项 PDB 等权 macro F1 的等权和: semantic、coverage@0.3 与 one-to-one@0.3.
    local alpha="$1"
    local score_mode="$2"
    stage1 tune \
        --producer "${PRODUCER}" \
        --pdb-json "${CALIBRATION_JSON}" \
        --split calibration \
        --output-root "${OUTPUT_ROOT}" \
        --alpha "${alpha}" \
        --objective-beta 1 \
        --score-mode "${score_mode}" \
        --prefiltered-min-voxel 8 \
        --data-root "${CALIBRATION_DATA_ROOT}"
}

run_evaluate() {
    # 使用 calibration 冻结参数评估完整 test_0; test_1 随后只聚合既有逐 PDB 事实.
    local alpha="$1"
    local artifact="$2"
    local evaluation_name="$3"
    local selection_name="$4"
    stage1 evaluate \
        --producer "${PRODUCER}" \
        --pdb-json "${TEST_0_JSON}" \
        --split "${TEST_SPLIT}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha "${alpha}" \
        --artifact "${artifact}" \
        --evaluation-name "${evaluation_name}" \
        --selection-parameters "${OUTPUT_ROOT}/${PRODUCER}/tuning/${selection_name}" \
        --data-root "${TEST_DATA_ROOT}"
}

run_probability "${CALIBRATION_JSON}" calibration "${CALIBRATION_DATA_ROOT}"
run_blobs "${CALIBRATION_JSON}" calibration 1 --fit-semantic --data-root "${CALIBRATION_DATA_ROOT}"
run_tune 1 basic
run_blobs "${CALIBRATION_JSON}" calibration 2 --fit-semantic --data-root "${CALIBRATION_DATA_ROOT}"
run_centered "${CALIBRATION_JSON}" calibration "${CALIBRATION_DATA_ROOT}"
run_tune 2 gaussian
run_two_gpu_shards "${CALIBRATION_DATA_ROOT}" centered \
    --producer "${PRODUCER}" \
    --pdb-json "${CALIBRATION_JSON}" \
    --split calibration \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
    --score-only

run_probability "${TEST_0_JSON}" "${TEST_SPLIT}" "${TEST_DATA_ROOT}"
run_blobs "${TEST_0_JSON}" "${TEST_SPLIT}" 1 \
    --semantic-parameters "${OUTPUT_ROOT}/${PRODUCER}/tuning/F1_semantic.json"
run_evaluate 1 blobs f1_blobs_basic_macro_selected F1_basic.json
run_blobs "${TEST_0_JSON}" "${TEST_SPLIT}" 2 \
    --semantic-parameters "${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_semantic.json"
run_centered "${TEST_0_JSON}" "${TEST_SPLIT}" "${TEST_DATA_ROOT}"
run_evaluate 2 centered f2_centered_gaussian_macro_selected F2_gaussian.json
