#!/usr/bin/env bash

# 使用已冻结的 Find_1 F2 Gaussian 参数生成 Stage2, Stage3 所需的 scored-centered 产物.
#
# calibration 只复用已有 F2 centered 字段并回填 score/selected, 不执行模型前向.
# validation 依次生成 probability, 应用冻结阈值的 F2 blobs, F2 centered, 再回填同一组 score/selected.
# 本入口不重新调参, 也不生成 calibration 或 validation 评估指标.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"
DATA_ROOT="/storage/penghongen/AdaLigand/Ori_Data"
CALIBRATION_JSON="${DATA_ROOT}/stage1_preparation_box_pool_3/split/pdb_split/calibration.json"
VALIDATION_JSON="${DATA_ROOT}/stage1_preparation_box_pool_3/split/pdb_split/validation.json"
RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume"
CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_04_score_0.6654.ckpt"
RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts"
PRODUCER="Find_1"
F2_SEMANTIC_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_semantic.json"
F2_GAUSSIAN_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_gaussian.json"
export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3.yaml"
# 两个 GPU 进程各自管理 26 个请求物化线程, BLAS 不再扩展子线程.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

stage1() {
    # 全部生产阶段统一通过 Stage1 V3 官方入口进入 Python CLI.
    bash "${STAGE1_ENTRY}" "$@"
}

run_two_gpu_shards() {
    # 按固定随机顺序把 PDB 划分为两个互斥子序列, 任一进程失败都使正式入口返回非零值.
    local first_exit=0
    local second_exit=0
    CUDA_VISIBLE_DEVICES=0 stage1 "$@" --shard-count 2 --shard-index 0 &
    local first_pid=$!
    CUDA_VISIBLE_DEVICES=1 stage1 "$@" --shard-count 2 --shard-index 1 &
    local second_pid=$!
    wait "${first_pid}" || first_exit=$?
    wait "${second_pid}" || second_exit=$?
    if ((first_exit != 0 || second_exit != 0)); then
        return 1
    fi
}

run_probability() {
    # validation 完整图概率使用与 calibration/test 相同的 checkpoint 与训练快照代码.
    local pdb_json="$1"
    local split="$2"
    run_two_gpu_shards probability \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}"
}

run_f2_blobs() {
    # 不访问 validation 标签; 直接应用 calibration 冻结的 F2 语义概率阈值.
    local pdb_json="$1"
    local split="$2"
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --semantic-parameters "${F2_SEMANTIC_PARAMETERS}"
}

run_centered() {
    # 来源体素数至少为 8 的候选进入前向; 超过 1,000 个 blob 时仍保留该 PDB.
    local pdb_json="$1"
    local split="$2"
    run_two_gpu_shards centered \
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

run_score_only() {
    # 只按冻结 Gaussian 参数替换 score/selected, 其他 centered 数组, 候选顺序和 offsets 保持不变.
    local pdb_json="$1"
    local split="$2"
    run_two_gpu_shards centered \
        --producer "${PRODUCER}" \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
        --score-only
}

run_score_only "${CALIBRATION_JSON}" calibration
run_probability "${VALIDATION_JSON}" validation
run_f2_blobs "${VALIDATION_JSON}" validation
run_centered "${VALIDATION_JSON}" validation
run_score_only "${VALIDATION_JSON}" validation
