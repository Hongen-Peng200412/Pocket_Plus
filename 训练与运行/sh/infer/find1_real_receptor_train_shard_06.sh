#!/usr/bin/env bash

# 使用冻结的 Find_1 F2 语义阈值和 Gaussian 参数, 为训练 PDB 的用户第 6 片生成 Stage2 和 Stage3 输入.
# 唯一合法位置参数是用户片号 6. 传给 CLI 前严格减一, 因此四个阶段均使用 --shard-index 5.
# probability、F2 blobs 和不带 --score-only 的 centered 完整前向按完成标记续跑. score-only 幂等增加或替换 F2_centered.npz 中的 score/selected.
# 每个 PDB 的主要产物依次为 probability/probability_map.npz、blobs/F2_blobs.npz 和 centered/F2_centered.npz.
# 产物根为 OUTPUT_ROOT/Find_1/train/<pdb_id>. 本入口不调参、不评估, 也不使用 overwrite.
set -euo pipefail

if (($# != 1)) || [[ "$1" != "6" ]]; then
    echo "[find1_train_shard_06][错误] 唯一合法参数是用户片号 6。" >&2
    exit 2
fi

# SHARD_NUMBER: int, 该脚本固定接受的 1-based 用户片号.
SHARD_NUMBER=6
# SHARD_INDEX: int, 传给官方 CLI 的 0-based 固定分片下标.
SHARD_INDEX=$((SHARD_NUMBER - 1))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# PROJECT_ROOT: str, 当前脚本所在的 Pocket Plus 冻结代码发布目录绝对路径. 不允许 TASK_PROJECT_ROOT 覆盖.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"
TRAIN_JSON="/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/train.json"
RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume"
CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_04_score_0.6654.ckpt"
RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts"
PRODUCER="Find_1"
F2_SEMANTIC_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_semantic.json"
F2_GAUSSIAN_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_gaussian.json"
export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3_h100_32cpu.yaml"
# probability 和 centered 使用配置中的 26 个请求物化线程. BLAS 不再扩展子线程.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

verify_sha256() {
    # 两个位置参数依次是预期 SHA-256 和只读输入路径. 身份不一致时在任何产物续写前终止.
    local expected_sha256="$1"
    local input_path="$2"
    if ! printf '%s  %s\n' "${expected_sha256}" "${input_path}" | sha256sum --check --status -; then
        echo "[find1_train_shard_06][错误] 冻结输入身份不一致：${input_path}" >&2
        exit 2
    fi
}

# 五个外部文件不会随代码发布目录锁定, 因此每次运行都在正式四阶段之前核验文件内容的 SHA-256.
verify_sha256 "8e7f975ea49ee94e6819f2b35bf596c9bc4abacac9afc0aaebe2018b90d94d00" "${TRAIN_JSON}"
verify_sha256 "3f5dd715da76a2337ddb94017f442783798566cac584b8dd16882d8afd792ee4" "${CHECKPOINT}"
verify_sha256 "9a4ea5cf95d50deb3d42c8420e4dddf1d3a7eb0ebb6d0ee4edbff0388f1884da" "${RESOLVED_CONFIG}"
verify_sha256 "ac9d4b058700906db8a5c74a46094422b87ba684197c84af20b23ee46000ffd2" "${F2_SEMANTIC_PARAMETERS}"
verify_sha256 "6f116ba857b578aa5a70bade8af9fa6e47790a7ad55020636194747e1a062f23" "${F2_GAUSSIAN_PARAMETERS}"

# probability 复用冻结 checkpoint 和训练代码快照, 为该片生成完整图配体概率.
CUDA_VISIBLE_DEVICES=0 bash "${STAGE1_ENTRY}" probability \
    --producer "${PRODUCER}" \
    --checkpoint "${CHECKPOINT}" \
    --resolved-config "${RESOLVED_CONFIG}" \
    --model-code-source training_snapshot \
    --pdb-json "${TRAIN_JSON}" \
    --split train \
    --output-root "${OUTPUT_ROOT}" \
    --shard-count 50 \
    --shard-index "${SHARD_INDEX}"

# blobs 仅应用 calibration 冻结的 F2 语义阈值, 不重新拟合语义阈值.
bash "${STAGE1_ENTRY}" blobs \
    --producer "${PRODUCER}" \
    --pdb-json "${TRAIN_JSON}" \
    --split train \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --semantic-parameters "${F2_SEMANTIC_PARAMETERS}" \
    --shard-count 50 \
    --shard-index "${SHARD_INDEX}"

# centered 只前向来源体素数至少为 8 的 F2 blob. 当前 PDB 的来源 blob 数量严格大于 1,000 时保留 _BLOB_EXCEED 标记并继续生成 centered.
CUDA_VISIBLE_DEVICES=0 bash "${STAGE1_ENTRY}" centered \
    --producer "${PRODUCER}" \
    --checkpoint "${CHECKPOINT}" \
    --resolved-config "${RESOLVED_CONFIG}" \
    --model-code-source training_snapshot \
    --pdb-json "${TRAIN_JSON}" \
    --split train \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --forward-min-voxels 8 \
    --continue-on-blob-exceed \
    --shard-count 50 \
    --shard-index "${SHARD_INDEX}"

# score-only 复用已有 centered/F2_centered.npz, 以 calibration 冻结参数原子替换其中的候选 score 和 selected, 其他字段保持不变.
CUDA_VISIBLE_DEVICES=0 bash "${STAGE1_ENTRY}" centered \
    --producer "${PRODUCER}" \
    --pdb-json "${TRAIN_JSON}" \
    --split train \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
    --score-only \
    --shard-count 50 \
    --shard-index "${SHARD_INDEX}"
