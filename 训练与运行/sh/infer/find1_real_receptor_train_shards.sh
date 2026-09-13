#!/usr/bin/env bash

# 使用冻结的 Find_1 F2 语义阈值和 Gaussian 参数, 为训练 PDB 的指定 50 分片生成 Stage2 和 Stage3 输入.
# 四个位置参数依次表示 allocation 内 CUDA device 0 的两个用户片号和 device 1 的两个用户片号. 用户片号 1..50 传给 CLI 前严格减一.
# probability, F2 blobs 和正常 centered 按完成标记跳过; score-only 幂等重算 score/selected 并原子重发同一 F2_centered.npz.
# 产物写入 OUTPUT_ROOT/Find_1/train/<pdb_id>, 主要文件为 probability_map.npz, F2_blobs.npz 和 F2_centered.npz. 本入口不调参, 不评估, 也不使用 overwrite.
set -euo pipefail

if (($# != 4)); then
    echo "[find1_train_shards][错误] 必须依次传入 GPU 0 的两个片号和 GPU 1 的两个片号。" >&2
    exit 2
fi

# SHARD_NUMBERS: 长度 4 的 Bash 字符串数组, 每项经校验后表示用户片号 1..50; 前两项属于 allocation 内 CUDA device 0, 后两项属于 device 1.
SHARD_NUMBERS=("$1" "$2" "$3" "$4")
for shard_number in "${SHARD_NUMBERS[@]}"; do
    if [[ ! "${shard_number}" =~ ^([1-9]|[1-4][0-9]|50)$ ]]; then
        echo "[find1_train_shards][错误] 片号必须是 1..50 的整数：${shard_number}" >&2
        exit 2
    fi
done
for ((first_index = 0; first_index < 4; first_index++)); do
    for ((second_index = first_index + 1; second_index < 4; second_index++)); do
        if [[ "${SHARD_NUMBERS[first_index]}" == "${SHARD_NUMBERS[second_index]}" ]]; then
            echo "[find1_train_shards][错误] 四个片号不得重复。" >&2
            exit 2
        fi
    done
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"
TRAIN_JSON="/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/train.json"
RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume"
CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_04_score_0.6654.ckpt"
RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts"
PRODUCER="Find_1"
F2_SEMANTIC_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_semantic.json"
F2_GAUSSIAN_PARAMETERS="${OUTPUT_ROOT}/${PRODUCER}/tuning/F2_gaussian.json"
export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3.yaml"
# probability 与 centered 的两个 GPU 进程各自使用 26 个请求物化线程; BLAS 不再扩展子线程.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

stage1() {
    # 把调用点给出的 Stage1 子命令和参数原样传给官方 Shell 入口.
    bash "${STAGE1_ENTRY}" "$@"
}

wait_for_pair() {
    # 两个位置参数是同一阶段的后台进程 PID. 本函数等待两者结束, 仅在两者都成功时返回 0.
    local first_pid="$1"
    local second_pid="$2"
    local first_exit=0
    local second_exit=0
    wait "${first_pid}" || first_exit=$?
    wait "${second_pid}" || second_exit=$?
    if ((first_exit != 0 || second_exit != 0)); then
        return 1
    fi
}

run_shard_pair() {
    # 两个位置参数分别是本轮绑定到 allocation 内 CUDA device 0 和 1 的用户片号. 四个阶段之间设置屏障, 两个 blobs 子任务串行执行.
    # 本函数向 OUTPUT_ROOT/Find_1/train/<pdb_id> 发布三类 NPZ 和完成标记, 不拟合参数或计算评估指标.
    local gpu0_shard_number="$1"
    local gpu1_shard_number="$2"
    # gpu0_shard_index: int, device 0 本轮使用的 0-based CLI 分片下标.
    local gpu0_shard_index=$((gpu0_shard_number - 1))
    # gpu1_shard_index: int, device 1 本轮使用的 0-based CLI 分片下标.
    local gpu1_shard_index=$((gpu1_shard_number - 1))
    local gpu0_pid
    local gpu1_pid

    # probability 复用冻结 checkpoint 和训练代码快照, 为两片独立生成完整图配体概率.
    CUDA_VISIBLE_DEVICES=0 stage1 probability \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --shard-count 50 \
        --shard-index "${gpu0_shard_index}" &
    gpu0_pid=$!
    CUDA_VISIBLE_DEVICES=1 stage1 probability \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --shard-count 50 \
        --shard-index "${gpu1_shard_index}" &
    gpu1_pid=$!
    wait_for_pair "${gpu0_pid}" "${gpu1_pid}"

    # 两片 blobs 串行复用 56 个 CPU worker, 避免同节点同时建立 112 个线程.
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --semantic-parameters "${F2_SEMANTIC_PARAMETERS}" \
        --shard-count 50 \
        --shard-index "${gpu0_shard_index}"
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --semantic-parameters "${F2_SEMANTIC_PARAMETERS}" \
        --shard-count 50 \
        --shard-index "${gpu1_shard_index}"

    # centered 只前向来源体素数至少为 8 的 F2 blob; 超大 blob 保留标识并继续处理.
    CUDA_VISIBLE_DEVICES=0 stage1 centered \
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
        --shard-index "${gpu0_shard_index}" &
    gpu0_pid=$!
    CUDA_VISIBLE_DEVICES=1 stage1 centered \
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
        --shard-index "${gpu1_shard_index}" &
    gpu1_pid=$!
    wait_for_pair "${gpu0_pid}" "${gpu1_pid}"

    # score-only 复用 centered 数组, 以 calibration 冻结参数原子补充候选 score 和 selected.
    CUDA_VISIBLE_DEVICES=0 stage1 centered \
        --producer "${PRODUCER}" \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
        --score-only \
        --shard-count 50 \
        --shard-index "${gpu0_shard_index}" &
    gpu0_pid=$!
    CUDA_VISIBLE_DEVICES=1 stage1 centered \
        --producer "${PRODUCER}" \
        --pdb-json "${TRAIN_JSON}" \
        --split train \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 2 \
        --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
        --score-only \
        --shard-count 50 \
        --shard-index "${gpu1_shard_index}" &
    gpu1_pid=$!
    wait_for_pair "${gpu0_pid}" "${gpu1_pid}"
}

# 第一轮为 device 0 的第一个片号和 device 1 的第一个片号; 第二轮保持各 device 顺序进入下一片.
run_shard_pair "${SHARD_NUMBERS[0]}" "${SHARD_NUMBERS[2]}"
run_shard_pair "${SHARD_NUMBERS[1]}" "${SHARD_NUMBERS[3]}"
