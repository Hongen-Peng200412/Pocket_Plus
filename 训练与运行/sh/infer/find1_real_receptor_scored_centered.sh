#!/usr/bin/env bash

# 使用 tuning/F2_semantic.json 与 tuning/F2_gaussian.json 中已冻结的 Find_1 参数生成 Stage2, Stage3 输入.
# calibration 在 OUTPUT_ROOT/PRODUCER/calibration/<pdb_id>/centered/F2_centered.npz 的同一正式路径原子替换 score/selected, 不执行模型前向.
# validation 在 OUTPUT_ROOT/PRODUCER/validation/<pdb_id>/ 下生成 probability/probability_map.npz, blobs/F2_blobs.npz 和 centered/F2_centered.npz.
# 本入口不重新调参, 不生成 calibration 或 validation 评估指标.
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
# probability 与正常 centered 的两个 GPU 进程各自管理 26 个请求物化线程, BLAS 不再扩展子线程.
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

# N_candidate 是当前 PDB 的 F2_centered.npz 中已执行 centered 前向的候选数.
# source_blob_index: int32, (N_candidate,), 数组顺序定义 centered 候选轴; 每个值索引同一 PDB F2_blobs.npz 中以 blob_index 为代表的候选级数组第一维.
# voxel_offsets: int64, (N_candidate + 1,), voxel_offsets[i]:voxel_offsets[i + 1] 是第 i 个 centered 候选在 voxel_index_local_zyx, source_probability, centered_probability 和 voxel_final 中的同步半开区间; 首值为 0, 末值为全部候选局部体素总数.
# voxel_aux_offsets: int64, (N_candidate + 1,), voxel_aux_offsets[i]:voxel_aux_offsets[i + 1] 是第 i 个 centered 候选在 voxel_aux_index_local_zyx 和 voxel_aux_probability 中的同步半开区间; 首值为 0, 末值为全部候选受体占据体素总数.
# A_offsets: int64, (N_candidate + 1,), A_offsets[i]:A_offsets[i + 1] 是第 i 个 centered 候选在 A_global_index, A_coord_local_xyz, A_coord_centered_world, A_probability 和 A_feat_L0/L1/L2/L3 中的同步半开区间; 首值为 0, 末值为全部候选 A 原子总数.
# P_offsets: int64, (N_candidate + 1,), P_offsets[i]:P_offsets[i + 1] 是第 i 个 centered 候选在 P_coord_local_xyz, P_probability 和 P_feat_L2/L3 中的同步半开区间; 首值为 0, 末值为全部候选 P 锚点总数.
# score: float32, (N_candidate,), 冻结 Gaussian 参数计算的逐候选分数.
# selected: bool, (N_candidate,), True 表示同时达到冻结 score_threshold, prefiltered_min_voxel 和 min_voxels.
run_two_gpu_shards centered \
    --producer "${PRODUCER}" \
    --pdb-json "${CALIBRATION_JSON}" \
    --split calibration \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
    --score-only

# validation 完整图概率使用与第一阶段 calibration 和 held_out_test_0 相同的 checkpoint 与训练快照代码.
run_two_gpu_shards probability \
    --producer "${PRODUCER}" \
    --checkpoint "${CHECKPOINT}" \
    --resolved-config "${RESOLVED_CONFIG}" \
    --model-code-source training_snapshot \
    --pdb-json "${VALIDATION_JSON}" \
    --split validation \
    --output-root "${OUTPUT_ROOT}"

# 不访问 validation 标签; 直接应用 calibration 冻结的 F2 语义概率阈值.
stage1 blobs \
    --producer "${PRODUCER}" \
    --pdb-json "${VALIDATION_JSON}" \
    --split validation \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --semantic-parameters "${F2_SEMANTIC_PARAMETERS}"

# 来源体素数至少为 8 的 F2 blob 候选进入前向; 超过 1,000 个 blob 时仍保留该 PDB.
run_two_gpu_shards centered \
    --producer "${PRODUCER}" \
    --checkpoint "${CHECKPOINT}" \
    --resolved-config "${RESOLVED_CONFIG}" \
    --model-code-source training_snapshot \
    --pdb-json "${VALIDATION_JSON}" \
    --split validation \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --forward-min-voxels 8 \
    --continue-on-blob-exceed

# validation 使用与 calibration 相同的冻结 Gaussian 参数, 只向 F2_centered.npz 写入上述 score/selected.
run_two_gpu_shards centered \
    --producer "${PRODUCER}" \
    --pdb-json "${VALIDATION_JSON}" \
    --split validation \
    --output-root "${OUTPUT_ROOT}" \
    --alpha 2 \
    --selection-parameters "${F2_GAUSSIAN_PARAMETERS}" \
    --score-only
