#!/usr/bin/env bash

# 三种 unet_c1 采样模型的 F1 blobs+basic 校准与 held-out 测试入口.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
STAGE1_ENTRY="${PROJECT_ROOT}/训练与运行/sh/infer/stage1_v3.sh"
DATA_ROOT="/storage/penghongen/AdaLigand/Ori_Data"
CALIBRATION_JSON="${DATA_ROOT}/stage1_preparation_box_pool_3/split/pdb_split/calibration.json"
TEST_JSON="/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json"
PRODUCER="unet_c1"
TEST_SPLIT="held_out_test_0"

stage1() {
    # 只通过五阶段官方脚本进入 Python CLI.
    bash "${STAGE1_ENTRY}" "$@"
}

select_best_top_checkpoint() {
    # 按文件名中的浮点 score 选择当前训练目录内数值最高的 TOP checkpoint.
    find "$1/checkpoints" -maxdepth 1 -type f -name 'TOP_epoch_*_score_*.ckpt' -print \
        | awk -F'_score_' '{score=$2; sub(/[.]ckpt$/, empty, score); print score, $0}' \
        | sort -nr \
        | sed -n '1p' \
        | cut -d' ' -f2-
}

run_probability() {
    # 使用当前模式已经冻结的 checkpoint、训练配置和资源配置生成完整图概率.
    local pdb_json="$1"
    local split="$2"
    stage1 probability \
        --producer "${PRODUCER}" \
        --checkpoint "${CHECKPOINT}" \
        --resolved-config "${RESOLVED_CONFIG}" \
        --model-code-source training_snapshot \
        --pdb-json "${pdb_json}" \
        --split "${split}" \
        --output-root "${OUTPUT_ROOT}"
}

run_test() {
    # 复用 F1 semantic/basic 参数，在固定 held-out 清单生成 blobs 与评估.
    run_probability "${TEST_JSON}" "${TEST_SPLIT}"
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${TEST_JSON}" \
        --split "${TEST_SPLIT}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 1 \
        --semantic-parameters "${OUTPUT_ROOT}/${PRODUCER}/tuning/F1_semantic.json"
    stage1 evaluate \
        --producer "${PRODUCER}" \
        --pdb-json "${TEST_JSON}" \
        --split "${TEST_SPLIT}" \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 1 \
        --artifact blobs \
        --evaluation-name f1_blobs_basic_macro_selected \
        --selection-parameters "${OUTPUT_ROOT}/${PRODUCER}/tuning/F1_basic.json" \
        --data-root "${DATA_ROOT}"
}

run_calibration() {
    # 在固定 100-PDB calibration 清单独立拟合 F1 semantic 与 basic 参数.
    run_probability "${CALIBRATION_JSON}" calibration
    stage1 blobs \
        --producer "${PRODUCER}" \
        --pdb-json "${CALIBRATION_JSON}" \
        --split calibration \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 1 \
        --fit-semantic \
        --data-root "${DATA_ROOT}"
    stage1 tune \
        --producer "${PRODUCER}" \
        --pdb-json "${CALIBRATION_JSON}" \
        --split calibration \
        --output-root "${OUTPUT_ROOT}" \
        --alpha 1 \
        --objective-beta 1 \
        --score-mode basic \
        --prefiltered-min-voxel 8 \
        --data-root "${DATA_ROOT}"
}

MODE="${1:-}"
case "${MODE}" in
    occurrence)
        RUN_ROOT="/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal"
        CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_00_score_0.6030.ckpt"
        RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
        OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950/artifacts"
        export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3_a100_16cpu.yaml"
        run_test
        ;;
    pdb_centric_1)
        RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal"
        CHECKPOINT="${RUN_ROOT}/checkpoints/TOP_epoch_01_score_0.4918.ckpt"
        RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
        OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v1/artifacts"
        export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3_a800_24cpu.yaml"
        run_calibration
        run_test
        ;;
    pdb_centric_2)
        RUN_ROOT="/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal"
        CHECKPOINT="$(select_best_top_checkpoint "${RUN_ROOT}")"
        RESOLVED_CONFIG="${RUN_ROOT}/config.yaml"
        OUTPUT_ROOT="/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts"
        export STAGE1_INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference/stage1_v3_a800_24cpu.yaml"
        run_calibration
        run_test
        ;;
    *)
        echo "用法: $0 occurrence|pdb_centric_1|pdb_centric_2" >&2
        exit 2
        ;;
esac
