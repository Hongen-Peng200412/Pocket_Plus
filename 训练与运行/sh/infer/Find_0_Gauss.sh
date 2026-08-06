#!/usr/bin/env bash
set -euo pipefail

# 为已经完成且当前没有生产者的 Find_0 F1-centered 产物回填 Gauss scorer 字段。
# 任务只读取现有概率图、组件、F1-centered 和冻结参数，不重新运行模型，也不需要 GPU。
# 尚未完成前置产物或正被 GPU 持有的 PDB 会被记录并跳过；以后用同一参数再次运行即可补齐。
# 默认用当前 calibration 参数强制刷新 `gauss_score` 与 `gauss_selected`；其他 forest 字段不变。
#
# 在 Pocket_Plus 服务器项目根目录提交一个 calibration CPU16 任务：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_Gauss.sh \
#   --resource cpu \
#   --cpus 16 \
#   --job-name find0_gauss
#
# validation 或 train 可以在 GPU 主线运行期间增量回填。先把 `target_split` 和
# `global_shard_count` 改成目标值，再用 `--array '0-N'` 提交需要处理的分片编号。
# 数组编号就是全局分片编号；可以只提交部分编号，之后再补交其余编号。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${TASK_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd -P)}"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

inference_root="/storage/penghongen/AdaLigand_stage1_inference"
li_inference_root="/storage/penghongen/AdaLigand_stage1_LI_inference"
run_name="Find_0-CPC1-ligand_PRAUC_0.675477"
centered_role="F1_centered"                 # 七个 F_alpha-centered 角色之一，或独立的 Li_centered。

if [[ "${centered_role}" == "Li_centered" ]]; then
    output_root="${li_inference_root}/${run_name}/artifacts"          # Li 不写入主线 forest，只回填自身 centered 文件。
else
    output_root="${inference_root}/${run_name}/artifacts"             # F_alpha 共用主线 forest；最后一次正式回填决定两个 Gauss 字段。
fi
if [[ "${centered_role}" == "F1_centered" ]]; then
    calibration_json="${output_root}/Find_0/gauss_scorer/calibration.json" # 历史 F1 正式参数路径保持不变。
else
    calibration_json="${output_root}/Find_0/gauss_scorer/${centered_role}/calibration.json" # 新策略按角色隔离参数。
fi

target_split="calibration"                   # 可选 calibration、validation 或 train；每次提交只处理一个数据划分。
global_shard_count=1                          # 所有分片编号合起来覆盖目标清单；train 可使用较大的固定分片数。
shard_index="${SLURM_ARRAY_TASK_ID:-0}"       # 数组编号就是分片编号；非数组任务固定处理分片 0。

case "${target_split}" in
    calibration|validation|train)
        pdb_list="${inference_root}/${target_split}_pdb_ids.json"
        ;;
    *)
        echo "target_split 必须是 calibration、validation 或 train" >&2
        exit 2
        ;;
esac

[[ -f "${pdb_list}" && -f "${calibration_json}" ]] || {
    echo "缺少 ${target_split} 清单或冻结 Gauss 参数" >&2
    exit 2
}

cd "${PROJECT_ROOT}"
python -u -m src.inference.Gauss_Scorer.cli \
    --pdb-list "${pdb_list}" \
    --output-root "${output_root}" \
    --producer Find_0 \
    --split "${target_split}" \
    --calibration-json "${calibration_json}" \
    --centered-role "${centered_role}" \
    --evaluate-on-blob-exceed \
    --force-overwrite \
    --shard-index "${shard_index}" \
    --shard-count "${global_shard_count}"
