#!/bin/bash
# ============================================================
# A100 单卡训练 (1 GPU, 16 核 CPU)
# ============================================================
# 用法 (推荐): sbatch sbatch/a100/1gpu.sbatch "+experiment=exp001_baseline"
# 用法 (旧配置): sbatch sbatch/a100/1gpu.sbatch CONFIG_NAME ["HYDRA_OVERRIDES"]
# ============================================================
#SBATCH -o /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.out
#SBATCH -e /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/%x_%j.err
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

SBATCH_ROOT="${POCKET_PLUS_SBATCH_ROOT:-/home/penghongen/My_Project/Pocket_Plus/sbatch}"
export POCKET_PLUS_LOCK_PRE_ENABLED="${POCKET_PLUS_LOCK_PRE_ENABLED:-0}"
export POCKET_PLUS_LOCK_AFTER_ENABLED="${POCKET_PLUS_LOCK_AFTER_ENABLED:-1}"
export POCKET_PLUS_TRY_AFTER_END_ENABLED="${POCKET_PLUS_TRY_AFTER_END_ENABLED:-1}"
source "${SBATCH_ROOT}/_common.sh"




# ============================================================
# [高级选项 / Advanced Override]
# 如果你想改变这个配置卡的默认执行逻辑（比如不跑 train.py 而去跑验证），
# 你可以取消注释下面的这个 Bash 函数，_train_core 将优先使用它来生成指令。
# 在这个函数中，你可以直接使用已经被 _train_core 智能解析过的变量，如: 
#   - $CONFIG_NAME       : Hydra 读取到的最终配置名
#   - $HYDRA_OVERRIDES   : 额外的 Hydra 参数组合
#   - $NNODES / $DEVICES : 节点分配信息
# ============================================================
# write_custom_run_cmd_hook() {
#     local TARGET_FILE="$1"
#     cat << EOF > "${TARGET_FILE}"
# #!/bin/bash
# # 这是一个由 Hook 自定义生成的指令文件, 想写什么就写什么, 也就是说 "--config=..." 不加也行; 一些自定义参数怎么加也行: "input=..."
# python Pocket_Plus/src/inference/run.py \\
#     --config="${CONFIG_NAME}" \\
#     ${HYDRA_OVERRIDES}
# EOF
# }
# ============================================================
write_custom_run_cmd_hook() {
    local TARGET_FILE="$1"
    cat << EOF > "${TARGET_FILE}"

#!/bin/bash
python /home/penghongen/My_Project/Pocket_Plus/src/inference/run.py --config=标准0_cryoatom_hard

EOF
}
# ============================================================
# write_custom_run_cmd_hook() {
#     local TARGET_FILE="$1"
#     cat << EOF > "${TARGET_FILE}"
# #!/bin/bash
# python Pocket_Plus/src/inference/run.py \\
#     --config=raw_single \\
#     ckpt_path="feedback_plus/logs/.../checkpoints/last.ckpt"
# EOF
# }




source "${SBATCH_ROOT}/_train_core.sh" "${1:-}" 1 1 "${2:-}"
