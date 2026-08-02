#!/usr/bin/env bash
set -euo pipefail

# 通用 Slurm allocation wrapper。资源参数由 submit_task.sh 通过 sbatch 命令
# 传入，因此这里不固定任何 #SBATCH 硬件声明。
#
# Slurm 会在提交时保存本文件的内容，但不会保存 --task-root 指向的项目树。
# Job 真正启动时，本文件从任务根目录加载 allocation_runner.sh。完整模式会在每次
# run_cmd 开始前冻结当时最新的项目；simple 模式直接执行当前任务。

fail() {
    printf '[task.sbatch][错误] %s\n' "$*" >&2
    exit 2
}

task_root=""
feedback_root=""
task_path=""
hold_mode=0
simple_mode=0
resource_type=""
gpu_count=0
node_count=1
cpu_count=1
array_spec=""
task_arguments=()

while (($# > 0)); do
    case "$1" in
        --task-root) task_root="$2"; shift 2 ;;
        --feedback-root) feedback_root="$2"; shift 2 ;;
        --task) task_path="$2"; shift 2 ;;
        --simple) simple_mode="$2"; shift 2 ;;
        --hold) hold_mode="$2"; shift 2 ;;
        --resource) resource_type="$2"; shift 2 ;;
        --gpus) gpu_count="$2"; shift 2 ;;
        --nodes) node_count="$2"; shift 2 ;;
        --cpus) cpu_count="$2"; shift 2 ;;
        --array) array_spec="$2"; shift 2 ;;
        --)
            shift
            task_arguments=("$@")
            break
            ;;
        *) fail "未知参数：$1" ;;
    esac
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "本文件只能在 Slurm Job 内运行"
[[ -n "${task_root}" ]] || fail "缺少 --task-root"
[[ -n "${task_path}" ]] || fail "缺少 --task"
if [[ "${simple_mode}" == "0" ]]; then
    [[ -n "${feedback_root}" ]] || fail "完整模式缺少 --feedback-root"
fi

task_root="$(cd "${task_root}" 2>/dev/null && pwd -P)" \
    || fail "任务根目录不存在：${task_root}"
runtime_dir="${task_root}/训练与运行/runtime"
[[ -f "${runtime_dir}/allocation_runner.sh" ]] \
    || fail "任务根目录中缺少四锁执行器：${runtime_dir}/allocation_runner.sh"

if [[ "${simple_mode}" == "0" ]]; then
    allocation_root="${feedback_root}/allocations"
    job_directory="${allocation_root}/${SLURM_JOB_ID}"
    mkdir -p "${job_directory}"

    # 完整模式把 allocation 控制器与任务输出集中到 allocations/<job-id>/out 和 err。
    exec >>"${job_directory}/out" 2>>"${job_directory}/err"
fi

export TASK_ROOT="${task_root}"
export TASK_FEEDBACK_ROOT="${feedback_root}"
export TASK_PATH="${task_path}"
export TASK_HOLD_MODE="${hold_mode}"
export TASK_SIMPLE_MODE="${simple_mode}"
export TASK_RESOURCE_TYPE="${resource_type}"
export TASK_GPUS_PER_NODE="${gpu_count}"
export TASK_NODE_COUNT="${node_count}"
export TASK_CPU_COUNT="${cpu_count}"
export TASK_ARRAY_SPEC="${array_spec}"

# allocation_runner.sh 与本文件在同一个 shell 中运行，所以它能直接读取
# task_arguments 数组，不需要把带空格的参数压成不安全的字符串。
source "${runtime_dir}/allocation_runner.sh"
run_allocation
