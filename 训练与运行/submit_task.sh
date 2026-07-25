#!/usr/bin/env bash
set -euo pipefail

# 通用 Slurm 提交入口。它只把任务脚本与人类可读的资源参数交给 sbatch；
# release 不在提交时创建，而在 allocation 内每次真正执行 run_cmd 前创建。
#
# 默认发布源是本脚本上一层的项目根目录。需要发布另一份同结构项目时，可用
# --release-source 覆盖；脚本内部没有写死 Pocket_Plus 或任何本机绝对路径。

usage() {
    cat <<'EOF'
用法：
  bash 训练与运行/submit_task.sh \
    --sh Find_1.sh \
    --resource h100 \
    --gpus 2 \
    --cpus 48 \
    [资源选项] \
    [-- 传给任务脚本的参数]

必需参数：
  --sh PATH              任务脚本。只有文件名时从“训练与运行/sh/”查找；
                         带相对目录时从“训练与运行/”解析。
  --resource TYPE        cpu、a100、a800、h100 或 h200。

常用资源选项：
  --gpus N               每个节点的 GPU 数；CPU 任务不填写。
  --cpus N               每个 Slurm task 的 CPU 核数；不填写时按硬件类型推导。
  --nodes N              节点数，默认 1。
  --array SPEC           原样传给 Slurm，例如 0-11%4。
  --hold                 allocation 启动后创建 pre_lock，等待人工删除再执行。
  --job-name NAME        Slurm 作业名；默认使用任务脚本文件名。

高级选项：
  --partition NAME       覆盖硬件类型对应的默认 partition。
  --qos NAME             覆盖硬件类型对应的默认 QOS。
  --nodelist NAME        显式请求节点，例如 hnode01。
  --mem VALUE            原样传给 sbatch --mem。
  --time VALUE           原样传给 sbatch --time。
  --feedback-root PATH   覆盖反馈根目录；默认是 $HOME/Feedback/<发布源目录名>。
  --release-source PATH  覆盖每次执行时冻结的项目目录。相对路径从本脚本默认项目
                         根目录解析；默认就是本脚本上一层。

示例：
  bash 训练与运行/submit_task.sh --sh Find_1.sh --resource h100 --gpus 2 --cpus 48
  bash 训练与运行/submit_task.sh --sh unet_c1.sh --resource h100 --gpus 1 --cpus 24
  bash 训练与运行/submit_task.sh --sh ../其他任务.sh --resource cpu --cpus 16 --array 0-9
EOF
}

fail() {
    printf '[submit_task][错误] %s\n' "$*" >&2
    exit 2
}

is_positive_integer() {
    [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

task_argument=""
resource_type=""
gpu_count=""
cpu_count=""
node_count=1
array_spec=""
hold_mode=0
job_name=""
partition=""
qos=""
nodelist=""
memory_request=""
time_request=""
feedback_root=""
release_source_argument=""
task_arguments=()

while (($# > 0)); do
    case "$1" in
        --sh)
            (($# >= 2)) || fail "--sh 缺少任务脚本"
            task_argument="$2"
            shift 2
            ;;
        --resource)
            (($# >= 2)) || fail "--resource 缺少硬件类型"
            resource_type="${2,,}"
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || fail "--gpus 缺少数量"
            gpu_count="$2"
            shift 2
            ;;
        --cpus)
            (($# >= 2)) || fail "--cpus 缺少数量"
            cpu_count="$2"
            shift 2
            ;;
        --nodes)
            (($# >= 2)) || fail "--nodes 缺少数量"
            node_count="$2"
            shift 2
            ;;
        --array)
            (($# >= 2)) || fail "--array 缺少 Slurm 数组表达式"
            array_spec="$2"
            shift 2
            ;;
        --hold)
            hold_mode=1
            shift
            ;;
        --job-name)
            (($# >= 2)) || fail "--job-name 缺少名称"
            job_name="$2"
            shift 2
            ;;
        --partition)
            (($# >= 2)) || fail "--partition 缺少名称"
            partition="$2"
            shift 2
            ;;
        --qos)
            (($# >= 2)) || fail "--qos 缺少名称"
            qos="$2"
            shift 2
            ;;
        --nodelist)
            (($# >= 2)) || fail "--nodelist 缺少节点名"
            nodelist="$2"
            shift 2
            ;;
        --mem)
            (($# >= 2)) || fail "--mem 缺少值"
            memory_request="$2"
            shift 2
            ;;
        --time)
            (($# >= 2)) || fail "--time 缺少值"
            time_request="$2"
            shift 2
            ;;
        --feedback-root)
            (($# >= 2)) || fail "--feedback-root 缺少目录"
            feedback_root="$2"
            shift 2
            ;;
        --release-source)
            (($# >= 2)) || fail "--release-source 缺少项目目录"
            release_source_argument="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            task_arguments=("$@")
            break
            ;;
        *)
            fail "未知参数：$1"
            ;;
    esac
done

[[ -n "${task_argument}" ]] || fail "必须提供 --sh"
[[ -n "${resource_type}" ]] || fail "必须提供 --resource"
is_positive_integer "${node_count}" || fail "--nodes 必须是正整数"

# --release-source 没有填写时，发布源就是 submit_task.sh 所在项目。
# 相对覆盖路径从默认项目根目录解析，使行为不依赖调用命令时的工作目录。
if [[ -z "${release_source_argument}" ]]; then
    release_source_root="${DEFAULT_PROJECT_ROOT}"
elif [[ "${release_source_argument}" == /* ]]; then
    release_source_root="$(
        cd "${release_source_argument}" 2>/dev/null && pwd -P
    )" || fail "发布源不存在：${release_source_argument}"
else
    release_source_root="$(
        cd "${DEFAULT_PROJECT_ROOT}/${release_source_argument}" 2>/dev/null && pwd -P
    )" || fail "发布源不存在：${DEFAULT_PROJECT_ROOT}/${release_source_argument}"
fi

project_name="$(basename "${release_source_root}")"
feedback_root="${feedback_root:-${HOME}/Feedback/${project_name}}"
runner_root="${release_source_root}/训练与运行"
live_sbatch="${runner_root}/sbatch/task.sbatch"
runtime_dir="${release_source_root}/与服务器交互/other/training_runtime"

[[ -f "${live_sbatch}" ]] || fail "发布源中缺少通用 sbatch：${live_sbatch}"
[[ -f "${runtime_dir}/allocation_runner.sh" ]] \
    || fail "发布源中缺少 allocation 执行器：${runtime_dir}/allocation_runner.sh"
[[ -x "${runtime_dir}/create_release.sh" ]] \
    || fail "发布源中缺少可执行的 release 工具：${runtime_dir}/create_release.sh"

# 任务脚本只接受项目内相对路径。只有文件名时默认位于“训练与运行/sh/”；
# 带目录时从“训练与运行/”解析，例如 ../推理/某任务.sh。
[[ "${task_argument}" != /* ]] || fail "--sh 只接受相对路径"
if [[ "${task_argument}" == */* ]]; then
    task_candidate="${runner_root}/${task_argument}"
else
    task_candidate="${runner_root}/sh/${task_argument}"
fi
task_parent="$(cd "$(dirname "${task_candidate}")" 2>/dev/null && pwd -P)" \
    || fail "任务脚本父目录不存在：${task_candidate}"
task_path="${task_parent}/$(basename "${task_candidate}")"
[[ -f "${task_path}" ]] || fail "任务脚本不存在：${task_path}"
case "${task_path}" in
    "${release_source_root}/"*) ;;
    *) fail "任务脚本必须位于发布源内：${task_path}" ;;
esac
task_relative_path="${task_path#"${release_source_root}/"}"

# 资源映射来自项目中稳定使用的 CPU/A100/A800/H100/H200 模板。
# --cpus 显式给出时优先；否则采用每种硬件的常用默认值。
case "${resource_type}" in
    cpu)
        [[ -z "${gpu_count}" || "${gpu_count}" == "0" ]] \
            || fail "CPU 任务不能申请 --gpus"
        gpu_count=0
        partition="${partition:-cpu}"
        qos="${qos:-Cpu96}"
        cpu_count="${cpu_count:-16}"
        ;;
    a100)
        is_positive_integer "${gpu_count}" || fail "a100 必须提供正整数 --gpus"
        partition="${partition:-a100}"
        qos="${qos:-a100g2}"
        cpu_count="${cpu_count:-$((16 * gpu_count))}"
        ;;
    a800)
        is_positive_integer "${gpu_count}" || fail "a800 必须提供正整数 --gpus"
        partition="${partition:-nvlink}"
        qos="${qos:-nvlinkg8}"
        cpu_count="${cpu_count:-$((24 * gpu_count))}"
        ;;
    h100)
        is_positive_integer "${gpu_count}" || fail "h100 必须提供正整数 --gpus"
        partition="${partition:-h100}"
        qos="${qos:-h100g2}"
        cpu_count="${cpu_count:-$((24 * gpu_count))}"
        ;;
    h200)
        is_positive_integer "${gpu_count}" || fail "h200 必须提供正整数 --gpus"
        partition="${partition:-h200}"
        qos="${qos:-h200g2}"
        cpu_count="${cpu_count:-$((24 * gpu_count))}"
        ;;
    *)
        fail "--resource 仅支持 cpu、a100、a800、h100、h200"
        ;;
esac
is_positive_integer "${cpu_count}" || fail "--cpus 必须是正整数"

job_name="${job_name:-$(basename "${task_path}" .sh)}"
sbatch_arguments=(
    "--job-name=${job_name}"
    "--partition=${partition}"
    "--qos=${qos}"
    "--nodes=${node_count}"
    "--ntasks-per-node=1"
    "--cpus-per-task=${cpu_count}"
    "--output=/dev/null"
    "--error=/dev/null"
)
if [[ "${resource_type}" != "cpu" ]]; then
    sbatch_arguments+=("--gres=gpu:${resource_type}:${gpu_count}")
fi
[[ -z "${array_spec}" ]] || sbatch_arguments+=("--array=${array_spec}")
[[ -z "${nodelist}" ]] || sbatch_arguments+=("--nodelist=${nodelist}")
[[ -z "${memory_request}" ]] || sbatch_arguments+=("--mem=${memory_request}")
[[ -z "${time_request}" ]] || sbatch_arguments+=("--time=${time_request}")

printf '[submit_task] 发布源：%s\n' "${release_source_root}"
printf '[submit_task] 任务：%s\n' "${task_relative_path}"
printf '[submit_task] 反馈根：%s\n' "${feedback_root}"
printf '[submit_task] 资源：type=%s nodes=%s gpus_per_node=%s cpus_per_task=%s\n' \
    "${resource_type}" "${node_count}" "${gpu_count}" "${cpu_count}"
printf '[submit_task] 此刻不创建 release；每次实际执行前才冻结发布源。\n'

# SBATCH_BIN 只供无卡测试替换成假 sbatch；正式使用时默认为系统 sbatch。
sbatch_program="${SBATCH_BIN:-sbatch}"
"${sbatch_program}" "${sbatch_arguments[@]}" "${live_sbatch}" \
    --source-root "${release_source_root}" \
    --feedback-root "${feedback_root}" \
    --task "${task_relative_path}" \
    --hold "${hold_mode}" \
    --resource "${resource_type}" \
    --gpus "${gpu_count}" \
    --nodes "${node_count}" \
    --cpus "${cpu_count}" \
    --array "${array_spec}" \
    -- "${task_arguments[@]}"
