#!/usr/bin/env bash
set -euo pipefail

# 为一次实际 run_cmd 执行保存启动证据。launch 不保存训练产物；训练配置、
# checkpoint 和源码快照继续由 src/train.py 放入 logs/。

fail() {
    printf '[create_launch][错误] %s\n' "$*" >&2
    exit 2
}

[[ $# -eq 4 ]] || fail \
    "用法：create_launch.sh FEEDBACK_ROOT LAUNCH_ID RUN_CMD_FILE TASK_PATH"

feedback_root="$1"
launch_id="$2"
run_cmd_file="$3"
task_path="$4"
job_id="${SLURM_JOB_ID:?缺少 SLURM_JOB_ID}"
launch_directory="${feedback_root}/launches/${job_id}/${launch_id}"

[[ ! -e "${launch_directory}" ]] || fail "launch 已存在：${launch_directory}"
mkdir -p "${launch_directory}"
cp "${run_cmd_file}" "${launch_directory}/run_cmd.sh"
chmod a-w "${launch_directory}/run_cmd.sh" 2>/dev/null || true

json_escape() {
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

created_at="$(date --iso-8601=seconds)"
cat >"${launch_directory}/launch.json" <<EOF
{
  "schema_version": 1,
  "launch_id": "$(printf '%s' "${launch_id}" | json_escape)",
  "slurm_job_id": "$(printf '%s' "${job_id}" | json_escape)",
  "created_at": "$(printf '%s' "${created_at}" | json_escape)",
  "release_project_root": "$(printf '%s' "${TASK_PROJECT_ROOT}" | json_escape)",
  "task_script": "$(printf '%s' "${task_path}" | json_escape)",
  "task_run_stamp": "$(printf '%s' "${TASK_RUN_STAMP}" | json_escape)",
  "resource_type": "$(printf '%s' "${TASK_RESOURCE_TYPE}" | json_escape)",
  "nodes": ${TASK_NODE_COUNT},
  "gpus_per_node": ${TASK_GPUS_PER_NODE},
  "cpus_per_task": ${TASK_CPU_COUNT},
  "array_spec": "$(printf '%s' "${TASK_ARRAY_SPEC}" | json_escape)"
}
EOF

printf '%s\n' "${launch_directory}"
