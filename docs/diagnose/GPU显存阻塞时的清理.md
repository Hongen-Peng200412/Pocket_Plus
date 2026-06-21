# GPU 显存阻塞时的温和清理

适用场景：Slurm 任务已经失败并进入 `try_lock_JOBID`，报错里显示同一张 GPU 上残留了多个旧进程占用显存。目标是先确认 GPU 状态，只温和清理属于这个 Job 的残留进程，然后删除 `try_lock` 让任务重试。

下面命令默认从 Windows PowerShell 执行，通过登录节点 `10.102.33.220` 进入服务器。通常只需要改第一行 `$JOB_ID`；如果自动识别节点失败，再把 `$NODE` 改成 `gnode02` 这类节点名。

## 推荐：一键诊断、温和清理、重试

只改 `$JOB_ID`，必要时改 `$NODE`。

```powershell
$JOB_ID = "292666"
$NODE = ""
$SSH = "10.102.33.220"

$script = @'
set -u

JOB_ID="__JOB_ID__"
NODE="__NODE__"
USER_NAME="$(whoami)"
TRY_LOCK="/home/${USER_NAME}/try_lock_${JOB_ID}"
AFTER_LOCK="/home/${USER_NAME}/after_lock_${JOB_ID}"

echo "========== Job =========="
squeue -j "${JOB_ID}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" || true
echo

if [ -z "${NODE}" ]; then
  NODE="$(scontrol show job "${JOB_ID}" 2>/dev/null | awk -F= '/BatchHost=/{split($2,a," "); print a[1]; exit}')"
fi

if [ -z "${NODE}" ]; then
  echo "[STOP] 无法自动识别节点。请把 PowerShell 里的 NODE 变量改成 gnode02 这类节点名后重试。"
  exit 1
fi

echo "JOB_ID=${JOB_ID}"
echo "NODE=${NODE}"
echo "TRY_LOCK=${TRY_LOCK}"
echo "AFTER_LOCK=${AFTER_LOCK}"
echo

SRUN_NODE_ARGS=()
if [ -n "${NODE}" ]; then
  SRUN_NODE_ARGS=(-w "${NODE}")
fi

echo "========== Locks =========="
ls -l "${TRY_LOCK}" "${AFTER_LOCK}" 2>/dev/null || true
echo

if [ ! -e "${TRY_LOCK}" ]; then
  echo "[STOP] ${TRY_LOCK} 不存在：任务没有停在 try_lock，避免误清理正在运行的任务。"
  exit 0
fi

echo "========== GPU Before Cleanup =========="
srun --jobid="${JOB_ID}" --overlap "${SRUN_NODE_ARGS[@]}" -N1 -n1 bash <<'NODE_SCRIPT'
set -u

JOB_ID="${SLURM_JOB_ID:-}"
ME="$(whoami)"
SMI="/usr/bin/nvidia-smi"
if [ ! -x "${SMI}" ]; then
  SMI="$(command -v nvidia-smi || true)"
fi

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"

if [ -z "${SMI}" ]; then
  echo "[STOP] nvidia-smi 不可用。"
  exit 1
fi

"${SMI}" || true
echo

PIDS="$("${SMI}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -u)"
if [ -z "${PIDS}" ]; then
  echo "[OK] GPU 上没有 compute 进程。"
  exit 0
fi

echo "========== Candidate GPU Processes =========="
ps -o pid,ppid,user,stat,etime,args -p $(echo "${PIDS}" | tr '\n' ',') 2>/dev/null || true
echo

KILL_PIDS=""
for PID in ${PIDS}; do
  OWNER="$(ps -o user= -p "${PID}" 2>/dev/null | awk '{print $1}')"
  if [ "${OWNER}" != "${ME}" ]; then
    echo "[SKIP] PID ${PID}: owner=${OWNER}, not ${ME}"
    continue
  fi

  SAME_JOB=0
  if tr '\0' '\n' < "/proc/${PID}/environ" 2>/dev/null | grep -qx "SLURM_JOB_ID=${JOB_ID}"; then
    SAME_JOB=1
  elif grep -q "${JOB_ID}" "/proc/${PID}/cgroup" 2>/dev/null; then
    SAME_JOB=1
  fi

  if [ "${SAME_JOB}" != "1" ]; then
    echo "[SKIP] PID ${PID}: 属于当前用户，但无法确认是 JOB ${JOB_ID}，不自动清理。"
    continue
  fi

  KILL_PIDS="${KILL_PIDS} ${PID}"
done

if [ -z "${KILL_PIDS}" ]; then
  echo "[INFO] 没有可自动 TERM 的同 Job GPU 进程。"
else
  echo "[TERM] 温和终止同 Job GPU 进程:${KILL_PIDS}"
  kill -TERM ${KILL_PIDS} 2>/dev/null || true
  sleep 10
fi

echo
echo "========== GPU After TERM =========="
"${SMI}" || true
REMAIN="$("${SMI}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -u)"
if [ -n "${REMAIN}" ]; then
  echo "[STOP] TERM 后仍有 GPU 进程，先不删除 try_lock。剩余 PID:"
  ps -o pid,ppid,user,stat,etime,args -p $(echo "${REMAIN}" | tr '\n' ',') 2>/dev/null || true
  exit 2
fi

echo "[OK] GPU compute 进程已清空。"
NODE_SCRIPT

SRUN_CODE=$?
if [ "${SRUN_CODE}" != "0" ]; then
  echo "[STOP] 清理未确认成功，保留 ${TRY_LOCK}。"
  exit "${SRUN_CODE}"
fi

echo
echo "========== Retry =========="
rm -f "${TRY_LOCK}"
echo "[OK] 已删除 ${TRY_LOCK}，Slurm 脚本会自动重试。"
echo "[KEEP] ${AFTER_LOCK} 保留，用于继续占住 allocation。"
echo
echo "========== Recent Log =========="
tail -n 80 "/home/${USER_NAME}/My_Project/feedback_plus/logs/_temp_slurm/train_${JOB_ID}.out" 2>/dev/null || true
'@ -replace "__JOB_ID__", $JOB_ID -replace "__NODE__", $NODE

$script | ssh $SSH "bash -s"
```

## 只诊断，不清理

只改 `$JOB_ID`，必要时改 `$NODE`。这个命令不会删除 `try_lock`，也不会杀进程。

```powershell
$JOB_ID = "257059"
$NODE = ""
$SSH = "10.102.33.220"

$script = @'
set -u

JOB_ID="__JOB_ID__"
NODE="__NODE__"
USER_NAME="$(whoami)"

if [ -z "${NODE}" ]; then
  NODE="$(scontrol show job "${JOB_ID}" 2>/dev/null | awk -F= '/BatchHost=/{split($2,a," "); print a[1]; exit}')"
fi

SRUN_NODE_ARGS=()
if [ -n "${NODE}" ]; then
  SRUN_NODE_ARGS=(-w "${NODE}")
fi

echo "========== Job =========="
squeue -j "${JOB_ID}" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" || true
scontrol show job "${JOB_ID}" 2>/dev/null | sed -n "1,80p" || true
echo

echo "========== Locks =========="
ls -l "/home/${USER_NAME}/try_lock_${JOB_ID}" "/home/${USER_NAME}/after_lock_${JOB_ID}" 2>/dev/null || true
echo

echo "========== GPU / Processes =========="
srun --jobid="${JOB_ID}" --overlap "${SRUN_NODE_ARGS[@]}" -N1 -n1 bash <<'NODE_SCRIPT'
SMI="/usr/bin/nvidia-smi"
if [ ! -x "${SMI}" ]; then
  SMI="$(command -v nvidia-smi || true)"
fi

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<empty>}"
if [ -n "${SMI}" ]; then
  "${SMI}" || true
  PIDS="$("${SMI}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF{print $1}' | sort -u)"
  if [ -n "${PIDS}" ]; then
    echo
    ps -o pid,ppid,user,stat,etime,args -p $(echo "${PIDS}" | tr '\n' ',') 2>/dev/null || true
  fi
fi
NODE_SCRIPT
'@ -replace "__JOB_ID__", $JOB_ID -replace "__NODE__", $NODE

$script | ssh $SSH "bash -s"
```

## GPU 已经空了，只重试

如果诊断显示 `No running processes found`，或者 `--query-compute-apps` 没有任何 PID，可以只删除 `try_lock` 让原 Job 重跑。

```powershell
$JOB_ID = "257059"
$SSH = "10.102.33.220"

ssh $SSH "rm -f /home/penghongen/try_lock_${JOB_ID} && echo '[OK] removed /home/penghongen/try_lock_${JOB_ID}' && ls -l /home/penghongen/*lock*${JOB_ID}* 2>/dev/null || true"
```

## 结果怎么判断

* 看到 `[OK] GPU compute 进程已清空。` 和 `[OK] 已删除 ... try_lock_JOBID`：说明已经触发重试。
* 看到 `[STOP] ... 不存在`：说明任务没有停在 `try_lock`，脚本为了避免误伤没有操作。
* 看到 `[STOP] TERM 后仍有 GPU 进程`：说明温和清理没有完全成功；先不要重试，继续看剩余 PID 的命令行和归属。
* 重试后可以看日志是否出现新的 run stamp：`[Run] Local run stamp: ...`。

本次处理过的例子：`JOB_ID=257059` 在 `gnode07`，先确认 GPU 上无残留 compute 进程，然后删除 `/home/penghongen/try_lock_257059`，保留 `/home/penghongen/after_lock_257059`，任务随后重新启动。