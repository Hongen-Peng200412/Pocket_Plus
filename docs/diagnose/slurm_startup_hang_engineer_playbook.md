# Slurm Startup Hang Engineer Playbook

This playbook is for engineers or AI agents debugging Pocket Plus Slurm jobs
that appear to be `RUNNING` but make no useful progress during Python startup.
It preserves the practical path we used, including failed routes, so the same
problem can be diagnosed again without rediscovering the traps.

## Final Diagnosis

The startup hang was not a deterministic project-code exception. The strongest
evidence came from `-X importtime` on job `245087`: even Python standard-library
and NumPy imports were taking seconds to tens of seconds while the job was
already inside the allocation.

Observed examples:

```text
site             ~16s cumulative
rootutils        ~54s cumulative
numpy.core       ~77s cumulative
```

This points to very slow Python/Conda import metadata loading from shared
storage or node-local filesystem contention. The practical fix is to reduce
startup dependence on shared storage by unpacking the conda environment and
syncing the repo to node-local disk before launching Python.

## Final Fix

The fix is implemented as local-cache startup in `sbatch/_common.sh`.

Normal GPU sbatch files now enable it by default:

```bash
export POCKET_PLUS_LOCAL_CACHE_ENABLED="${POCKET_PLUS_LOCAL_CACHE_ENABLED:-1}"
export POCKET_PLUS_LOCAL_CACHE_ENV_PACK="${POCKET_PLUS_LOCAL_CACHE_ENV_PACK:-/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz}"
```

Runtime behavior:

- Use `${SLURM_TMPDIR}` if Slurm provides it, otherwise `/tmp/${USER}/slurm_${SLURM_JOB_ID}`.
- Unpack the packed conda env into `${LOCAL_BASE}/conda_env`.
- Sync `/home/penghongen/My_Project/Pocket_Plus` into `${LOCAL_BASE}/My_Project/Pocket_Plus`.
- Create `${LOCAL_BASE}/My_Project/.project-root` so `rootutils` resolves the local project root.
- Export `PROJECT_ROOT`, `PYTHONPATH`, `PATH`, `CONDA_PREFIX`, `LD_PRELOAD`, and `PYTHONPYCACHEPREFIX` to prefer local files.
- Keep outputs under `/home/penghongen/My_Project/feedback_plus`.
- Clean the current job's local cache when the allocation is finally released.

## Known Bad Or Weak Diagnostic Routes

Do not start with direct SSH to the compute node:

```bash
ssh gnode02
```

On this cluster it may be blocked by policy or fail from the login node. Treat
that as expected, not as evidence about the Python process.

Be careful with this form:

```bash
JOBID=245087; bash -lc 'scontrol show job -dd "$JOBID"'
```

The semicolon assignment does not export `JOBID` into the child shell. In our
first attempt this caused Slurm queries to run with an empty job id and dump many
unrelated jobs. Use `JOBID=245087 bash -lc '...'` instead.

`srun --jobid=<JOBID> --overlap ...` can be useful, but it is not reliable on an
unhealthy or blocked allocation. If it hangs, stop using that route and switch to
the `kill_lock` diagnostic restart workflow below.

## First Snapshot From Login Node

Run this before touching the job. It confirms Slurm state, output paths, command
file, lock state, and accounting without compute-node SSH.

```bash
JOBID=245087 bash -lc '
set +e
echo "========== Slurm Job =========="
date
echo "JOBID=$JOBID"

JOBINFO="$(scontrol show job -dd "$JOBID" 2>&1)"
echo "$JOBINFO" | sed "s/ /\n/g" | egrep "JobId=|JobState=|RunTime=|NodeList=|BatchHost=|Command=|WorkDir=|StdOut=|StdErr=|Reason=|ExitCode=|TRES=|JOB_GRES="

OUT=$(printf "%s\n" "$JOBINFO" | tr " " "\n" | awk -F= "/^StdOut=/{print \$2; exit}")
ERR=$(printf "%s\n" "$JOBINFO" | tr " " "\n" | awk -F= "/^StdErr=/{print \$2; exit}")
RUN="/home/penghongen/run_cmd_${JOBID}.sh"

echo
echo "========== Files / Locks =========="
echo "OUT=$OUT"
echo "ERR=$ERR"
echo "RUN=$RUN"
ls -lh "$OUT" "$ERR" "$RUN" 2>/dev/null
stat "$OUT" "$ERR" "$RUN" 2>/dev/null
ls -l /home/penghongen/*_lock_${JOBID} 2>/dev/null || true

echo
echo "========== run_cmd =========="
sed -n "1,160p" "$RUN" 2>/dev/null

echo
echo "========== Recent logs =========="
tail -n 120 "$OUT" 2>/dev/null
tail -n 160 "$ERR" 2>/dev/null

echo
echo "========== Accounting =========="
sacct -j "$JOBID" --format=JobID,JobName%25,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList%20 -P 2>/dev/null
'
```

## Optional In-Allocation Probe

Use only if Slurm allows overlapping job steps. It is useful when it works, but
it is not the primary route.

```bash
JOBID=245087
NODE=$(scontrol show job -dd "$JOBID" | tr " " "\n" | awk -F= "/^BatchHost=/{print \$2; exit}")
PROBE=/home/penghongen/probe_${JOBID}_$(date +%Y%m%d_%H%M%S).log
srun --jobid="$JOBID" --overlap -w "$NODE" -N1 -n1 --cpus-per-task=1 bash -lc "exec > '$PROBE' 2>&1; echo HOST=\$(hostname) DATE=\$(date); ps -u \$USER -o pid,ppid,stat,etime,%cpu,%mem,rss,wchan:32,cmd --sort=-%cpu | egrep 'PID|train.py|run_cmd_${JOBID}|python|bash|slurm' | head -120; nvidia-smi; free -h; df -h /home /tmp 2>/dev/null"
echo "PROBE=$PROBE"
tail -n 200 "$PROBE"
```

If this does not enter the node quickly, do not spend time fighting it. Continue
with the reliable restart workflow.

## Reliable Diagnostic Restart

This is the main technique. It kills only the current Python/run_cmd process
group, keeps the Slurm allocation alive, rewrites `run_cmd_<JOBID>.sh`, and
retries with import-level diagnostics.

```bash
JOBID=245087
RUN=/home/penghongen/run_cmd_${JOBID}.sh
OUT=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.out
ERR=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.err
EXP=$(grep -oE '\+experiment=[^"[:space:]]+' "$RUN" | head -1)
EXP=${EXP:-+experiment=ligand001}

touch /home/penghongen/kill_lock_${JOBID}
echo "[diag] sent kill_lock, waiting for try_lock..."
for i in $(seq 1 90); do
  [ -f /home/penghongen/try_lock_${JOBID} ] && break
  sleep 2
done
if [ ! -f /home/penghongen/try_lock_${JOBID} ]; then
  echo "[ERROR] try_lock did not appear"
  tail -n 80 "$OUT"
  tail -n 80 "$ERR"
  exit 2
fi

cp "$RUN" "${RUN}.bak.importdiag.$(date +%Y%m%d_%H%M%S)"
cat > "$RUN" <<EOF
#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
python -u -X faulthandler -X importtime /home/penghongen/My_Project/Pocket_Plus/src/train.py "$EXP"
EOF
chmod +x "$RUN"
rm -f /home/penghongen/try_lock_${JOBID}

echo "[diag] restarted $JOBID with import diagnostics: $EXP"
sleep 30
echo "===== OUT ====="
tail -n 140 "$OUT"
echo "===== ERR ====="
tail -n 220 "$ERR"
```

Watch the import stream:

```bash
JOBID=245087
OUT=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.out
ERR=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.err
for i in $(seq 1 12); do
  echo
  echo "===== $(date) round=$i ====="
  ls -lh "$OUT" "$ERR"
  echo "--- OUT ---"
  tail -n 30 "$OUT"
  echo "--- last import lines ---"
  grep "import time:" "$ERR" | tail -n 50
  echo "--- key imports / errors ---"
  grep -E "import time:.*(site|rootutils|numpy|torch|hydra|lightning|wandb|src\\.)|GLOO_SOCKET_IFNAME|Traceback|Error executing job" "$ERR" "$OUT" 2>/dev/null | tail -n 80
  sleep 20
done
```

## Local Cache Proof Test On Existing Allocation

Use this when a job is already paused at `try_lock` and you want to prove local
cache helps before resubmitting.

```bash
JOBID=245099
EXP="+experiment=ligand002"
RUN=/home/penghongen/run_cmd_${JOBID}.sh
cp "$RUN" "${RUN}.bak.localcache.$(date +%Y%m%d_%H%M%S)"
cat > "$RUN" <<EOF
#!/bin/bash
set -euxo pipefail

EXP="$EXP"
ENV_PACK="/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz"
LOCAL_BASE="\${SLURM_TMPDIR:-/tmp/\${USER}/slurm_\${SLURM_JOB_ID}}"
LOCAL_PROJECT_ROOT="\${LOCAL_BASE}/My_Project"
LOCAL_ENV="\${LOCAL_BASE}/conda_env"
LOCAL_REPO="\${LOCAL_PROJECT_ROOT}/Pocket_Plus"

mkdir -p "\${LOCAL_BASE}" "\${LOCAL_PROJECT_ROOT}"
touch "\${LOCAL_PROJECT_ROOT}/.project-root"
df -h "\${LOCAL_BASE}"

if [ ! -x "\${LOCAL_ENV}/bin/python" ]; then
  rm -rf "\${LOCAL_ENV}"
  mkdir -p "\${LOCAL_ENV}"
  tar -xzf "\${ENV_PACK}" -C "\${LOCAL_ENV}"
  "\${LOCAL_ENV}/bin/conda-unpack" || true
fi

rsync -a --delete --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" --exclude ".pytest_cache" --exclude "feedback_plus" /home/penghongen/My_Project/Pocket_Plus/ "\${LOCAL_REPO}/"

export CONDA_PREFIX="\${LOCAL_ENV}"
export PATH="\${LOCAL_ENV}/bin:\${PATH}"
export PROJECT_ROOT="\${LOCAL_PROJECT_ROOT}"
export PYTHONPATH="\${LOCAL_PROJECT_ROOT}:\${LOCAL_REPO}:\${PYTHONPATH:-}"
export EXPERIMENT_FEEDBACK_ROOT="/home/penghongen/My_Project/feedback_plus"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTHONPYCACHEPREFIX="\${LOCAL_BASE}/pycache"
export LD_PRELOAD="\${LOCAL_ENV}/lib/libstdc++.so.6:\${LOCAL_ENV}/lib/libgcc_s.so.1"

cd "\${LOCAL_PROJECT_ROOT}"
python -u Pocket_Plus/src/train.py "\${EXP}"
EOF
chmod +x "$RUN"
rm -f /home/penghongen/try_lock_${JOBID}
```

Important details from the successful test:

- The first local-cache run failed because `.project-root` was created at the
  wrong level. The correct marker is `${LOCAL_BASE}/My_Project/.project-root`.
- After that fix, job `245099` reached `GLOO_SOCKET_IFNAME`, seed setup,
  model/data initialization, and WandB from local env Python.
- The later `exit code 137` was caused by manual `kill_lock`, not local-cache
  failure.

## Interpreting Results

- If `-X importtime` shows very slow stdlib/NumPy imports and keeps crawling,
  suspect import/filesystem slowness, not project logic.
- If the job reaches `[Train] GLOO_SOCKET_IFNAME`, data module setup, model
  setup, or WandB, the original startup hang has been bypassed. Later tracebacks
  should be debugged separately.
- If `try_lock_<JOBID>` exists, the allocation is paused and safe to edit.
- If `after_lock_<JOBID>` exists, the allocation is intentionally being held.
- If local cache reaches training quickly, prefer resubmitting with the normal
  GPU sbatch scripts, which now enable local cache by default.
