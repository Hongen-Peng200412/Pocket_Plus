cat > ~/diagnose_pid_hang.sh <<'SCRIPT'
#!/usr/bin/env bash
set +euo pipefail

: "${PID:?Usage: PID=19215 JOBID=257058 bash ~/diagnose_pid_hang.sh}"
JOBID=${JOBID:-unknown}
TS=$(date +%Y%m%d_%H%M%S)
DIAG_DIR=${DIAG_DIR:-/tmp/diag_pid_${JOBID}_${PID}_${TS}}
mkdir -p "$DIAG_DIR"
exec > >(tee "$DIAG_DIR/report.txt") 2>&1

echo "DIAG_DIR=$DIAG_DIR"
echo "PID=$PID"
date
hostname

echo
echo "========== Process Detail =========="
ps -o pid,ppid,user,stat,pcpu,pmem,etime,wchan:40,cmd -p "$PID" || true
printf "cwd="
readlink -f "/proc/$PID/cwd" 2>/dev/null || true
printf "cmdline="
tr "\0" " " < "/proc/$PID/cmdline" 2>/dev/null
echo

echo
echo "========== Important Env =========="
tr "\0" "\n" < "/proc/$PID/environ" 2>/dev/null \
  | grep -E "^(SLURM_JOB_ID|CUDA_VISIBLE_DEVICES|CONDA_PREFIX|PYTHONPATH|OMP_NUM_THREADS|MKL_NUM_THREADS)=" || true

echo
echo "========== Children =========="
ps --ppid "$PID" -o pid,ppid,stat,pcpu,pmem,etime,wchan:40,cmd --sort=-pcpu || true

echo
echo "========== Process Tree =========="
pstree -ap "$PID" || true

echo
echo "========== Threads / Kernel Wait =========="
ps -L -p "$PID" -o pid,tid,stat,pcpu,etime,wchan:40,comm --sort=-pcpu | head -120 || true
printf "main_wchan="
cat "/proc/$PID/wchan" 2>/dev/null || true
echo
cat "/proc/$PID/stack" 2>/dev/null || true

echo
echo "========== Python Stack Snapshot 1 =========="
PY1="$DIAG_DIR/pyspy_1.txt"
timeout 30s py-spy dump --pid "$PID" --locals 2>&1 | tee "$PY1" | head -420

sleep 8

echo
echo "========== Python Stack Snapshot 2 =========="
PY2="$DIAG_DIR/pyspy_2.txt"
timeout 30s py-spy dump --pid "$PID" --locals 2>&1 | tee "$PY2" | head -420

echo
echo "========== Stack Keyword Hints =========="
grep -Ein "two_stage|voxel|threshold|grid|cache|metric|save|dump|json|pickle|np.save|to_csv|open|write|flush|joblib|loky|multiprocessing|resource_tracker|join|wait|queue|futex|torch|cuda|synchronize|cpu\(|numpy" \
  "$PY1" "$PY2" | head -220 || true

echo
echo "========== Open Files =========="
if command -v lsof >/dev/null 2>&1; then
  lsof -p "$PID" 2>/dev/null | tee "$DIAG_DIR/lsof.txt" | tail -180
else
  ls -l "/proc/$PID/fd" 2>/dev/null | tee "$DIAG_DIR/fd.txt" | head -180
fi

echo
echo "========== Short strace =========="
STRACE="$DIAG_DIR/strace.txt"
if command -v strace >/dev/null 2>&1; then
  timeout 18s strace -f -tt -T -s 240 \
    -e trace=read,write,openat,close,stat,newfstatat,lseek,fsync,fdatasync,futex,wait4,clone,nanosleep,clock_nanosleep,poll,ppoll,select,pselect6 \
    -p "$PID" -o "$STRACE" 2>"$DIAG_DIR/strace_attach.err"
  echo "--- strace attach stderr ---"
  cat "$DIAG_DIR/strace_attach.err" 2>/dev/null || true
  echo "--- strace tail ---"
  tail -n 220 "$STRACE" 2>/dev/null || true
else
  echo "strace not found"
fi

echo
echo "========== GPU For PID =========="
nvidia-smi || true
nvidia-smi pmon -c 5 2>/dev/null || true

echo
echo "========== Report =========="
echo "$DIAG_DIR/report.txt"
SCRIPT

chmod +x ~/diagnose_pid_hang.sh
PID=19215 JOBID=257058 bash ~/diagnose_pid_hang.sh
