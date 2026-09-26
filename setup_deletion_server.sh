#!/usr/bin/env bash
# Existing RTX A5000 server: original locked runtime, existing Base teacher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" == "env" ]]; then
  command -v uv >/dev/null || { echo 'uv is required on the server'; exit 1; }
  uv sync --frozen
  .venv/bin/python -c 'import torch; print("Server runtime:", torch.__version__, "CUDA build:", torch.version.cuda)'
  echo 'Ready. No training started. Next: bash setup_deletion_server.sh check'
  exit 0
fi
case "$mode" in
  check|prepare|benchmark|teacher|pilot|run|start|evaluate|parallel|parallel-benchmark|parallel-evaluate|test) ;;
  *) echo 'Usage: bash setup_deletion_server.sh [env|check|benchmark|start|parallel|parallel-benchmark|parallel-evaluate|evaluate|test]'; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  echo 'First run: bash setup_deletion_server.sh env'
  exit 2
fi
mkdir -p logs "$MPLCONFIGDIR"
task_log="logs/deletion_server_${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Project: $PWD"
echo "Log: $PWD/$task_log"
if [[ "$mode" == "test" ]]; then
  .venv/bin/python -m pytest -q tests/test_deletion.py tests/test_deletion_server.py tests/test_deletion_parallel.py "$@" 2>&1 | tee -a "$task_log"
  exit 0
fi
case "$mode" in
  parallel) parallel_action=start ;;
  parallel-benchmark) parallel_action=calibrate ;;
  parallel-evaluate) parallel_action=evaluate ;;
  *) parallel_action='' ;;
esac
if [[ -n "$parallel_action" ]]; then
  .venv/bin/python -m coco_kd.deletion_parallel "$parallel_action" "$@" 2>&1 | tee -a "$task_log"
  exit 0
fi
options=(--config configs/deletion_server.json)
if [[ "$mode" != "teacher" ]]; then
  options+=(--teacher-checkpoint outputs/teacher_base_pilot/seed_0/teacher/best.pt)
fi
if [[ "$mode" == "start" ]]; then
  # Every stage must succeed before the next stage can run. No automatic retry.
  .venv/bin/python -m coco_kd.deletion_run check "${options[@]}" --strict "$@" 2>&1 | tee -a "$task_log"
  .venv/bin/python -m coco_kd.deletion_run benchmark "${options[@]}" --steps 8 "$@" 2>&1 | tee -a "$task_log"
  .venv/bin/python -m coco_kd.deletion_run run "${options[@]}" "$@" 2>&1 | tee -a "$task_log"
  exit 0
fi
if [[ "$mode" == "check" ]]; then options+=(--strict); fi
.venv/bin/python -m coco_kd.deletion_run "$mode" "${options[@]}" "$@" 2>&1 | tee -a "$task_log"
