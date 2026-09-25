#!/usr/bin/env bash
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
runtime="$PWD/.venv-deletion/bin/python"
if [[ "$mode" == "env" ]]; then
  command -v uv >/dev/null || { echo 'uv is required'; exit 1; }
  python3 - <<'PY'
import shutil
if shutil.disk_usage('.').free < 25*1024**3:
    raise SystemExit('Need at least 25 GiB free before installing the isolated CUDA runtime')
PY
  if [[ ! -x "$runtime" ]]; then uv venv --python 3.11 .venv-deletion; fi
  uv pip install --python "$runtime" torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$runtime" -r requirements-deletion.txt
  "$runtime" -c 'import torch; print("Runtime:", torch.__version__, "CUDA build:", torch.version.cuda)'
  echo 'Environment ready. No training started. Next: bash setup_deletion.sh check'
  exit 0
fi
case "$mode" in
  check|prepare|benchmark|teacher|pilot|run|evaluate) ;;
  *) echo 'Usage: bash setup_deletion.sh [env|check|prepare|benchmark|teacher|pilot|run|evaluate]'; exit 2 ;;
esac
if [[ ! -x "$runtime" ]]; then
  if [[ "$mode" == "check" && -x .venv/bin/python ]]; then
    runtime="$PWD/.venv/bin/python"
  else
    echo 'First run: bash setup_deletion.sh env (separate environment; original .venv is kept)'
    exit 2
  fi
fi
mkdir -p logs "$MPLCONFIGDIR"
task_log="logs/deletion_${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Project: $PWD"
echo "Log: $PWD/$task_log"
"$runtime" -m coco_kd.deletion_run "$mode" "$@" 2>&1 | tee -a "$task_log"
