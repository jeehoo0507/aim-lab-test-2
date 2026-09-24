#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p logs "$MPLCONFIGDIR"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 없습니다. 기존 서버에서 사용한 uv 실행 경로를 PATH에 추가해 주세요. sudo는 사용하지 않습니다."
  exit 1
fi
uv sync --frozen
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
task_log="logs/setup_${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$task_log") 2>&1
echo "Project: $PWD"
echo "Log: $PWD/$task_log"
case "$mode" in
  check)
    .venv/bin/python -m pytest -q
    .venv/bin/python run.py smoke --device cpu
    .venv/bin/python run.py check "$@"
    ;;
  prepare)
    .venv/bin/python run.py prepare "$@"
    .venv/bin/python run.py check "$@"
    ;;
  benchmark) .venv/bin/python run.py benchmark "$@" ;;
  pilot)
    .venv/bin/python run.py pipeline --seeds 0 --methods ce full "$@"
    ;;
  run) .venv/bin/python run.py pipeline "$@" ;;
  evaluate) .venv/bin/python run.py evaluate "$@" ;;
  export) .venv/bin/python run.py export "$@" ;;
  teacher-base) .venv/bin/python -m coco_kd.teacher_base run "$@" ;;
  teacher-base-benchmark) .venv/bin/python -m coco_kd.teacher_base benchmark "$@" ;;
  teacher-base-evaluate) .venv/bin/python -m coco_kd.teacher_base evaluate "$@" ;;
  teacher-base-export) .venv/bin/python -m coco_kd.teacher_base export "$@" ;;
  anneal) .venv/bin/python -m coco_kd.anneal run "$@" ;;
  anneal-export) .venv/bin/python -m coco_kd.anneal export "$@" ;;
  adaptive) .venv/bin/python scripts/run_adaptive200.py run "$@" ;;
  adaptive-evaluate) .venv/bin/python scripts/run_adaptive200.py evaluate "$@" ;;
  random-low-sweep)
    .venv/bin/python scripts/run_adaptive200.py run \
      --config configs/random_low_sweep.json --seeds 1 2 --inits scratch \
      --methods adaptive_random_to_low_10 random_rescue_to_low_70 random_low_mixed_10 \
      --jobs 6 "$@"
    ;;
  random-low-evaluate)
    .venv/bin/python scripts/run_adaptive200.py evaluate \
      --config configs/random_low_sweep.json --seeds 1 2 --inits scratch \
      --methods adaptive_random_to_low_10 random_rescue_to_low_70 random_low_mixed_10 "$@"
    ;;
  *) echo "Usage: bash setup.sh [check|prepare|benchmark|pilot|run|evaluate|export|teacher-base|teacher-base-benchmark|teacher-base-evaluate|teacher-base-export|anneal|anneal-export|adaptive|adaptive-evaluate|random-low-sweep|random-low-evaluate]"; exit 2 ;;
esac
