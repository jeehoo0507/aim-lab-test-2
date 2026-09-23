"""Run the four seed-matched Random/Full to MaskedKD switches in parallel."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METHODS = ("random_to_student_20", "random_to_student_50",
           "full_to_student_20", "full_to_student_50")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_switch_pilot.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("scratch", "imagenet"), default="scratch")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads(Path(args.config).read_text())
    if config["num_workers"] != 0 or config["epochs"] != 100:
        raise ValueError("Four-job comparison requires 0 DataLoader workers and 100 epochs")
    output = Path(config["output_root"])
    base = output / f"seed_{args.seed}"
    run_root = base / args.init
    required = [base / "teacher" / "best.pt"]
    for method in ("student", "random", "full"):
        required.extend((run_root / method / "result.json", run_root / method / "test_metrics.json"))
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Existing teacher/baseline required: {path}")
    if not Path(config["data_root"], "manifest.json").exists():
        raise FileNotFoundError("Prepared COCO dataset manifest is missing")

    benchmark = Path("reports/benchmark/latest.json")
    if benchmark.exists() and config["device"].startswith("cuda"):
        saved = json.loads(benchmark.read_text())
        required = saved.get("required_free_gib_by_jobs", {}).get(str(args.jobs))
        try:
            free_mib = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).splitlines()[0].strip())
            free_gib = free_mib / 1024
            print(f"GPU free={free_gib:.2f}GiB; measured conservative requirement for {args.jobs} jobs={required}GiB", flush=True)
            if required is not None and free_gib < required:
                raise RuntimeError(f"Need at least {required:.2f}GiB free for {args.jobs} jobs on this benchmark")
        except FileNotFoundError:
            print("nvidia-smi unavailable; continuing without a live VRAM check", flush=True)

    logs = output / "_jobs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(config.get("num_threads", 2)),
               MKL_NUM_THREADS=str(config.get("num_threads", 2)))

    def task(method):
        completed = run_root / method / "result.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if result.get("method") != method or result.get("last_epoch") != 100:
                raise ValueError(f"Unexpected completed run at {completed}")
            print(f"REUSE completed {method}", flush=True)
            return method, 0
        logfile = logs / f"seed_{args.seed}_{args.init}_{method}.log"
        command = [sys.executable, "-u", str(ROOT / "run.py"), "train", "--config", args.config,
                   "--seed", str(args.seed), "--init", args.init, "--method", method]
        print(f"START {method}; log={logfile}", flush=True)
        with logfile.open("a", buffering=1) as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {method} (exit {result.returncode})", flush=True)
        return method, result.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(task, METHODS))
    failed = [method for method, code in results if code != 0]
    if failed:
        raise RuntimeError(f"Training failed for {failed}; inspect the corresponding _jobs logs")

    evaluation = [sys.executable, str(ROOT / "run.py"), "evaluate", "--config", args.config,
                  "--seeds", str(args.seed), "--inits", args.init, "--methods", *METHODS]
    subprocess.run(evaluation, cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_switch_grid.py"),
                    "--output-root", str(output), "--seed", str(args.seed), "--init", args.init],
                   cwd=ROOT, env=env, check=True)
    print("COMPLETE: four runs, test evaluation, and comparison report", flush=True)


if __name__ == "__main__":
    run()
