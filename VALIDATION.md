# Implementation validation

2026-09-22, local macOS ARM64, Python 3.11.15, torch 2.5.1, CPU.

- `uv lock` and `uv sync --frozen`: passed. Lock contains Linux x86_64 CUDA 12.4 torch/torchvision sources and macOS sources.
- `python -m pytest -q`: **25 passed**. Matplotlib's pyparsing deprecation notices do not affect test outcomes.
- `bash -n setup.sh`: passed.
- `bash setup.sh check --device cpu`: passed, including synthetic teacher + all seven student methods, best/last evaluation, raw/counterfactual probes, analysis tables and scientific plots.
- `run.py export` on synthetic output: passed; smoke report is kept in ignored `reports/smoke/` and cannot be pushed as a real experiment result.
- Rendered main figure inspected locally. Synthetic figure is labeled as a smoke test.
- Resume equivalence: interrupted/resumed tiny model parameters match uninterrupted training exactly on CPU.
- Upstream parity: official student logits/attention/gradients and full/masked teacher forward match the adapted model.
- Mask invariants: 98 unique tokens, exact10 random swaps, strongest88 retention for low-score rescue, FG feasibility, no FG-label use in fixed random rescue.
- Data checks: strict category filter considers all annotations, segmentation union decoding, duplicate priority, held-out/probe isolation, file hash integrity.
- Statistical checks: fixed starting error cohort, censored non-corrections retained, three-epoch threshold, seed pairing rather than image pseudo-replication.
- Benchmark: actual spawned one-, two- and three-process CPU trials on synthetic data passed for teacher/all seven methods, validation, normal/detailed probes and temporary checkpoint I/O. Temporary files are removed and experiment checkpoints are untouched.
- Benchmark recommendation tests: accounts for three-seed tail, rejects slower parallel execution, falls back on failed parallel trials, includes teacher/resume/midpoint storage, and rejects incomplete/expired automatic recommendations. CUDA throughput/memory remains unmeasured locally.

**Not yet validated on the server:** CUDA driver/runtime, A5000 peak VRAM/throughput, real COCO image download/QC counts, real training convergence/accuracy. The code includes `check`, `prepare`, `benchmark`, and `pilot` for these steps. No real COCO experiments were run locally.

The storage-limited policy keeps student epoch50/best/final, teacher best/final, and full resume state only while a run is active plus MaskedKD epoch50. The completed checkpoint projection is about 3.8GB; benchmark projects raw probe/validation output and warns above 10GB without imposing a training cap. Exported GitHub reports do not contain model weights or the complete attention tensor files.
