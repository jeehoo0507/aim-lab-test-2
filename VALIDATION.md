# Implementation validation

2026-09-25, frozen DINO attention pilot, Linux x86_64, Python 3.11.16, torch 2.5.1+cu124, CPU:

- `python -m pytest -q`: **62 passed**, 14 existing Matplotlib/pyparsing deprecation warnings. `bash -n setup.sh`, CLI help and `git diff --check` passed.
- Official DINO reference is pinned to `7c446df5b9f45747937fb0d72314eb9f7b66930a`. Untouched class definitions match the adapter's features, last-block head-mean CLS attention and top-98 selection.
- Loaded the actual official DINO ViT-S/16 pretrained checkpoint (file SHA256 `1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7`) and verified those three outputs against the official full-size implementation on CPU. Selector tensor fingerprint: `acb98ef896cc94acceaed9db9ed40bf0fe7c6da5039bf31a18a5f73dbc58a438`.
- Tests check exact 10-token replacement, no foreground-label dependence, epoch-50/51 boundary and 78+20 uniqueness, selector freezing/RNG isolation, bit-exact interrupted/resumed students, fixed DINO probe selection with changing student attention, and batch/seed-independent diagnostic masks.
- Synthetic integration exercised the actual three-process DINO benchmark and training launcher, rejection of mismatched teacher/config, test separation until all three runs finish, best/last evaluation, paired baseline comparison CSVs and compact report export.
- No real COCO GPU run was performed locally. DINO accuracy gains, runtime and VRAM remain to be measured on the server. This pilot uses DINO as a frozen patch selector; it does not implement DINO self-supervised training.

2026-09-25, DeiT-Base teacher pilot, Linux x86_64, Python 3.11.16, torch 2.5.1+cu124, CPU:

- `uv sync --frozen`, `bash -n setup.sh`, `git diff --check`: passed.
- `python -m pytest -q`: **57 passed** (14 existing Matplotlib/pyparsing deprecation warnings).
- New pilot integration test: actual three-process Full KD / MaskedKD / Random10 benchmark and training, interrupted teacher resume, completed-run reuse, test separation, paired Small/Base CSVs and report export passed on synthetic data.
- Real DeiT-Base: 85,806,346 parameters with the 10-class head; official pretrained checkpoint successfully downloaded/loaded, and both 196/98-patch CPU forwards produced finite logits.
- Student initialization matches the unmodified `c1cbe580` builder on this host. All nine archived Small baseline configurations match the new pilot except teacher size. Exact archived initialization hashes must also match at launch on the original server; this CPU host does not reproduce the archived server hashes, so they were not bypassed.
- No local CUDA device or prepared COCO images: Base GPU VRAM, runtime and real-data accuracy are **not measured here**. The launcher measures Base-specific serial and mixed three-process workloads on the server before training.

Earlier validation:

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
- Benchmark recommendation tests: models staged teachers plus a shared 42-student task queue, rejects slower parallel execution, falls back on failed parallel trials, includes teacher/resume/midpoint storage, and rejects incomplete/expired automatic recommendations. CUDA throughput/memory remains unmeasured locally.

**Not yet validated on the server:** CUDA driver/runtime, A5000 peak VRAM/throughput, real COCO image download/QC counts, real training convergence/accuracy. The code includes `check`, `prepare`, `benchmark`, and `pilot` for these steps. No real COCO experiments were run locally.

The storage-limited policy keeps student epoch50/best/final, teacher best/final, and full resume state only while a run is active plus MaskedKD epoch50. The completed checkpoint projection is about 3.8GB; benchmark projects raw probe/validation output and warns above 10GB without imposing a training cap. Exported GitHub reports do not contain model weights or the complete attention tensor files.
