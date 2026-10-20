# Multi-material compact kernels: corrected-reference study

## Original behavior and scope

The immutable CUDA source contains five compact GPU kernels. Its active dense reference is CPU code; the dense material-centric call is commented out. Three active build modes are covered: unfused CSR, fused CSR and fused linked lists. `FUSED` combines the pure/mixed work within each density/pressure algorithm; it does not fuse all three algorithms.

An actual unchanged `multimat 32 24 0.3 0.2 0.1` run returns zero and prints “All tests passed.” A capture-only driver using unchanged source computations observes **1,287 infinities** in both dense and compact neighborhood outputs. The neighborhood includes its own cell, whose squared distance is zero. The original `fabs(Inf-Inf) > tolerance` check does not reject NaN. Full local captures, source/geometry and hashes are recorded in `native_diagnostics/original_nonfinite_witness.json`; this is an observed scientific numerical failure, not a framework limitation.

All measured four-way configurations are explicitly **corrected-reference secondary**, excluded from untouched-native aggregates. The only GPU arithmetic correction is to skip the zero-distance self neighbor; the exact diff is retained. Positive-distance neighbor counts are nonzero for every measured interior material. A no-neighbor generalization is not certified.

The named fixture is 96×80 cells and all 50 source material slots, with 1,920 cells each containing one, two, three or four materials. Density, temperature and material parameters vary. The 5,760 mixed cells contain 17,280 fractions. All pure/mixed, CSR and linked branches are exercised. The bundled real 1000×1000 file and Makefile 2000×2000 regimes are present but unmeasured; no missing-data or unsupported claim is made.

## Numerical evidence

Each of the four implementations executes all active compact stages on the GPU. CUDA.jl uses `@cuda`, KA uses public `@kernel`/`@index`, and JACC uses public `parallel_for`, with no foreign CUDA device calls or native-kernel delegation. All are framework device kernels. Cell workgroups remain 32×4; mixed workgroups use 128 threads. JACC explicitly requests zero shared memory.

Each configuration has five checked in-process trials, with JIT warmup outside them and the mean of trials 2–5 retained. Each artifact contains the **actual GPU readback of all inputs** (149,811 binary64-encoded values, including exact Int32 indices), then all five complete output arrays and 17 guard values per output. All 60 artifacts, 99,598,080 bytes, pass the independent audit. Inputs and guards are byte exact; all outputs must be finite and satisfy `abs_error <= 1e-8 + 2e-13*abs(reference)`. CUDA/CUDA.jl and both portable versions agree under this contract; raw floating-point identity is not claimed.

The independent CPU oracle enumerates a dense `(cell, material)` dictionary and same-material neighbors, using `math.fsum`; it does not follow compact pointers. The source neighborhood divides its inverse-distance sum by neighbor count, **not** by the sum of inverse-distance weights. We preserve that operator and do not certify a physical equation of state or full continuum simulation.

## Timing and current reporting qualification

The timed region sums density submission/completion, pressure submission/completion and neighborhood submission/completion. Allocation, transfers, fixture creation, CPU dense reference, oracle and artifact capture are outside. Individual algorithm times are retained. Host API/dispatch and full-device completion costs are included, so these are not kernel-only or causal frontend measurements.

The immutable structured rows name their identical, explicit descriptor `region` rather than `timing_region`. All three modes now have **matched performance valid** through the hash-bound immutable replays `20261020T150159402265Z`, `20261020T150304716954Z` and `20261020T150306115447Z`. Every original row, time, source and numerical criterion remains unchanged. `timing_metadata_reconciliation_audit.json` independently checks all60 unchanged rows, source bindings and means. Original unmatched receipts are retained; this was descriptive reconciliation with **zero GPU reruns**, not an arithmetic repair or selection of faster samples.

Native NVCC 12.0 and Julia's artifact compiler 12.9.41 differ. The source local nine-element distance array is represented by a fixed Julia tuple. Register/local allocation, achieved occupancy, actual traffic and bottleneck attribution are unknown. The small scope and compiler/API resource differences limit general performance conclusions.

## Reproduction and assets

`prepare_assets.py OUTPUT_DIRECTORY` uses Python standard library only. It generates all three input files, the complete CPU oracle and `study_plan.json`. A fresh isolated run reproduces all five files byte for byte (`cold_regeneration_audit.json`); generated binary inputs total 4,056,944 bytes. No irreplaceable binary or raw GPU capture needs adding to Git. Hashes and sizes are in `study_plan.json`.

Use the existing pinned KA and JACC projects. The native configurations compile an owned corrected source copy for the three macro modes. The complete final numerical runs are:

- mode0: `20261020T012409086234Z_multimaterial`
- mode1: `20261020T012442739070Z_multimaterial`
- mode2: `20261020T012446471625Z_multimaterial`

`prepare_cold_config.py --source-root SOURCE_ROOT --mode 0 --output NEW_CONFIG` relocates the saved model specifications, removes reuse and assigns a new explicit cold-run ID. Regenerate the fixtures first; execute via the central verifier on physical GPU6 (it locks internally). The old field alias remains faithfully in the measured source: a fresh numerical run therefore needs the same separately reviewed immutable metadata replay to describe its timing region. That replay is not a second GPU series. The cold fixture generator was executed; an end-to-end fresh four-way replay has not been run again.

## Effort and tokens

Each target has one submitted device source bundle, five kernel definitions and three actual configuration/device-JIT process checks, with zero compiler or numerical repairs. Eight arithmetic helpers and the common host harness are created with CUDA.jl and reused in ordered KA/JACC translation. Three modes or fifteen trials do not count as additional candidates. Native scientific correction, shared reporting review and annotation have separate effort/token records. The annotation helper had one Python syntax rejection before executing/writing; it is not a GPU/compiler attempt.

Observed counters include reused session context and cached inputs; cached/reasoning counters are subsets. They are not exclusive billed costs or recovered historical translation usage. See `observed_token_windows.json`, canonical per-model manifests and `calibration_537/multimaterial/`.
