# PRNA: RNA partition-function GPU pipeline

## Covered path and scientific meaning

This study covers the original **`-t` pair-probability matrix path**, using the complete provided 292-base test sequence and four controls: a seven-base two-pair model, nine unpairable bases, a seven-base lowercase constraint model, and a 27-base stage-trace case. Every source GPU entry is translated: boundary initialization and eight stages for every sequence distance. Three generic device functions/callbacks specialize into the nine logical source entries; neither distances nor input shapes are translation attempts.

The model evaluates negative-log Boltzmann partition functions using the source RNA thermodynamic tables, normalized by `RT` at `T37=310.15 K`. The source's unused `param.temperature` field is zero defined for capture; it does **not** mean a zero-Kelvin simulation. The index wrap encodes exterior context of a linear sequence, not circular RNA. The source pairing mask requires uppercase endpoints, canonical A–U/C–G/G–U pairs and a potential neighboring pair; it does not require the neighboring pair to be present in every sampled microstate.

The larger default HIV sequence, DNA/alternative-precision modes, and default ProbKnot CT/helix-pruning postprocessing are unmeasured and unported here. The required data archives were recovered from their exact declared DVC objects with size/MD5/SHA checks; they are not unavailable-data blockers.

## Original-source qualification

An actual unchanged 292-base CUDA CLI produces 85,264 finite symmetric probabilities, with maximum 0.878828 and maximum row mass below one. On an unchanged 27-base run, initcheck reports 92 uninitialized **host readback** accesses because the full V matrix includes uncomputed non-pair entries; it reports zero device uninitialized reads. Racecheck reports zero hazards on that control, not a general proof.

The named harness defines every state allocation as positive infinity before source initialization and value initializes the parameter structure before the original parser. Original GPU bodies are included verbatim. This is an explicit **defined-storage corrected-reference secondary**, excluded from untouched-native aggregates. It is not an observed wrong-probability or GPU arithmetic correction: all 85,264 harness probabilities match the original CLI at its six-digit text precision. Original unrounded probability bits were not captured, so stronger original-output identity is not asserted.

## Full numerical and scientific evidence

Every checked trial captures actual GPU input readbacks, all eleven full final state buffers and probabilities for all five cases, plus a separately executed **untimed** complete 27-base trace: initialization and all eight stages at all 26 distances (209 snapshots). This does not claim that overwritten timed 292-base intermediate states were captured.

Positive infinity is a legitimate impossible DP state, corresponding to zero Boltzmann weight. Every output Float32 is stored losslessly as three Float64 serialization fields: finite value (zero for structural infinity), exact semantic class, and its complete UInt32 IEEE bits scaled by `2^-64`. The raw bits and classes are decoded and checked; NaN outputs are rejected. Known unused NaN thermodynamic **input** words are retained bit exactly. Artifact Float64 storage is not GPU Float64 computation.

All 20 full artifacts, **780,827,520 bytes**, pass the mandatory field audit, including all 30 model-pair/trial comparisons. Actual input prefix bytes (7,485,048 per trial), guard words and nonfinite classes are exact. Predeclared finite-state budget is `0.002 + 2e-5*abs(reference)`; probabilities use `0.0002 + 0.0002*abs(reference)`, and independent tiny partition controls use absolute `2e-6`. The central comparison envelope alone is insufficient; the hash-bound field audit is required and complete.

All GPU state payload bits, including the complete small-case stage traces, are identical across all four implementations. The three Julia artifacts are entirely identical; native host probability evaluation differs by at most `5.960464477539063e-8` (one ULP). The largest difference from the separate source-CPU state recurrence is `0.00030517578125`.

The full source-CPU replay is a crosscheck of the same recurrence, **not an independent physical algorithm**. An independently expressed enumeration of all seven Boltzmann microstates for two possible nested pairs validates the tiny partition and probabilities, including constraint and no-pair controls. Full probability matrices also satisfy symmetry, forbidden-pair zeros and predeclared probability/row-mass bounds. These are necessary model checks; no full 292-base biological ground truth or experimentally validated structure is claimed.

## Implementations and timings

CUDA.jl uses its own indexing, static shared storage and barriers. KA uses its public local-memory and barrier APIs. JACC uses its own `parallel_for`, `shared` and module-level `sync_workgroup`; the latter is unexported and not listed public, so its availability is pinned to the actual JACC 1.4.0 source hashes. KA/JACC do not call foreign CUDA device helpers. All three are framework device kernels on the selected CUDA backend; other backends are unverified.

Each reduction uses 128 threads and 512 bytes of scratch. The internal-loop mapping preserves the source 8×16 physical lanes and transposed shared-tree index. JACC uses a zero-valued 128-element resource seed; full actual GPU readbacks before and after all trials are retained in its log. This extra seed is not a mathematical input or expected-output copy: every active scratch location is overwritten with freshly computed GPU partial energies. The API source requests a preload/barrier; remaining compiled instructions were not profiled.

Five checked in-process trials exclude JIT by explicit warmup and retain the mean of trials 2–5:

| Implementation | Mean milliseconds |
|---|---:|
| Native CUDA | 26.237185 |
| CUDA.jl | 31.245251 |
| KA | 32.301551 |
| JACC | 69.221422 |

The ROI sums each case's full initialization/diagonal GPU pipeline and device completion. Source-derived dispatch count is 2,701 per timed trial, not a profiler measurement. Resets, transfers, oracle checks, probability postprocessing and the separate trace are excluded. Host launch and completion overhead are included, so these are not kernel-only or causal frontend effects. JACC launch metadata paths, scratch API policy, compiler versions, and the matched 68-byte guard offset are covariates. Registers, local allocation, actual traffic, driver query cost, counters and roofline bottleneck are unknown.

Final result: `20261020T024319667697Z_prna`; `matched_performance_valid:true`. `saved_field_audit.json`, `final_source_binding_audit.json` and hash-bound `study_adjudications.json` retain the stricter scope and evidence.

## Reproduction

All essential CPU assets are deterministic and replaceable. `regenerate_assets.py NEW_DIRECTORY [--native-assets EXACT_CACHE]` fetches exact DVC data if needed, compiles the source CPU serializer, evaluates the independent tiny oracle and checks all nine output hashes. A fresh compiler run and the new orchestration recipe both reproduced all nine files byte exactly on this recorded environment. No raw GPU captures need adding to Git. A different compiler/libm that changes fixture bytes must be reported, not silently substituted.

Copy regenerated binary assets into a fresh checkout's PRNA directory. `prepare_cold_config.py --source-root SOURCE_ROOT --output NEW_CONFIG` relocates all model specifications, resolves exact external JACC source hashes and removes all reuse. Execute the central verifier on physical GPU6 (it locks internally), then `audit_saved_fields.py RESULT --output AUDIT` with NumPy 2.5.4 available. CPU regeneration and fresh-config/external-dependency resolution were executed on the current source root; cross-root execution and a second full cold GPU timing series were not run.

## Effort and token accounting

Each target has one authored device-algorithm bundle and three generic GPU functions/callbacks. CUDA.jl has two whole-program checks: the first fails host parsing before device JIT; a whitespace-only fix then passes its first device-JIT check. KA and JACC each pass their first program/device-JIT check. There are zero device compiler or numerical repairs. Shared native setup/name-collision fixes and the automatic-review generator-command rejection are separate, preserved events. A safer standalone PRNA generator was created after that rejection; the FRNA source hash remained unchanged.

Observed target activity windows: CUDA.jl **4,264,913**, KA **1,654,153**, JACC **3,815,823** total tokens. These include reused context and cached inputs, required verification and coordination; cached/reasoning counts are subsets, not additional costs. They are not exclusive billed costs or historical recovery. KA-specific architecture planning occurred in native preparation, so KA's calibration eligibility is conservatively false while its actual window remains recorded. See canonical manifests and `observed_token_windows.json` for phase boundaries and all component counters.
