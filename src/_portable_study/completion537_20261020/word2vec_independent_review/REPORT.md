# Independent review: bounded full asynchronous word2vec property study

## Verdict

No material contradiction found in the declared **property-only, additional workflow** acceptance. This does **not** establish identical trained embeddings, equivalence of every SGD update, convergence/generalization, the original text8 workload, or a pure Julia host application. The existing numerically verified CBOW component remains separate primary evidence; the new workflow must not add another benchmark population name.

This review performed no GPU runs, benchmark source/state/shared edits, package changes, or Git mutations. It inspected source and receipts, compared immutable Git blobs, and ran the existing CPU-only saved audit with its raw-file writes intercepted. The only newly written outputs are under this review's `/tmp` directory, apart from the explicitly requested prospective usage-boundary records written by the existing capture tool. No benchmark tests were added.

## Recomputed evidence

The existing `audit_saved.py` was invoked with final result `20261020T155223860367Z_word2vec` and output `recomputed_audit.json` here. Its resulting JSON is **identical** to the committed `full_saved_property_audit.json`:

- 20 retained trials, 415 raw files completely rehashed; all property checks pass.
- All 20 complete final `syn0` arrays have **different SHA-256 hashes**.
- Complete `syn0`, `syn1`, RNG, actual callback input/alpha/RNG traces, unit-control arrays, vocabulary and binary vector exports are present and bound to per-trial `properties.json` and the structured trial's audit hash.
- All 73 execution dependency bindings match current source bytes, independently checked against `source_binding_audit.json`.
- The common 400 MB actual-device input table is byte-compared during every original observation; its canonical file remains completely rehashed and monotonic/RLE/entry-distribution checked. It is not replaced by a checksum-only numerical certificate.

`receipt_and_metadata_review.json` contains all 73 source checks and the exact Git/metadata comparisons below.

## The certificate does not stand in for raw model identity

`protocol.json`, final model rows, proofs, README, and paper observations explicitly set numerical/raw-model identity claims false and use `property_equivalence_not_identical_raw_values`. The central comparison artifact contains 272,624 losslessly encoded **static-input values**, including initial weights/RNG/sentence state, table histogram and vocabulary counts. Its equality is not an equality test of trained models or endogenous consumed sentence schedules.

The acceptance is stronger than this static artifact alone: `run_native.py` and selected `run_target_v2.py` require every warmup and trial's independent `check(...)` to pass, bind the generated property audit SHA into the emitted row, and exit nonzero if any check fails. Saved verification rehashes complete raw files against those same audit hashes before recomputing every property. There is no PASS obtained solely by comparing constant certificates while ignoring absent/failed raw model checks.

The full saved raw checks independently establish:

1. Exact initial LCG weights, zero output weights/sentence state, libc-initialized RNG, full sigmoid lookup-table accuracy, actual table distribution and actual-device byte comparison.
2. Full final model finiteness and actual change from initial values, plus complete binary embedding export equal to the observed final `syn0` bytes.
3. All callback records present, execution-order indices forming a permutation, valid actual consumed IDs/positive bounded alphas, complete before/after per-word RNG chain and final RNG, with 26 LCG steps per sentence.
4. A fresh one-position/one-sentence control: full updated model arrays against independently expressed Float64 mean/dot/gradient evaluation with the actual lookup/clamp/sampler policy and declared `1e-6 + 2e-5*abs(expected)` budget. State is explicitly restored before training.
5. Full fixed 1024 cyclic-context expected-NCE diagnostic, including exact UInt32 modulo bias/fallback and skip-positive distribution; predeclared normalized improvement >=0.0001. Observed improvements are around0.65.
6. Actual allocation guards in every retained trial: five source allocations for native and six including the mapped global exp table for targets. JACC's separate extra template is observed at whole-process boundaries only, correctly disclosed rather than counted as per-trial evidence.

These properties reject nonfinite/no-learning/missing-trace/invalid-RNG/guard/export failures; they do not uniquely identify the correct stochastic trajectory. The fresh gradient control starts with zero output embeddings and is a limited local control, not an independent full asynchronous SGD replay. Source inspection and the prior separately verified hot-state CBOW component provide complementary fidelity evidence, without converting the new full-workflow contract into raw-output numerical agreement.

## Source, concurrency and ROI

The selected target provider `study_bridge_v2.cu` caches the trace counter's host pointer before training; its `broker` performs no per-request CUDA API call. It acknowledges enqueue after the Julia dispatcher launches work, rather than waiting for each SGD kernel's completion. Original host sentence/alpha races, source plain non-atomic model writes, and their scheduling sensitivity remain; the provider does not make training race-free. Original synchronous transfer/default-stream semantics are retained: “asynchronous” does not prove overlapping GPU kernels.

The former `study_bridge.cu` performed a synchronous per-request `cudaMemcpyFromSymbol`. The prior CUDA.jl/KA property-passing timing series remain preserved and are hash-bound excluded in `src/word2vec-julia/study_adjudications.json`. The selected provider and fresh target series correct integration/measurement semantics, not mathematical equations. Native source/five-trial evidence is unchanged and reused.

All four ROI boundaries cover the named source host training pipeline (vocabulary/table preparation, allocations, host threads, training, output writing), subtracting explicitly timed observation blocks; final completion is enforced. Unit-control evaluation, capture/readback and CPU property checks are outside the retained ROI, while in-kernel telemetry and target queue/polling remain genuine measured covariates. Physical sentence initialization, global exp-table mapping and extra group barriers are disclosed interventions. These timings cannot support untouched-source or causal pure-frontend rankings.

Mean trials2–5 seconds independently reproduced from retained rows:

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
| 0.85490386525 | 1.55147371625 | 1.45419408875 | 1.69250484 |

## Committed-source receipts and historical metadata

Nested actual main bytes match current bytes AND recorded execution hashes in all three cited commits:

- CUDA.jl `9a1ca9dd14000e85fd8e36477d110dd1275ea275`: SHA `01b18fceb9db1ff8f8e2b652658b82e5341c760bea9b3cad06e437b1938b290b`.
- KA `91a1df5ab59e4d595c1777df580c3559fd11978d`: SHA `42310154e069a4c4d0d17200fde8c10754bfee187f74d35ad16f3427e4734424`.
- JACC `d9ce72d33a96695dd2f6ec14cef50be5716a6a5b`: SHA `0200370c18f03c98ee88aa600b261e0336ac09753c89115a432290cb897b7ca5`.

Compared all nine canonical original metadata records with `src/_portable_study/completion537_20261020/word2vec_original_metadata.json`: every original field is unchanged, including old candidate/check counts, measured source hashes, source commit receipts, times and token records. Only `additional_workflow_studies` (manifests/statuses) and `additional_workflow_verification` (proofs) were added. The workflow's new counts and token windows remain in its scoped packet.

## Actionable descriptive refinement (no acceptance/source repair)

`implementation_traits.json` uses the descriptive raw class `framework_kernels_native_CPU_controller_CABI_queue_CUDA_telemetry` for all targets. Preserve that raw label, but in normalized ontology use the whole-workflow native-host/CABI mixed control classification and distinguish device components:

- CUDA.jl's `CUDA.atomic_add!` telemetry is within its own framework.
- KA/JACC explicitly call `CUDA.atomic_add!` for the trace execution counter. Their device component therefore has a **CUDA device extension / hybrid_device_intrinsics**, limited to telemetry; the mathematical SGD body uses the owning frontend's primitives.

This is model-relative metadata precision, not an algorithmic or numerical failure. It does not require retiming or changing accepted source. The current label already discloses CUDA telemetry, so it is not a hidden pure-framework claim; ensure the paper's normalized class layer does not silently drop this detail.

## Cost interpretation

The scoped effort summary preserves original component costs and new workflow windows separately. CUDA.jl creates the shared broker; KA/JACC reuse it. KA overlaps a shared integration-repair window and a CUDA.jl completion window: those counters cannot be summed as exclusive target costs or clean frontend rankings. No historical costs or per-target exclusive costs were invented.

This review has its own combined shared `independent_async_property_review` window, BEGIN `87a8b5198ac7e138488df7de`. Its separately captured END and raw counters are in `usage_receipt.json`; no retrospective per-model allocation is made.
