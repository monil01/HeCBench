# Proposed retained-trial selectors

CPU source review covers nine benchmarks × four models: 36 records and 92 alternatives in `recipes.json`. All initial plan source/config/result hashes still match; current census selection matches those plans. Exact scientific regexes differ by model and are included in every selector. These are source-derived proposals, not executed GPU profiles or recovery of historical retained executions.

Use **global** filtering with `--kernel-name-base demangled`, the record's scientific `--kernel-name regex:...`, and the following matching-family skips/counts. Remove the initial first-per-geometry `--filter-mode per-launch-config --kernel-id ::regex:.*:1` options. Installed NCU help confirms `--launch-skip` counts only matching launches. Do not use `--launch-skip-before-match`. `--kill 0` leaves all later runtime, validation and audit stages executing.

| Benchmark | Trial 2 scientific window | Skip | Count | Scope |
|---|---|---:|---:|---|
| triad | launches 102–201 | 101 | 100 | all 100 independent triads; one-launch alternative also included |
| babelstream | dot launches 12–21 | 11 | 10 | selected dot operation only |
| spmv | dense N1 / 17 / 33 / 257 / 1024 | 55 / 65 / 75 / 85 / 95 | 10 each | separate windows; library kernels excluded |
| jacobi | scientific steps 22–41 | 21 | 20 | all 20 diffusion states; reset/copy/host norm still execute |
| fdtd3d | steps 7–11 | 6 | 5 | all five ping-pong states |
| maxFlops | cases 385–576 | 384 | 192 | complete selected case sweep, including explicit R0 controls |
| md5hash | searches 25–44 | 24 | 20 | five batches of four searches; audit still executes |
| nbody | N1 / 2 / 33 / 257 | 94 / 104 / 117 / 127 | 9 / 12 / 9 / 15 | separate core windows exclude each untimed initial-force replay |
| bfs | paired stages 53–78 | 52 | 26 | both complete traversals: 9 pairs plus 4 pairs |

For maxFlops, nonzero-repeat FP32/FP64 `N512,R2,MAdd4` alternatives are matching launch 475/571 (skip474/570, count1). `maxFlops_cases.json` decodes and hashes all 192 fixture cases and records their runtime parameters and trial 2 ordinals; names alone do not encode repeat count.

For BFS, the CPU reconstruction of the exact selected graphs yields maximum distances8/3 and final-no-update-inclusive pair counts9/4. This is a source/fixture derivation, not an observed GPU launch trace or proof that non-atomic source writes are race free.

## SpMV library stages and complete host regions

No CSR/COO name-only ordinal is certified. A cuSPARSE call can produce several kernels, including shared helpers. The exact proposed source-copy locations, source anchors, trial/case/stage predicates and NVTX start/end API calls are recorded in `host_instrumentation_proposals` and `host_copy_instrumentation`. For example, bracket only trial2/N1024/CSR's existing ten-call batch with the start/end range `retained_spmv_trial2_N1024_CSR`; profile using `--nvtx --nvtx-include retained_spmv_trial2_N1024_CSR` and no guessed library count. Keep all original resets, descriptor setup, vendor calls and validation. Julia k2 maps CSR; native k1 maps CSR. The copy root is `/tmp/roofline-method-review/retained_roi_selectors/proposed_source_copy/src`; no copy, edit or build was performed.

Jacobi may likewise need a host range to inventory its in-ROI error reset/runtime stages. Scientific kernel counters alone omit scalar D2H and host norm work even when every stencil step is captured. Host range timing remains diagnostic.

## Qualification

The source proves call ordering and input/parameter state under the selected contract. A fresh capture must confirm emitted scientific families, multiplicities, case order and geometry; mismatches invalidate positional assignment. Initial mangled/function sampling and source review are not a whole-run launch trace. Cache-control/replay alter hardware state, so input equivalence does not imply identical uninstrumented cache behavior. Instrumented times never replace the original five-trial means. No single point establishes a wholebenchmark causal bottleneck.

`source_excerpts.json` contains numbered source text and full-file hashes. Historical CUDA.jl triad `main.jl` explicit-binding omission remains a historical gap; its current supplemental hash is prospective only. Historical native scratch binaries may have been removed and need a fresh collector build with binary hashes. All commands are proposals; no GPU/build/Git/sudo/shared benchmark edits occurred. `ncu --help` was the sole profiler-tool invocation and performed no profiling.
