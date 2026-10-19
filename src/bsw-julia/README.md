# BSW protein local alignment — Julia study

## Contract and evidence

Source: paired protein local alignment with BLOSUM62 and affine gaps (open−6, extension−1). Each pair uses a cooperative anti-diagonal GPU DP with register states, warp shuffles, shared boundary/spill storage and barriers. The forward pass finds score/endpoints; **actual endpoint readback controls reverse launch geometry**, then reverse-prefix DP finds starts. No CPU DP replaces the GPU algorithm.

Named synthetic fixture: 65 positive pairs, maxreference160/query128, global65blocks×128threads,780dynamicshared bytes, sourceits1. Primary A≥B; one separate **untimed swapped-orientation diagnostic per model** exercises A<B with unchanged device bodies. The original native CLI also matches the independent full-matrix oracle on all65scores/fourcoordinatefields. External DVC protein regressiondata is absent. No biologicaldataset/traceback/applicationaccuracy claim, >20k batching, empty/zero-score inputs, arbitraryhostqueryallocation or overflow/longsequence certification.

Final primary bundle: `.porting-state/portable_julia/runs/20261019T182044883828Z_bsw/result.json`.

All20primary actual captures are bitidentical: **23,167 Int32-encoded values/trial**, including22,682 exact actual-input/forward-host-feedback prefix values (90,728 bytes), all65scores/fourcoordinates and allguards. Packed ASCII, offsets and tables are captured from actual GPU buffers before/after; initial output buffers are defined. Actual forward endpoint host copies are included. Only final surviving state is claimed, not overwritten register/DP intermediates.

The independent oracle uses full row-major H/E/F matrices in Python integers, BLOSUM62 input tables and declared source endpoint ties (forward min(longerrow,shortercol); reverse min(shortercol,longerrow) in consumed-DP coordinates). Forward/reverse optimum agreement, allcoordinates/bounds and guards are checked. Maximum possible score11×128=1408 fits sourceInt16. All primary/supplemental values match exactly; no tolerance relaxation. `saved_output_audit.json` fully rehashes/rechecks20primary captures,12sourcebindings, four full supplemental captures and11exact cold-generated files. Mathematical DP is reused only for identical whole-artifact bytes; no claim of20distinct input sets.

## Timings and implementation boundaries

Five checked trials/model, warm exact complete pipeline, discard only firsttime, average2–5. ROI: sequence-vector copy, actual cumulative prefix reconstruction/per-sequence bytepacking, H2D, forwardkernel, endpointD2H/completion, hostmaximum/minimum decision, reversekernel, start/scoreD2H/completion. Allocation/constants/verification outside; diagnosticprintf excluded all4.

Mean: native477.805µs, CUDA.jl785.997µs, KA726.874µs, JACC764.574µs. These are full named host/GPU pipeline observations. NativeC++deep string-vector copies versusJuliaimmutableStringvector copies, runtime transfers and framework launch wrappers differ; allocation counts/counters are unmeasured. **No kernel-only, causal frontend, or original production throughput ranking.**

The first native-only adapter precomputed offsets/packed arrays and omitted source hostpacking work. Its five numericallycorrect captures/timings and source are retained, hash-bound excluded by `study_adjudications.json` before selecting the corrected boundary (not speed). OriginalCUDAdevicekernel unchanged; nameddriver integration is explicit/excluded from untouched-native performance aggregates, not a mathematical native repair.

Wholepipeline class: `hybrid_host_control`, because numerical endpoint readback controls subsequent device work. GPU component:

- CUDA.jl: `framework_device_kernel`, own CUDA.jl warp/shared/barrier API.
- KA/JACC: `hybrid_device_intrinsics`, explicit CUDA warp ballot/shuffle/shared/barrier components.

KA uses actualKA.allocate and KA-generatedkernel/context. The pinned CUDA backend call has no dynamic shared-memory keyword; an explicit CUDA launch preserves780bytes (`ka_backend_boundary.json` binds actual installed source). JACC uses JACC.array/parallel_for with explicit65blocks/128threads/shmem780/syncfalse. These are CUDA-backend integrations, not claims of cross-backend portability or frameworkunsupported features. Formal source warp/shared-memory racefreedom is unproved by numerical agreement. Registers, occupancy, compilerpadding and causal bottlenecks are unknown. Integer DP is not assigned a floating-point roofline coordinate.

## Effort and tokens

CUDA.jl: two programcandidates/checks; the first generator-spacing ParseError occurred **before deviceJIT**, preserved under childcandidate_history; the unchanged devicebody passes its first actualdeviceJIT. KA/JACC firstcandidate/deviceJITprocesscheck each. One authored algorithm kernel/model, three devicehelpers, zero observerkernels. One additional untimed orientation-validation process/model is separately counted under sharedverification (two launches), with no added primary timing series.

Shared native timing-harness correction and offline audit integer-buffer-decoding correction remain separate events. Actual usage boundaries include context/cache/session reuse and coordination. Native readiness, each ordered translation, verification and annotation are separate; CUDA.jl creates shareddevicehelpers/hostdriver, KA/JACC inherit. No exclusivebilling or historicaltoken recovery. Canonical manifests/calibration/status/traits and `paper_observations.json` retain RQ1–RQ5 notes.

## Cold reproduction

Python3.12.3 stdlib regenerates all fixtures and full-matrix CPU oracle from originalsourceBLOSUM62constants plus deterministic named sequences:

```sh
python3 src/bsw-julia/generate_assets.py
python3 src/bsw-julia/native_diagnostics/prepare_swapped_orientation.py
```

The two CPU-generated oracle binaries are92,668bytes each, exactly reconstructible; no irreplaceable binary input is needed. They may stay local if the source artifact retains recipes/hashes and explicitly requires regeneration. Raw GPU capture binaries are local-only and **not source assets**. Preserve original `src/bsw-cuda` plus all parent `candidate_history`/`native_diagnostics` recipes/sourceproofs and childfrontendcandidatearchive.

Pinned projects: Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0, localCUDA12.0/native nvcc12.0.140, GPU6V100sm70; requirements/manifest hashes should be interpreted with the environment snapshot. No external DVC download is needed for the declared fixture.

Relocated fresh primary measurement:

```sh
python3 src/bsw-julia/prepare_cold_config.py --source-root /path/to/source --output /tmp/bsw.json
python3 tools/verify_portable_julia.py bsw --config /tmp/bsw.json
```

The runner internally acquires the shared GPU lock. **Do not outerflock the runner.** Coldconfig removes historical-series reuse. Fresh supplemental diagnostics require native build plus each selected pinned Julia project; source commands/proofs are retained in `native_diagnostics` and the saved audit, and run under GPU6/sharedflock. They are untimed checks, not primary performance series.

Read-only saved-evidence audit (requires local historicalcaptures):

```sh
python3 src/bsw-julia/audit_saved_outputs.py .porting-state/portable_julia/runs/20261019T182044883828Z_bsw/result.json
```

Regeneration reproduces CPU fixtures/source, not historical GPU timings. The coordinator alone exports shared ontology/paper data and commits each verified version separately.
