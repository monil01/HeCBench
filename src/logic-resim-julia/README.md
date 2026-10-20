# logic-resim: full-graph timing-aware digital logic study

## What is translated and checked

The source is an NVDLA **gate-level circuit re-simulator**, not numerical neural-network inference. Named contract `logic-resim_fulltoy_109batches_300Mps_sourceCPUfallback_20MEventarena_v1` executes the entire original140,371-net,101,848-gate graph:23GPU batches85,402gates in actual CUDA.jl/KA/JACC device code, followed by the original C++86CPU batches16,446gates through an explicit host-only CABI component. CPU fallback, VCD/header decoding and SAIF text frontend are **retained/not translated**; no whole pure-Julia/portable CLI claim. Source graph preprocessing is supplied by the original intermediate input file, not replaced by a manually selected active subset.

The original CLI supports an end-time argument. Primary window0–300,000,000ps retains the complete graph and109batches. Original full2,972,036,001ps CLI/SAIF and all-wave diagnostics are preserved separately, not four-way certified. Actual CPU-decoded initial inputs are byte identical to the original source. Each GPU batch consumes actual prior GPU outputs; the independent oracle is read only after computation for validation, never as kernel input.

Final receipt `.porting-state/portable_julia/runs/20261019T224741519352Z_logic-resim/result.json`. All twenty49,972,265-UInt64 captures (399,778,120B each), including33,498,270 input-prefix values, are bit identical. Every actual GPU input event/guard and gate metadata field, all23 complete GPU output histories/control fields, all5,545,944 final events on140,371nets, and421,113 SAIF time-in-state fields are checked. Independent CPU event-priority-queue/truth/delay/pending-transition evaluation exactly matches actual original source waveforms. A producer-order audit proves all GPU output histories carry unchanged into the independently checked final waves. T0+T1+TX equals300Mps on every net.

UInt64 timestamps, MAX sentinel and guard values stay exact: **no Float64 timestamp encoding**. Physical native pointer values are converted only into equivalent arena offsets, with actual GPU gate/readback mapping checks. All time/state fields and event ordering are preserved. C++/Julia Event structures retain16B layout; unused struct padding and unused allocation capacity are not numerical outputs. Event histories include their full live sentinel fields. Artifact wire-ID order is declared serialization; raw SAIF text serialization is not translated or claimed byte identical for ports.

## Source semantics and qualifications

Source output begins X until a postinitial input transition; truth/delay/inertial cancellation policy is preserved. This is source-semantic fidelity, not a certificate of canonical time0 settling or general EDA quality. Source SAIF combines X and Z into TX; TC/IG are literal zero, not measured switching counts.

The source parser allocates packed truth bytes with `new char[]` then `|=` without initialization. The actual original GPU table matched every source truth byte in the observed run. A controlled actual CPU bit-packing witness with poisoned initial storage corrupts14,432 used bytes; this is a latent host-initialization hazard, **not an observed original GPU wrong-wave/fault**. The CPU input-only decoder zeroes unused parser scratch and skips a GPU upload it never uses. Its output is byte identical to original actual initial wave inputs. No GPU computation stub or fabricated benchmark pass is used. Native GPU body and retained CPU methods are exact source excerpts; full study input tables are deterministically specified from source text. Native arithmetic is unchanged, integration/resources adapted, excluded from untouched-native aggregates.

## Performance boundary and implementation class

Five checked in-process trials after full pipeline warmup; mean trials2–5: native143.540112ms, CUDA.jl261.57571075ms, KA260.11309ms, JACC264.72537025ms. Region sums23 metadata-H2D/simulation/flag-D2H/host-control intervals; source C++ CPU fallback, actual-input readbacks, captures and mathematical checking are outside timers. Source original printed “2 loops” means one actual GPU launch per batch (counter starts1 and increments after completion); the study records23 actual launches/trial.

Whole measured implementation is **hybrid host control**, because computed overflow flags decide termination/possible growth/relaunch. Device components are **framework device kernels**: own entry/index/allocation APIs plus generic Julia Event/private MVector6 operations, no foreign CUDA device intrinsics. Original C++ CPU fallback/bookkeeping library contains no GPU implementation/runtime imports, independently checked in `host_component_proof.json`.

Native `gpu_GATE`344B AoS pointers become target field-major UInt64 function/6offset and UInt32 72delay arrays; Event16B layout remains. All use a common preallocated20MEvent arena320MB and16Event guards per end (512B/logical buffer). Native physical pointer-vector formation and Julia hostview/offset-copy wrappers differ inside the metadata/control region; target metadata allocation differs. Input readbacks can warm/evict caches. JACC's selected host-array arena constructor adds an untimed320MB host copy; direct `JACC.array(T,n)` exists but is an unmeasured alternative, **not unsupported**. First valid immutable series is preserved, no fastest-run selection. No pure frontend/kernel-only causal ranking. Native compiler48regs/96Bstack/zero spills; target resources unknown, no inferred bottleneck/FP roofline position for integer-event work.

Primary source data uses one actual pass/batch; general overflow/resume and original dynamic-allocator performance are not certified. Growth code is present, but no unexecuted branch is counted verified. Full graph/time-window results above remain valid scoped evidence.

## CPU-only cold reproduction

Original graph/VCD input assets are already tracked under `src/logic-resim-cuda/Simulation/toy`. Generated binary inputs/oracles total about184MB and regenerate exactly; keep recipes/hashes instead of new binary copies in Git. Raw GPU/source waveform captures remain local with hash/path provenance. Source MIT notices are in `LICENSE.source`.

```sh
python3 src/logic-resim-julia/cold_regenerate.py --materialize
python3 src/logic-resim-julia/prepare_cold_config.py --output /tmp/logic-fresh.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config /tmp/logic-fresh.json
```

Cold regeneration compiles the source CPU decoder with the unused GPU-upload side effect removed and unused dispatch sections discarded, never constructs a GPU Simulator, and invokes no GPU kernel. Original CUDA headers are needed as declarations by retained source headers; g++ and source files suffice at runtime. Independent event oracle uses pure integer C++, no GPU output and no NumPy floating arithmetic. The cold audit reproduced12 inputs/metadata/source excerpts byte for byte. The config helper relocates source paths and removes historical reuse. Pinned CUDA/Julia projects in `environment_snapshot.json`; verifier owns the GPU6 lock internally, do not add outer flock. Saved audit uses NumPy2.5.4 only for exact UInt64 memory-map/chunk comparisons.

## Feedback and actual token windows

First authored target candidate/program GPU-JIT check passed for each model, zero target repairs. Shared native preparation includes source/independent-oracle/arena/CABI architecture and controls (14.48M actual context-inclusive tokens), separately metered. An oracle candidate initially appended sentinels for CPU-only AddC lines which source skips (2327 extra END records); corrected before primary. A saved-audit helper initially inferred Float64 for a mixed Python list containing2^64−8; explicit UInt64 corrected its false guard failure, with no source/GPU/criteria change. These are separate scientific/harness effort, not target compiler failures.

Native preexecution build/namespace/static guard reviews are recorded in `reference_effort_537.json`; one namespace error exists only as tool-return/descriptive evidence, no claim of a retained standalone raw log. Earlier preexecution source drafts are explicitly **retrospectively reconstructed**, not falsely claimed contemporaneous snapshots. No actual GPU validation of that draft is claimed. No OS/system package update or sudo command was invoked.

Actual prospective input/cache/output/reasoning counters, configured model, marker lag and calibration adjudications live in canonical STATE and `observed_token_windows.json`. Cached input/reasoning are subsets; inherited context/coordination count. CUDA.jl creates the shared Julia GPU body/driver; KA then JACC reuse it plus pre-existing source C++ components. These are current ordered-workflow activity costs, not independent clean-room or exclusive billed translation cost, and cannot recover unknown historical consumption.
