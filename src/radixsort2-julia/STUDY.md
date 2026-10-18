# radixsort2: four sorting modes, explicit algorithm boundary

## Scientific and numerical contract

This is a domain-neutral sorting primitive, not a supplied scientific application dataset. The original CUDA code calls **Thrust `sort` / `sort_by_key`**. Source default **N=1,048,576**, seed19937 C++ `std::mt19937` distributions and all four modes are retained: positive finite Float32 or full-range UInt32 keys, each with or without UInt32 original-index payloads. The Makefile's536,870,912-element regime and arbitrary CLI shapes are **unmeasured**. Custom block-offset scan currently targets the default1024groups; this is not a general sorting-library replacement.

All four models pass every case, including full raw key/value order. Independent CPU audit checks **80 final artifacts**: four cases × four models × five trials. It verifies exact sorted keys, the full index permutation, and every key/payload association. Tolerance is0; Float64 artifacts represent UInt32 keys/indices exactly, while GPU integer arithmetic remains UInt32. Prefix lengths are1,048,581 for keys-only and2,097,157 for pairs; complete artifacts contain2,097,157 or4,194,309 values.

The native source checks only key sortedness; pair association/permutation checking strengthens it. Equal-key order is **not canonicalized**. Thrust's API does not guarantee stability, but the current compiled implementation's raw outputs match stable CPU order in every captured trial:21,369 Float32 and138 UInt32 duplicate adjacencies. This is an observation for the pinned library/fixture, not an API guarantee.

## Implementation and timing boundary

Native compiles **Thrust2.0.1**; installed arithmetic-key dispatch reaches CUB DeviceRadixSort. Exact vendor kernel counts, digit policies and workspace sizes are unprofiled. The three Julia versions instead share an authored stable **32-pass binary LSD** algorithm: block-local zero-prefix rank, block-offset scan, then scatter, totaling96launches/sort. Four generic definitions are authored because keys-only and pair scatter differ; three are active per case. Float32 raw-bit order is valid for the source's finite nonnegative inputs, not arbitrary negative/NaN sorting.

CUDA.jl uses own indices/static shared/barriers; KA uses own index/localmem/synchronization API; JACC uses own indices/shared-template copy/barriers. Rank/scan groups have1024threads and4KiB scratch; scatter uses0sharedbytes. JACC has separate explicit4KiB/0B LaunchSpecs, preventing default/cached resource ambiguity. No foreign CUDA device intrinsics are used in Julia variants. CUDA host allocation/completion glue is tested only on NVIDIA.

**Native versus Julia includes an algorithm and workspace-policy confound.** Julia workspaces are preallocated outside the timed sort; native Thrust manages temporary storage inside its API. All time complete sorting work plus device-wide completion, with inputH2Dreset outside each interval. CUDA.jl `Base.sort!` bitonic is an available but unmeasured alternative, not claimed unavailable or equivalent to Thrust. Performance differences against native cannot be attributed solely to abstraction.

Each trial averages the source's100 sort calls, followed by full final-output checking. One complete sort warms/JIT-compiles before five in-process trials; the reported mean uses trials2–5. These are **four retained sample means**, not400independent timing observations. Intermediate99results are not individually serialized/audited; the original also validates only the final result. No outlier removal or fastest selection.

| Case | Native CUDA (ms) | CUDA.jl (ms) | KA (ms) | JACC (ms) |
|---|---:|---:|---:|---:|
| float_keys | 0.695 | 1.526 | 1.573 | 9.083 |
| float_pairs | 0.831 | 1.989 | 2.028 | 9.392 |
| uint_keys | 0.689 | 1.537 | 1.587 | 9.151 |
| uint_pairs | 0.985 | 2.000 | 2.051 | 9.385 |

JACC's roughly9ms versus1.5–2ms for CUDA.jl/KA remains an **unattributed measured gap**; no guessed allocation/shared-memory/occupancy cause is asserted. Integer bitwise scans and ordering are not an ordinary floating-FLOP roofline workload. Actual registers, spills, occupancy and physical traffic require profiling.

## Effort, evidence and restart

Historical Julia marker/cannedPASS is archived. Current CUDA.jl preparation authors the shared custom algorithm; that design/fixture/oracle work is separately costed from portable wrappers. Reference, KA and JACC each have **one program candidate and four actual device-JIT process checks** for the four dtype/mode cases, with0feedback repairs. Four definitions, three active definitions/case,32bitpasses,100inner sorts and five outer trials are distinct from compiler repair iterations. Internal compiler passes/specialization count are unknown. Prior framework/reference/KA knowledge is explicit; this is not a context-free first-program claim.

Prospective usage includes context/cache/reasoning/coordination and overlapping workflow/child windows, not additive exclusive billed costs. Root/maps ownership handoffs are disclosed in the relevant broad/KA window. Source chars/4 proxies remain partial and separate.

Use `make run-all` inside each child folder or run the four `study_config_{float_keys,float_pairs,uint_keys,uint_pairs}.json` files through `tools/verify_portable_julia.py`. All case result paths, source/dependency hashes, fixtures/generator, attempts, usage, independent audit, per-model ontology and paper observation are checkpointed. Root coordinates separate variant commits. Next case:reshapeKVCache; untouched tail transfers are recorded in canonical ownership.
