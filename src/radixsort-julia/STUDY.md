# radixsort: scientific and implementation observation

The two sizes (32,768 and 65,536 keys) exercise the complete eight-digit SDK radix pipeline, including all six kernel stages per digit. Independent Python integer simulation checks every final key, local temporary key, histogram, local start, global prefix, and top-level scan buffer in all 20 captures. The second fixture has 1,024 lower-28-bit groups, every upper nibble in each group, and four copies of each full key. This is stronger than the legacy monotonic-only check, which cannot prove multiset preservation. All three Julia versions retain the pipeline but use an explicit block scan instead of the native warp-synchronous helper. Thus synchronization adaptation is a concrete performance explanation candidate; no bottleneck or occupancy claim is made without counters.

## Reference audit

Historical CUDA.jl substitutes high-level sort! for the native four-stage SDK radix pipeline. This is an algorithm/mapping difference, not evidence of a wrong sorting operator. Its single sort elapsed time is divided by repeat without repeating the sort, a separate historical S2 measurement defect. Original bytes are preserved; the named study authors the native pipeline and aligns actual repeated work.

## Numerical and performance evidence

Configuration `radixsort_sdk32bit32768_65536_upper_tiecoverage_threepasses_fullscratch_v2`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T002733261235Z_radixsort/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One authored device candidate per model. CUDA.jl has two successful program checks across a fixture-only coverage refinement, not a compiler or numerical repair. KA and JACC each have one successful program/JIT check compiling six authored kernels. Prospective usage windows include verification and metadata, not exclusive billed coding cost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/radixsort-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scope and configuration history

The primary fixture refinement changes upper-nibble tie coverage before either portable translation; both earlier reference series passed and remain archived. It is not a failed translation or a timing selection. All 215,046 output words and 215,060 input/control words are checked per capture. The independent witness checks all 20 captures. Key-only ordering does not certify associated-value stability, arbitrary tails, batches, or the default large CLI.

JACC copies complete shared templates and adds an initialization barrier in every kernel stage. Julia local sorting uses 72 algorithm block barriers versus approximately 25 in native source. This documented adaptation is not evidence of an unsupported framework feature.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
