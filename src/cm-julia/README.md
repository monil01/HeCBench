# CM: connectivity mapping of gene-expression profiles

## Scope and source

This benchmark scores a signed gene signature against drug-reference profiles,
then estimates a two-tailed Monte Carlo p-value, expected false positives (ENFP),
and thresholded significance. It is not a generic matrix multiplication benchmark.

The named contract
`cm_U133A22283_sig31_R4099_two_drugsets_full3GPUstages_hostfeedback_v1`
uses the source's complete22,283-gene universe,31 query genes, two drug sets formed
from three reference profiles (set sizes2/1), and4,099 random signatures. All three
original GPU algorithms execute: integer dot reduction, random connection scores,
and threshold-count reduction. The native adapter includes unchanged original
`process.cu`/`kernels.h`; it calls the original host p-value helper as well.

The original native CLI also ran on a generated full-size gene-profile directory
with the original100,000 random generations. Independent CPU calculations match
its final printed statistics at the original text precision. External biological
cudaMap data is absent; synthetic inputs do not certify biological/clinical use.

A real source defect is retained: `computePValueHelper(50000,100000)` returns0
instead of the independent two-tail result1. All Julia host helpers deliberately
preserve the source branch and actually print that result in untimed diagnostics.
The declared **odd4,099-sample primary cannot encounter exactly50%**. No global
statistical-domain certificate, hidden fix, or tolerance relaxation is claimed.
The strict source `score > threshold` predicate is preserved.

## Evidence

Final receipt: `.porting-state/portable_julia/runs/20261019T192233084931Z_cm/result.json`.
Every one of20 primary captures contains361,102 Float64-encoded values and is
bitidentical across all four implementations. The352,282-value exact input prefix
contains actual device-readback references/query/RNG inputs, parameters, initial
working arrays and guards. Every full random score, integer partial dot/count,
output guard, host-feedback threshold, p-value, ENFP and significance is checked.
All three GPU stage outputs survive and are captured; no overwritten intermediate
reconstruction is claimed. Cross-GPU and independent CPU tolerances are zero.

The independent oracle uses unbounded integer dot/score numerators, exact dyadic
set-size normalization for1/2, Float32 index/score rounding and a separate two-tail
formula. The source C++ `std::default_random_engine(123)` and
`uniform_real_distribution<float>(0,1)` generate fixed input bits shared by all
four versions. GCC/libstdc++ are version/hash recorded; there is no claim to
reproduce the same random sequence in a Julia RNG. All actual selected indices
are in bounds. General near-zero RNG endpoint behavior is not certified.

`audit_saved_outputs.py` fully rereads/rehashes all20 captures, rechecks actual
inputs/math,20 source bindings, the256 actual JACC scratch readbacks, and nine
byte-exact cold-regenerated files. It adds no GPU runs. CPU oracle recomputation
is cached only for identical whole artifact bytes; this remains two drug-set
inputs, not20 independently generated datasets.

## Runtime and implementation classes

| Version | Mean of trials2–5 (µs) |
|---|---:|
| CUDA |79.239|
| CUDA.jl |206.347|
| KA |269.792|
| JACC |323.403|

Each version executes five actual checked in-process trials, with exact warmup
outside timing; discard only trial1 and average2–5. The resident ROI includes six
GPU launches, partial-dot D2H, dependent host normalization/threshold, count D2H,
and host p-value/ENFP/significance. IO, query mapping/profile aggregation, RNG,
allocation, input H2D and final full-score D2H are outside ROI for all four.
This is an integration-adapted scope excluded from untouched-native aggregates;
no native arithmetic is repaired. Original whole-CLI walltime is a different ROI.

All three GPU components use their own framework APIs. Whole implementations
are `hybrid_host_control` because computed dot/count readbacks govern dependent
work/results. KA uses static512B local memory; CUDA/CUDA.jl/JACC use dynamic512B
reduction tiles. JACC.shared adds a128-value template preload/barrier; every local
position is overwritten by the actual partial before reduction. Actual global
template initial/final values are zero. Score kernels reserve0 shared bytes.
These mapping, transfer/view/launch wrappers and unprofiled host allocations are
covariates. No measured roofline, occupancy or causal frontend ranking is claimed.
GPU score computation is Float32 and integer; Float64 artifact encoding does not
imply uniformly Float64 GPU arithmetic. Host normalization/statistics are Float64.

## Attempts and actual usage

CUDA.jl and JACC pass their first candidates/device-JIT processes. KA takes two
program checks: the first warms dot/score GPU kernels, then `count!` host constructor
dispatch fails with a MethodError showing Base's binding. Renaming the entry to
`cm_count!` fixes dispatch without changing arithmetic. Count two actual device-JIT
process checks, one arithmetic-body candidate, one host API repair; the first
count kernel did not reach device JIT. Failed source/stack/proof are archived in
`cm-julia-KA/candidate_history/attempt01_host_dispatch`.

Observed usage and calibration live in STATE `usage_boundaries_300/cm` and
`calibration_537/cm`. Target manifests preserve actual candidates/compiler
feedback and source receipts. CUDA.jl creates shared helpers/host harness, KA then
JACC inherit them. Actual counters include session/context/cache; shared native
readiness, verification and annotation are separate. Cached/reasoning subsets
are not additive, and these are not exclusive billed costs or historical recovery.

## Cold regeneration

Pinned Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0,
nvcc12.0.140, V100sm70; see `environment_snapshot.json` for compiler/stdlib hashes.
Python3.12.3 and g++ regenerate all required CPU fixtures exactly. Regenerable CPU
binaries remain local and need not enter source Git; raw GPU captures stay in
STATE. No irreplaceable external biological dataset is assumed.

```bash
python3 src/cm-julia/generate_assets.py
python3 src/cm-julia/prepare_cold_config.py /tmp/cm-cold.json
python3 tools/verify_portable_julia.py cm --config /tmp/cm-cold.json
```

The cold config relocates source paths, removes all reuse, and runs four fresh
series. The verifier owns the GPU6 lock; do not wrap it in another flock. The
pinned installed JACC dependency path/hash must exist. To reconstruct original CLI
readiness inputs, run `native_diagnostics/generate_cli.py OUTPUT_DIRECTORY`.
`paper_observations.*` records RQ1–RQ5, scope limits, timings and usage windows.
