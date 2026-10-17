# local-ht: conditional local genomic assembly study

The five K21 read fixtures exercise supported extension, an equal-support fork, rejection of low-quality reads, low-only support, and a repeat cycle. An independent Python read-window Counter and graph walk checks every occupied k-mer count, quality count, selected extension, actual GPU hash, visited-key record, and full initialized walk/temp buffer in all five trials of all four models. There are 3,211 exact serialized input words and 12,780 canonical output words per capture. Raw table slot order and device pointer identities are intentionally removed by host canonicalization. Termination flags are inferred by the independent oracle, not separately exported by the GPU.

## Conditional reference and scientific scope

The original CUDA broadcast uses `threadIdx.x & warpSize`, so odd warps have no initialized lane-zero value. Its minimal original GPU probe happened to match expected outputs; there is no observed original numerical failure in that probe. The measured CUDA copy changes only that expression to `threadIdx.x & (warpSize - 1)`. This is a conditional repaired-reference stratum, excluded from untouched-native aggregates. Output scratch buffers are initialized outside timing for complete diagnostic capture. Original CUDA and historical Julia sources remain preserved.

The actual integrated right-walk algorithm retains warp-parallel Murmur hashing, CAS publication, matching address cohorts, depth-aware quality decisions, visited-key tracking and iterative-k control. Native pointer/AoS records become Julia offset/SoA records, an explicit performance mapping adaptation. Left reverse complement, stitching, dataset parsing, other K values and short reads remain unmeasured. A static short-read offset problem is documented separately; no full-genome assembly or general race-freedom claim follows from these fixtures.

## Framework boundaries and compiler feedback

All three Julia models use an explicit foreign NVVM match-any intrinsic, so all are classified as CUDA-specific hybrid device mappings. CUDA.jl required two runtime repairs: one-based shuffle lane indexing, then a maxregs=128 cap after a 160-register launch failure at 512 threads. KA passed its first full candidate using its public static 512-thread workgroup, whose backend supplies a maxthreads compiler bound. Public JACC similarly failed register resources; candidate 2 invokes the pinned JACC index wrapper through its private CUDA extension with maxregs=128. It preserves the JACC device wrapper and algorithm. This is a compiler-control gap/workaround, not algorithm unsupported. The default failure remains archived. No counter measurements establish the cause of observed timing differences.

## Timing and effort

Each trial times one integrated right-walk kernel with synchronization. Allocation, reset, host canonicalization, output transfer and the extra GPU hash-capture diagnostic are outside the timed region. Five trials are retained and trials 2–5 averaged: CUDA 135.814 microseconds, CUDA.jl 305.34775, KA 9501.43675, and JACC 330.083. The large KA result is retained without a preferred rerun or an unmeasured bottleneck explanation.

Shared effort includes fixture-generation parsing repair, a strict reuse guard rejection, and two feature probes. CUDA.jl has three full candidate/check processes, KA one, JACC two. Prospective activity windows include repeated/cached context, tool feedback, coordination, waiting and annotation; they are not exclusive coding or billed cost. Source-size proxies are separately partial and omit shared algorithm code.

Result: `.porting-state/portable_julia/runs/20261017T220520566506Z_local-ht/result.json`.
Reproduce with `python3 tools/verify_portable_julia.py --config src/local-ht-julia/study_config_ka.json`, followed by the JACC configuration. Preserve saved evidence rather than rerun for metadata alone.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
