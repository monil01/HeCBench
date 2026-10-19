# CC: complete five-stage ECL connected components

The source algorithm uses a first-smaller-neighbor initialization, thread-granular low-degree CAS linking, medium-degree warp work queues, high-degree block work queues, and final root flattening. All five stages are retained in each implementation. Original CUDA is immutable and the native harness includes its actual device bodies. No native arithmetic/algorithm repair is used. Source licensing and attribution are in LICENSE.source and the original cc-cuda/main.cu header.

## Named scope and checks

Three deterministic undirected CSR graphs have1,65 and4099 vertices (largest12284 adjacency entries), including isolates, chains and both medium/high degree paths. The source V100 occupancy-derived grid is640 blocks×256 threads. R100 matches the source Makefile. The largest actual worklists contain32 medium-queue vertices and1 high-queue vertex. Degree counts alone are not worklist counts: root vertices are skipped.

Every trial captures37,916 Int32 values, including33,559 exact prefix values from complete before/after device CSR buffers and actual device scalar echoes, every final component-minimum label and all input/output/worklist guards. Independent Python BFS establishes the unique minimum-ID component labels. All20 artifacts are bitidentical. The supplementary saved-output audit reconstructs BFS solely from actual device-input readbacks, without loading the expected-label assets.

Actual untimed full intermediate parent arrays were captured for every stage and shape in all four models (60 stage captures). Every parent remains within its independent BFS component and is at most its vertex ID; final states match BFS minima. Concurrent intermediate/worklist ordering is not required to be identical. These traces are new untimed executions, separate from the five primary timing trials. Primary trials capture final overwritten outputs, not all R100 intermediate states.

Original full CLI on the generated4099-node ECL binary graph reports PASS. The original external delaunay dataset is absent; no production dataset or graph-scaling claim is made.

## CPU assets and cold reproduction

Python3.12.3, standard library only:

```sh
python3 src/cc-julia/generate_assets.py /tmp/cc-regenerated
```

The generator writes nine per-case CSR/label binaries, the complete oracle_i32.bin, study_cases.h and study_plan.json. Actual isolated regeneration reproduced all12 files exactly; see generation_reproduction.json and study_plan.json for hashes and byte counts. Copy generated files into this directory only when rebuilding missing assets. The independent oracle is a queue BFS over undirected adjacency sets, rather than the source's concurrent union-find algorithm. Guard and exact-prefix expectations are generated alongside it.

From the source repository root, after recreating assets:

```sh
python3 src/cc-julia/prepare_cold_config.py --output /tmp/cc-cold-config.json
python3 tools/verify_portable_julia.py cc --config /tmp/cc-cold-config.json
```

The helper relocates source/project paths and removes saved-series reuse. The verifier uses the shared physicalGPU6 lock internally; **do not apply an outer flock**. Native builds in an isolated working directory with all explicit runtime assets. Pinned projects are src/_julia_ka_env and src/_julia_jacc_env. The same unchanged projects were introspected after NN: Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0 on V100sm70 (see nn-julia/environment_snapshot.json). CC native CLI reports80 SMs and2048 maximum threads/SM; the native adapter asserts the declared grid640.

Each model warms the complete R100 host launch loop, then runs five checked trials. Discard only trial1 time and average2–5. Reported time is the **sum of three per-graph full-five-stage pipeline means**, including launch/completion and excluding allocations, copies, independent checks and serialization. It is not whole-application time or a single-size throughput. All raw timing samples are retained.

The independent audit can be rerun while saved captures remain locally available:

```sh
python3 src/cc-julia/audit_saved_outputs.py
```

`probe_stages.py` runs supplemental serialized untimed GPU diagnostics; it does not rerun primary timing trials. Its first harness-string transport attempt failed before device JIT and was preserved; Base64 transport fixed quoting without altering measured sources. Raw stage logs and GPU captures remain local, outside Git. Their hashes are provenance references, not a mechanism to recover missing data.

## Mapping and effort qualifications

CUDA.jl uses its own CUDA device APIs (framework_device_kernel). KA/JACC use their framework kernel entrypoints with explicit CUDA CAS/add, indexed warp shuffle, static shared storage and block barrier helpers (hybrid_device_intrinsics). This demonstrates a CUDA-backed implementation, not framework-only or cross-vendor portability. Common device helpers and host harness were authored during the CUDA.jl phase and reused later; this is recorded in actual-token calibration strata.

Four native CUDA device globals become an explicit4-entry Int32 device counter array. Native launch_bounds256/minimum8blocks hint is not explicitly replicated in Julia; actual registers/occupancy were not profiled. Installed CUDACore atomics lower through LLVM acquire_release/acquire, while actual native instruction ordering remains unprofiled. The observed Julia9–10× composite timing gap has no established causal explanation. No bandwidth or roofline claim is made for this integer irregular pipeline.

CUDA.jl's first candidate actually reached device execution and threw InexactError because its indexed shuffle source lane is1-based; both native lane0 broadcasts must map to Julia lane1. Candidate2 corrected that API indexing with unchanged fixtures/tolerance. KA/JACC each passed their first candidate/check. There are five sourcealgorithm device definitions plus one untimed input observer per target. Supplemental verification processes and harness failures are separate from translation candidate/JIT counts. See prospective usage/calibration records and paper_observations.json for RQ1–5.
