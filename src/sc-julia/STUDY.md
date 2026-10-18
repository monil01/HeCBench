# SC: source in-place compaction with CPU/GPU coexecution

All four versions pass five complete captures: 786019 Int32 words/trial, 565314-word actual CPU/GPU logical-input readback prefix, every retained stable-filter value/count, complete CPU/GPU flag arrays and120 guards. Independent direct stable-filter/tile-prefix reconstruction repeats full byte checks. Only the complete valid compacted prefix is an operator output; unused in-place tails are not claimed identical. Named inputs use original positive index+2/zero domain, not arbitrary application datasets.

## Algorithm and scientific scope

Historical Julia genuinely compacted values using flags/accumulate/scatter, but ignored CPU alpha and source in-place protocol: P1 mapping, not S1 wrong mathematics. Restored32 private words/thread, source reduction/shared-prefix scans/global atomic prefix flags, persistent4-block loop, plus unmodified source4-CPU-worker algorithm. All three whole pipelines `mixed_host_device_pipeline`; authored device component `framework_device_kernel`, source CPU component shared explicit C++ host interoperation. No vendor compaction fallback, no CUDA device extension in KA/JACC.

Fifteen cases: full tiles8192/32768/98304 ×alpha0/.1/.5/1 plusGPU-only1/33/8193 tails with nonzero last logical value. Defined padding zero, eight guards/case. DefaultDP=no/REGS32; DP=yes/system atomics/dynamic scheduling, default1024-block large CLI and original105 timed iterations remain unmeasured, not unsupported. Four blocks selected before measurement limit named spin-progress exposure; no general oversubscription/cross-backend progress/race-freedom proof. This is a filtering primitive, not a physical simulation validation.

## Timing and mapping covariates

Named preallocated compute ROI includes GPU launch, source main stdthread and4 workers, source work, full GPU completion and CPU join. Reset/H2D and finalD2H/source concatenation outside ROI; original reset-inclusive CLI performance unmeasured/excluded. Actual CPU/GPU input readbacks occur before in-place overwrite. Per-case means record exact discrete cuts/tasks/counts. Do not interpret differing alpha work partitions as GPU-only throughput. Shared CPU implementation compiledg++13.3 across all4, not Julia CPU lowering; CPU affinity/contention uncontrolled.

257 shared Int32/1028B everyfrontend/native; CUDAjl/KAstatic versus native/JACCdynamic allocation. JACC adds257-word global template copy/barrier even for idle blocks. Native GPU Float32 partition-cut computation hoisted to host in Julia; all named cuts and flags exact, but scalar instruction work not identical. Native CPU workers still compute their Float32 cut inside ROI. ExplicitVal32/Cartesian32 expansions preserve native unroll requests; physical registers/ISA/roofline bottlenecks unprofiled.

## Effort/provenance

Initial CUDAjl andKA numeric-valid series retained nonprimary; ordinary store loop lacked explicit native pragma-unroll request. Shared body corrected to static32 count/store expansions, freshCUDAjl+KA series, no mathematical/compiler/numerical failure or faster selection(CUDAjl aligned mean slower). JACC first body/process passes. Math algorithms1/model; source-lowering candidates/processes2 CUDAjl/KA,1 JACC; frontend wrappers1. Shared CPU CABI/reference costs separate, conditional reuse not from-scratch targets.

Final provenance audit found owned nativeharness indirectly compiled by build script lacked individual dependency guard, though unchanged/original source+hostshim+buildscript/fixtures bound and initial parent digest retained. Corrected explicit dependency and freshnative5only; reused three unchanged Julia bundles. First validnative series preserved nonprimary, no source/tolerance/time change or timing preference. All raw captures/configs immutable. This is sharedT1 bookkeeping, not a device failure. Seven sharedCPU CABI builds and two native adapter builds/processes; Julia backend compiler invocation totals unknown. Prospective nonexclusive cached/repeated-context activity includes science/source/metadata/waiting/coordination, shared unroll window covers CUDAjl+KA and is not exactly attributable to KA alone. Workflow overlaps phases; historical cost unknown, main char4 proxy partial.

Final proof `.porting-state/portable_julia/runs/20261018T122436711877Z_sc/result.json`; canonical study_config_jacc.json; independent_capture_witness.json; per_case_performance.json. All five correctness checks retained, mean trials2–5, warmup excluded. Sources frozen.
