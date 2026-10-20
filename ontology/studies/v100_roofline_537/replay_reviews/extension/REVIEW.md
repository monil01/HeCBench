# Replay safety extension

CPU-only source/event/memory review, all16 requested model slots. Safe is conditional static per-kernel replay state assessment, not fullROI/counters validation or unconditional execution permission.

## SSS event and memory proof

- Delete stream: host packed delete upload → observer input snapshot → record delete_ready → initialize delete output → delete kernel → observer output snapshot → delete D2H → delete stream synchronize.
- Add stream: host packed add upload → wait delete_ready → observer add input snapshot → initialize add output → add kernel → observer output snapshot → add D2H → add stream synchronize.
- Kernel read/write sets: Delete reads D, writes only O_delete plus CTA shared state. Add reads D and A, writes only O_add plus CTA shared state/templates read-only. There is no read of O_delete from Add and no read of O_add from Delete.
- Event recorded before Delete kernel is sufficient: it orders shared D input readiness, not dependent output. Both synchronizations precede function return/next input repack. Host graph-score work between synchronizations operates CPU graph storage, not device D/A.
- Julia driver verifies callback stream handle against registered streams. CUDA.jl passes stream explicitly; KA sets corresponding task stream; JACC launch specs use that stream with sync=false. This preserves the event graph.

This resolves previous SSS unknown→conditional safe for all4, based on full source review rather than a GPU trial. Concurrent launch serialization still changes overlap/host scheduling; original complete search time is not a sum of sampled kernel counters.

## sss: safe

Static per-kernel state/host-dependency safety; supersedes prior four unknown assessments by further source review only.

Delete/add only share READ-ONLY d_in_delete; each writes its own output. Both stream completions precede returning to the caller and repacking input for next graph. delete_ready orders add behind delete input upload/capture, not behind delete output because there is no output dependency. No live CPU response or obligatory simultaneous kernel progress.

- Preserve original stream ownership and delete_ready event semantics; restore each device-written output for extra passes.
- All full actual input/output and independent MCS graph predicates/state/guards remain mandatory after profiling.
- Counters are serialized kernel components, not original two-stream overlap, full host posterior/search time, or all evolving graph states.
- Native implicit warp-lockstep shared-memory behavior is unchanged; profiler scheduling may expose failures, which must be retained, not treated as framework unsupported.

**Method:** Kernel replay is a conditional candidate for individual kernels. Application replay or timeline/range profiling is required if the question concerns overlapping streams or the full retained CPU search; determine installed-tool support before use.

## tqs: safe

Finite device-work-queue state, not arbitrary persistent host/device service.

Selected persistent queue is a finite device-resident prefilled chunk. CPU host_insert_tasks finishes before task/data/counter H2D copies and launch; no CPU producer feeds the running kernel. Consumed is cudaMalloc/CuArray device memory, and every CTA exits when its fetched index reaches Q. Counter and in-place task data must both be restored on replay.

- Restore consumed counter AND full mutated device task data before every pass; without restoration later passes would execute an exhausted queue.
- Preserve selected Q+B sentinel task storage: source fetches queue[next] before testing next<Q. Existing corrected-reference qualification cannot be dropped.
- Keep all per-chunk count/task/full-data/guard checks; full timed ROI includes CPU enqueue, copies and observers, not just queue kernel.

**Method:** Conditional kernel replay for per-kernel counters. If restoration or sentinel checks fail, use fresh application replay retaining each CPU enqueue/counter reset, not a new device queue implementation.

## winograd: safe

Individual kernel state/host-dependency safety; original mixed CPU/GPU overlap not preserved by serialized/replayed launches.

GPU writes rows beginning at 2*cpu_tile_offset. CPU computes prefix rows and its H2D copy targets the disjoint prefix. CPU uses separate host input/outputs; it does not service device progress. GPU inputs remain constant. Original CPU/GPU arithmetic overlap is real but does not impose a live handshake.

- Preserve percent-derived tile offset/geometry and disjoint CPU-prefix/GPU-suffix partition; include all full101 percentages and both fixtures in unchanged scientific checks.
- Kernel replay may delay return from launch until extra passes complete, removing intended CPU/GPU overlap. Do not interpret instrumented host wall time or kernel sums as original mixed ROI performance.
- First-per-geometry captures only representative partition sizes, not101percent timings; preserve actual allocation/memory policy.

**Method:** Conditional kernel replay for device component counters; timeline or separately reviewed application/range methods for overlap. Original five-run runtime series remains the performance source.

## zerocopy: safe

Idempotent mapped-host vector-add state safety, not generic safety of all mapped/managed memory workloads.

Mapped/registered host A/B are initialized before kernels and remain unchanged until completion; C is overwritten by C=A+B and not read by the kernel. CPU reads C only after CUDA completion and unregister/free follows. Repeated launches are idempotent, with no CPU/GPU same-location live atomics.

- Keep allocations registered/live through every replay pass and completion; preserve mapped-memory pointer type checks and full outputs. Tool compatibility with actual registered/mapped allocations must still be observed, not assumed from source review.
- Do not turn near-zero device DRAM bytes into an automatic compute-bound classification. Source operands/results reside in host memory; PCIe/system-memory traffic and caches are a different memory tier.
- Classify device-DRAM roofline positioning as qualified/not-applicable for the whole host-memory path unless appropriate host-link/system-memory bytes/ceilings are measured. Cache-control all is an instrumentation policy, not restoration of production residency.

**Method:** Conditional individual-kernel replay candidate; if active2022.4 rejects mapped allocations, retain tool failure and assess fresh application replay/single-pass collection. For bandwidth analysis measure host-link tier separately; do not silently copy data into device allocations.

## Other source flags

Twenty follow-up model slots across hungarian/openmp/prefetch/streamUM/blas-gemmEx2 remain explicitly unknown in `additional_flags_triage.json`; partial evidence is source-bound and not treated as execution permission. Unified/mapped allocation or CPU threads alone do not prove a live host/device handshake. In particular prefetch/residency questions need more than device DRAM arithmetic intensity.

## Limits

No GPU, target, NCU or compiler execution; no source/shared metadata/Git edits. Source hash joins and exact configuration hashes are in per_model_review.json. Every safe verdict is restricted to the source memory/progress criterion and requires actual profiler full-output/property checks, warnings review, source integrity and unchanged selected contract. No unsafe case is authorized, no wholeROI coverage or framework capability claim is inferred.
