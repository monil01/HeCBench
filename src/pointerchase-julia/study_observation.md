# Pointer chasing: execution validation without a numerical output

rq_ids: RQ1, RQ2, RQ3, RQ4, RQ5
experiment_ids: E1, E2, E3, E4, E5
configuration_id: pointerchase_original_2MiB_64bit_pointers_stride16_1M_all_target_blocks_execution_only_v1

## Source contract and scientific interpretation

The original kernel follows a dependent chain of actual device pointers and asserts that its final pointer is non-null. It has no numerical store. The primary study is therefore execution-only and cannot contribute a four-way numerical-agreement result. This distinction is explicit in every structured row and per-model status.

The full source workload is retained: a two-megabyte allocation of262144 eight-byte pointers, successor(i)=(i+16) mod262144, one million dependent accesses per selected block, and all80 target block IDs on the measured V100. Every launch uses80 blocks with one thread each, of which only the matching block executes the chain. A block ID does not establish affinity to a particular physical SM. The cycle contains16384 nodes, spread across the original allocation at128-byte stride.

Historical CUDA.jl used genuine logical pointer chasing represented by UInt32 indices, reducing its physical array to one megabyte. Its source is preserved. Current Julia versions restore the original actual-address representation through an explicit LLVM load extension and CUDA device assertion. KA and JACC dispatch are genuine, but these device features make the implementations CUDA-specific hybrids. No unsupported framework feature or historical numerical failure is inferred from this representation change.

## Independent auxiliary checks

Each trial saves the actual allocation base, all262144 raw device pointers, and four raw final cursors in persistent ignored binary files. The entire graph is checked against its actual base and successor offsets. Separate untimed diagnostic kernels follow0,1,17 and1000000 pointers; required offsets are0,128,2176 and73728 bytes. The read-only Python audit checks every saved word in all20 captures and binds the files by SHA256.

Actual addresses naturally differ across processes. Raw addresses are retained, and no raw address equality, primary numerical artifact or invented sentinel result is claimed. These auxiliary graph and cursor checks remain separate from the original assertion-only primary contract. Primary rows have checked_values=0, null numerical errors and numerical_agreement_claim=false.

## Timing and effort

Bundle: `.porting-state/portable_julia/runs/20261018T101654575312Z_pointerchase/`.

All four models passed five checked trials; only the first timing is excluded from the mean. Mean trials2–5 in seconds: native CUDA11.754950444, CUDA.jl12.6853503175, KA12.6847206015 and JACC12.691308836. No outlier is removed. The common region sums all80 launch+device-completion intervals. Allocation, initialization, readback, auxiliary diagnostics and compilation/warmup are outside. Native per-node initialization and Julia bulk initialization differ outside timing.

These are execution-only timings. Dividing by the nominal80 million pointer accesses gives a host-inclusive rate, not isolated hardware load latency or evidence of cache behavior, actual executed instruction counts or SM affinity. No roofline or bandwidth claim follows from these clocks. Source mapping, loop code generation and dispatch are covariates, not established causes of the observed differences.

Each authored GPU implementation passed its first candidate and compiler/check invocation without a device algorithm repair. Shared graph, native adapter and diagnostics work belongs to the reference phase and is reused in KA-then-JACC order. Separate static reflection instrumentation is recorded separately. Prospective usage windows include repeated/cached context and coordination; overlapping outer and model windows must not be summed or interpreted as exclusive billed costs or intrinsic cold-start framework rankings. Historical creation cost remains unknown.

RQ1 records numerical agreement as inapplicable for the primary; RQ2 records phase-specific effort and taxonomy; RQ3 keeps execution timings outside numerical-performance recommendations; RQ4 links actual pointer representation and mapping to the workload; RQ5 does not recommend a universal abstraction from this microbenchmark.


Separate static inspection retains a loop and __assertfail in all four entry PTX captures. Native PTX uses generic ld.u64. Julia entry PTX calls an externally linked LLVM pointer-load helper; its body and final SASS were not captured, so identical load instructions or cache behavior are not claimed. A standalone reflection-wrapper include-path error was fixed after verification and is a shared T1 instrumentation event, not a device translation candidate or a failed primary trial. The measured sources and times are unchanged.
