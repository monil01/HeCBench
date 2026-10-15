Status: verified with native HIP and an independent compressed-format decoder.
The inherited control-code bug is repaired identically in the shared CUDA/HIP
kernel and IRIS kernel: reset code per delta to0 (nonnegative) or8 (negative)
before merging the byte count. The old uninitialized code also retained stale
bits across subchunks. Source snapshots and old failed attempts are retained.

IRIS DMem FPDC preserves original fixed double[i]=i input,64M-value default,
blocks/warps/repeat/optional dimensionality arguments,logical32-lane groups,
header/offset-table/chunk serialization and input.bin/output.bin names.
FPDC_VALUES permits bounded multiples of32 in both native and IRIS builds;
its default remains64M. Validation uses4096 values with arguments2 2 1,
3 2 2 2,1 4 3 7,8 8 2 32,and a repeat of2 2 1. Every full compressed artifact
matches native HIP byte-for-byte, native reruns are deterministic, and an
independent CPU decoder reconstructs every input double bitpattern exactly.
The decoder separately parses control nibbles, byte counts, chunk headers and
dimensional predictors; compression ratios alone are not correctness evidence.
Both original-default builds succeed, but the64M/100repeat default was not run.
Cache preference hint omitted (documented IRIS interface/performance compromise).
Compressed DMem buffer is flushed once before host chunk writes, avoiding offset
pointer aliases. Execute in isolated benchmark-state directories to preserve
existing data; kernel.hip must be available in working directory. CUDA unvalidated.
