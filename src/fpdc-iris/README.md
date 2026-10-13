Status: DEFERRED, not a verified port. Complete compressed artifacts differ from
native HIP, and unchanged native HIP is itself non-reproducible at the byte level.
Native compiler identifies the inherited uninitialized control-code variable.
Baseline correction is required before meaningful artifact equivalence can pass.

IRIS DMem FPDC preserves original fixed double[i]=i input,64M-value default,
blocks/warps/repeat/optional dimensionality arguments,logical32-lane groups,
header/offset-table/chunk serialization and input.bin/output.bin names.
FPDC_VALUES permits bounded multiples of32. Compression kernel code is preserved,
including an inherited uninitialized control-code variable; artifact equivalence
must be established and is not inferred from compression ratio. No decompressor
is provided in original benchmark. Cache preference hint omitted (IRIS interface).
Compressed DMem buffer is flushed once before host chunk writes, avoiding offset
pointer aliases. Execute in isolated benchmark-state directories to preserve
existing data; kernel.hip must be available in working directory. CUDA unvalidated.
