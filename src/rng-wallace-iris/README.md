# Wallace RNG with IRIS DMem

Original Hadamard device kernel, unsigned LCG, custom host RNG, 30 warmups,
CPU reference checks, repeat argument and16384block default retained.
IRIS initializes after all source RNG initialization. Pool DMem is read/write
across runs; output is full-write, corrections read-only; seed remains unsigned.
Optional compile HECBENCH_WALLACE_BLOCKS bounds only block count (default16384),
with identical isolated native constant override. HECBENCH_WALLACE_OUTPUT
optionally dumps the entire final output and persistent pool after timed runs.
Native adapter adds only dump and missing source kernel/reference headers.
