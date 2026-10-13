Retains original PCR/base/optimized, cyclic-reduction/base/optimized, local
sweep, global sweep, vec4 sweep and transpose kernels, CPU solver, inputs,
CLI and defaults. Every device buffer uses dmem; input host copies are
private backing vectors so transpose pointer swaps cannot overwrite CPU
inputs reused by the next solver. Final output flushes and fetches to x.
PCR/CR dynamic workspace becomes a static645-float maximum for the existing
system_size<=128 constraint; indexing/arithmetic remain unchanged.
BENCH_ITERATIONS defaults100; validation sets2 with source --num_systems=256.
HECBENCH_ARTIFACT dumps every GPU and CPU solution after each solver check;
isolated native copy adds the same dumps only. All default/lmem/vec4 modes
are checked, every output element finite and compared to native and CPU.
