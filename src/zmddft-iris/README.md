Spiral ZMDDFT IRIS dmem port
==========================

Retains every source-generated 256x256x256 complex forward DFT arithmetic instruction, twiddle constant, shared-memory layout, indexing, three-stage geometry (16384 blocks,64 threads) and seeded mt19937 initialization. No vendor FFT/library replacement. All four arrays use iris::DMem, stages use r/rw and final output flush. Source default 100 repeats and checksum retained; optional HECBENCH_DUMP captures all 33554432 output doubles.

Validation uses existing CLI repeat=1 at the original full fixed shape. Isolated native adapter adds only optional binary capture, with native originals unchanged. Compare every output double (finite and atol=rtol=1e-12) in streaming chunks plus full artifact hashes; timings and HIP backend3/type256 MI100 proof recorded. Kernel.cu is supplied for CUDA but unvalidated locally.
