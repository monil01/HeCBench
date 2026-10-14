IRIS DMem port of the original64K modular inverse-transform/CRT kernel.

The65536 seeded input values,64-thread blocks,modular arithmetic and output
permutation are retained. `make run` retains100000 repetitions. Validation
uses one repetition on native HIP and IRIS_ARCHS=HIP, compares the complete
65536-value uint32 output exactly, and runs the CUDA-derived host reference.
HECBENCH_NTT_DUMP optionally writes raw uint32 outputs for artifact comparison;
an isolated native host fixture adds only that opt-in dump. Native siblings
remain untouched. An element mismatch now causes a failing exit status even
if the aggregate checksum happens to match.
