IRIS dmem FRNA retaining original integer minimum-free-energy kernels, native
HIP reductions, thermodynamic parser and fold-save output/CLI/defaults. All
GPU calculations execute original device functions through HIP exported wrappers;
no CPU fill fallback or opaque vendor replacement. Original w5[-1] sentinel
allocation is retained with wrapper+1 offset. Partial writes use rw, output
matrices and vectors use explicit flush_out tasks. Init launch bounds are n+1,
initializing both terminal vector sentinels; the coaxial matrix starts at host
INF values. These repairs also apply to native CUDA/HIP. The serial reference
initializes w5-1 (the actual allocated base), fixing its one-element overrun.
No recurrence, inputs or tolerances were changed.

DATAPATH needs original RNAstructure data_tables. Validation recovers verified
original DVC assets in exclusive scratch, compares full native fold-save files
on original RD0260 sequence and DNA option, and attempts original HIV default
with bounded timeout. Untouched natives remain in expanded_baseline_original;
only HIP compiler
preinclude handles original header C-linkage ordering. CUDA backend retained,
NVIDIA unvalidated.

Validation qualification: full native HIP and IRIS saved artifacts match exactly
for RD0260 RNA/DNA and repeat runs. The original serial CPU recurrence disagrees
with both GPU implementations even after defined initialization (the original
source already notes CPU/GPU recurrence differences). This is native-GPU port
parity, not a claim that the original scientific GPU algorithm is CPU-validated.
