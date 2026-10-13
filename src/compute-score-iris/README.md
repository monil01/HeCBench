Preserves CUDA setupData, srand(2), custom rand_desh stream, default document
count, CLI, manual vectorization and both reduction stages. ulong remains
unsigned long with an eight-byte assertion. All nine device buffers are
IRIS dmem. Output flushes after the final reduction and the original CPU
check examines every document. HECBENCH_ARTIFACT additionally dumps every
unsigned 64-bit score for exact native comparison. The isolated native copy
adds only this dump; validation uses existing -n=256 -p=3 options.
