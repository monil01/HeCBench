Retains the original quantization kernel, repository-local BlockLoad/Store
headers, codebook, seed19937 and complete CPU reference. The source's
TH=BLOCK_SIZE=256 template becomes the exact same C-linkage specialization.
Three buffers use dmem. n explicitly narrows size_t to the kernel int as in
the native launch (unsupported oversized values are rejected). Defaults
16777216/100 remain. HECBENCH_ARTIFACT optionally dumps every byte result;
the isolated native adapter adds only the dump. Validation uses8193/3 to
exercise partial final blocks, complete CPU equality and native artifacts.
