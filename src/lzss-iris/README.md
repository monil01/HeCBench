# IRIS LZSS

Original block match, encoding, packing, and lossless decode kernels retained.
All buffers use IRIS DMem; the two host-dispatched hipCUB exclusive sums are
replaced by serial GPU prefix kernels. Their performance differs from hipCUB;
no CPU compression or GPU prefix fallback is used. All original input bytes
are checked bitwise after decompression. `HECBENCH_DUMP=path` writes complete
offset tables, consumed flag/payload bytes and reconstructed input for exact
native HIP artifact comparison. Input must be a nonempty uint32 binary array.

The unchanged default input name is available from the original upstream
https://github.com/hpdps-group/ICS23-GPULZ . Pass its path with `RUN_ARGS` if
it is not installed at the default sibling path.
