DXTC2 IRIS dmem port
===================

Retains the original eigenvector/permutation/least-squares endpoint search, thread-block reduction and DXT1 packing, cooperative groups, constant tables, 64 threads/block, full DDS writer, decompressed reference RMS check and tolerance0.02. Three dmem buffers use update for host input, rw for partial batched compression, flush for complete output; blockOffset scalar is int. Source defaults run Lena and teapot images with100 repeats; RUN_ARGS optionally overrides.

Validation copies both original image/reference pairs into isolated native/IRIS directories because the source writes output DDS beside the input. Existing repeat1 CLI used; every compressed block and DDS header compared byte-exact to native, with original full decompressed reference verdict required to PASS. Unchanged native source baseline, shared shrUtils snapshot included. No library/algorithm replacement.
