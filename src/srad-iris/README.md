Six CUDA SRAD stages, reduction order, argument widths, defaults and image
paths are preserved. All twelve buffers use dmem; sparse reductions use rw.
Reduction totals flush to host before ROI statistics and the final compressed
image flushes before image_out.pgm is written. HECBENCH_ARTIFACT optionally
writes every final float pixel for native comparison. The isolated native
adapter changes only its input path for an isolated working directory and
adds the same full buffer dump. Validation compares all pixels and the PGM.
