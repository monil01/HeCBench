# GRRT with IRIS DMem

Preserves both physical tasks, adaptive ray integration, grids and output files.
Default image is 1024 by 1024. Optional EXTRA_CFLAGS=-DIMAGE_SIZE=48 produces a
bounded validation image, without changing the default. The isolated native
validation copy adds only an ifndef around IMAGE_SIZE for the identical option.
All three output channels in both artifacts are compared with native HIP.
Deterministic validation uses HIPFLAGS=-O3 and matching native -O3 without
fast math. The source/default native flags use fast math; the first attempt's
adaptive redshift differed at 4 pixels under these flags. Native task2 emits
NaNs at some pixels; validation requires identical NaN/Inf masks and compares
all finite channels. No NaN values are substituted.
Task outputs are read/write to preserve partial grid writes. Variables are
updated through DMem after host reset. AMD HIP backend is required.
