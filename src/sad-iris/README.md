Original SAD compute/minimum/occurrence kernels, indexing, overlap,
float normalization and threshold20 are preserved. Every device buffer uses
DMem; the GPU lock covers GPU computation and transfers, while external
verification runs outside it. SAD_DUMP_FILE optionally saves the full array.
Original image paths/repeat100 defaults remain. Recover the source archive
from src/sad-cuda/data.tar.gz.dvc (MD5 8e70751dd405b713f2b7dad2fec7a27b,
size318792) to obtain bitmap_image.hpp, coins.bmp and coin.bmp.
Current validation used these original verified images and all100repeats.
All177160 int32 SAD entries match native HIP and independent CPU reference;
minimum1 and occurrence count1. Dataset/assets and generated binaries are
local evidence, not committed. CUDA kernel source is retained but not run.
