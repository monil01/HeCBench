IRIS DMem storeKVCache preserves CUDA logical 32-lane groups, mt19937 seed123,
8 layers, cache_size 262144, item sizes 2..64, batches1..512, repeat argument.
All buffers are conservatively iris_rw; explicit host updates/flushes.
Native HIP sibling uses physical 64-lane groups and larger default loop limits.
HECBENCH_BOUNDED_VALIDATION compile flag sets cache_size1024 and item sizes4..64;
validation-only HIP driver uses identical bounded inputs/batches1..512.
Without this flag all original CUDA defaults remain unchanged. Numerical check
is exact full-cache memcmp against CPU reference. CUDA source unvalidated.
