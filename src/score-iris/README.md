IRIS DMem top-K score port. Seed123, threshold0.4, K10, priors4096, bins2048,
block256, defaultclasses1000/batch128/repeat100 and original sorted memcmp check
are preserved. Optional HECBENCH_SCORE_CLASSES/BATCH bound validation sizes.
HIP kernel uses explicit width32 shuffles to retain CUDA's 32-lane suffix scan;
native HIP's wave64 grouping produces the same integer suffix sums.
Indices start at zero through iris_rw DMem; count is fully written iris_w;
scores iris_r. Both outputs are flushed before original host reference/sorting.
HECBENCH_DUMP saves complete sorted indices/counts and their host references.
A matching test-only native adapter adds optional shape/dump controls. External
validation compares every integer, preventing cancellation in the legacy unsigned
memcmp checksum from hiding differences. Bounded cases: classes3/batch2/repeat1
and classes5/batch3/repeat2. Full default performance workload is untested.
