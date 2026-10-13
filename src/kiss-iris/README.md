# KISS IRIS
Retains repeat argument/default run1000, seed42, 2^28 uint64 values, 4096x256
launch and grid-stride RNG. Host storage uses std::vector and IRIS DMem rather
than native runtime allocation. KISS_VALIDATION_SIZE optionally bounds n;
KISS_VALIDATION_DUMP optionally writes every output uint64 in binary. Both are
unset by default. Validation compares entire dumped buffer with local HIP
snapshot under identical settings, including n greater than the grid stride.
