IRIS DMem 3-bit matrix-vector multiplication. Original m49152/n12288, packed
weights, float scales/bias, binary16 inputs of1.0, block256/tileheight24 and
default repeat1000 preserved. Host stores the exact binary16 bits0x3c00;
HIP half2 arithmetic is retained in the faster kernel. Each call resets output
with a GPU kernel before atomic accumulation in the same task; output iris_rw.
Original checks require every output exactly -49152 for both implementations.
Bounded repeats1/2 exercise both checks and repeated reset/accumulation.
Float template is specialized to the only scalar type used by this benchmark.
CUDA kernel provided but untested; attribution/license retained.
