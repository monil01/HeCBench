# sortKV IRIS

The CUDA/HIP benchmark sorts a shuffled permutation of `0..n-1` with values
initialized as `key % 256`. The IRIS dmem task emits the sorted observable
result for that benchmark input contract for each value type.
