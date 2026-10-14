__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ float normal_from_uniform( uint32_t& rng) {
    float u1 = uint32_to_uniform(xorshift32(rng));
    float u2 = uint32_to_uniform(xorshift32(rng));

    // Box–Muller
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 6.28318530718f * u2;
    return r * cosf(theta);           // N(0,1)
}

template <typename T>
__device__ inline void uniform_fill_kernel(
    T* data,
    int64_t n,
    float low,
    float high,
    uint32_t seed
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
}

template <typename T>
__device__ inline void norm_fill_kernel(
    T* data,
    int n,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;
    data[idx] = (T)normal_from_uniform(rng);
}
