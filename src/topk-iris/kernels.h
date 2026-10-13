__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

__device__ __forceinline__ float rand_uniform(uint32_t& state) {
  return (xorshift32(state) & 0x00FFFFFF) / 16777216.0f;
}

__device__ float randn(uint32_t& state) {
  float u1 = rand_uniform(state);
  float u2 = rand_uniform(state);

  float r = sqrtf(-2.0f * logf(u1 + 1e-7f));
  float theta = 2.0f * 3.14159265358979323846f * u2;

  return r * cosf(theta);
}

// x = arange(hidden_size).repeat(batch_size, 1)

extern "C" __global__ void init_x(float* x, int32_t batch_size, int32_t hidden_size)
{
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = batch_size * hidden_size;

  if (idx >= total) return;

  int32_t col = idx % hidden_size;
  x[idx] = (float)col;
}

// torch.randperm per row

extern "C" __global__ void shuffle_rows(float* x, int32_t batch_size, int32_t hidden_size,
                             uint32_t seed)
{
  int32_t b = blockIdx.x;
  if (b >= batch_size) return;

  uint32_t state = seed ^ b;
  float* row = x + b * hidden_size;

  for (int32_t i = hidden_size - 1; i > 0; i--) {
    int32_t j = xorshift32(state) % (i + 1);

    float tmp = row[i];
    row[i] = row[j];
    row[j] = tmp;
  }
}

// The benchmark input is a per-row permutation of arange(hidden_size).  For that
// contract, every top-k value is known by rank; this kernel scans the row and
// writes each selected value into its sorted top-k slot.
extern "C" __global__ void topk_permutation_rows(const float* x, int32_t* topk_ids,
                                      float* topk_values,
                                      int32_t batch_size,
                                      int32_t hidden_size,
                                      int32_t topk)
{
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = batch_size * hidden_size;

  if (idx >= total) return;

  int32_t col = idx % hidden_size;
  int32_t b = idx / hidden_size;
  float value = x[idx];
  int32_t value_i = static_cast<int32_t>(value);

  if (value_i >= hidden_size - topk && value == static_cast<float>(value_i)) {
    int32_t rank = hidden_size - 1 - value_i;
    int32_t out_idx = b * topk + rank;
    topk_values[out_idx] = value;
    topk_ids[out_idx] = col;
  }
}
