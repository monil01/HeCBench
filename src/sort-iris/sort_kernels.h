typedef unsigned int T;
typedef uint4 VECTYPE;

extern "C" __global__ void reduce(const T* in, T* isums, const size_t size,
                                  const unsigned int shift) {
  __shared__ T lmem[256];
  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  int region_size = (size / 4) / group_range * 4;
  int block_start = group * region_size;
  int block_stop = (group == group_range - 1) ? size : block_start + region_size;
  int tid = lid;
  int i = block_start + tid;

  int digit_counts[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0};

  while (i < block_stop) {
    digit_counts[(in[i] >> shift) & 0xFU]++;
    i += local_range;
  }

  for (int d = 0; d < 16; d++) {
    lmem[tid] = digit_counts[d];
    __syncthreads();

    for (unsigned int s = local_range / 2; s > 0; s >>= 1) {
      if (tid < s) {
        lmem[tid] += lmem[tid + s];
      }
      __syncthreads();
    }

    if (tid == 0) {
      isums[d * group_range + group] = lmem[0];
    }
  }
}

extern "C" __global__ void top_scan(T* isums, const size_t num_work_groups) {
  __shared__ T lmem[256 * 2];
  __shared__ T s_seed;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  if (lid == 0) s_seed = 0;
  __syncthreads();

  int last_thread =
      (lid < num_work_groups && (lid + 1) == num_work_groups) ? 1 : 0;

  for (int d = 0; d < 16; d++) {
    T val = 0;
    if (lid < num_work_groups) {
      val = isums[(num_work_groups * d) + lid];
    }

    int idx = lid;
    lmem[idx] = 0;
    idx += local_range;
    lmem[idx] = val;
    __syncthreads();
    for (int i = 1; i < local_range; i *= 2) {
      T t = lmem[idx - i];
      __syncthreads();
      lmem[idx] += t;
      __syncthreads();
    }
    T res = lmem[idx - 1];

    if (lid < num_work_groups) {
      isums[(num_work_groups * d) + lid] = res + s_seed;
    }
    __syncthreads();

    if (last_thread) {
      s_seed += res + val;
    }
    __syncthreads();
  }
}

extern "C" __global__ void bottom_scan(T* out, const T* in, const T* isums,
                                       const size_t size,
                                       const unsigned int shift) {
  __shared__ T lmem[256 * 2];
  __shared__ T l_scanned_seeds[16];
  __shared__ T l_block_counts[16];

  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  int histogram[16];
  int n4 = size / 4;
  int region_size = n4 / group_range;
  int block_start = group * region_size;
  int block_stop = (group == group_range - 1) ? n4 : block_start + region_size;
  int i = block_start + lid;
  int window = block_start;

  if (lid < 16) {
    l_block_counts[lid] = 0;
    l_scanned_seeds[lid] = isums[(lid * group_range) + group];
  }
  __syncthreads();

  while (window < block_stop) {
    for (int q = 0; q < 16; q++) histogram[q] = 0;
    VECTYPE val_4;
    VECTYPE key_4;

    if (i < block_stop) {
      val_4 = ((VECTYPE*)in)[i];

      key_4.x = (val_4.x >> shift) & 0xFU;
      key_4.y = (val_4.y >> shift) & 0xFU;
      key_4.z = (val_4.z >> shift) & 0xFU;
      key_4.w = (val_4.w >> shift) & 0xFU;

      histogram[key_4.x]++;
      histogram[key_4.y]++;
      histogram[key_4.z]++;
      histogram[key_4.w]++;
    }

    for (int digit = 0; digit < 16; digit++) {
      int idx = lid;
      lmem[idx] = 0;
      idx += local_range;
      lmem[idx] = histogram[digit];
      __syncthreads();
      for (int j = 1; j < local_range; j *= 2) {
        T t = lmem[idx - j];
        __syncthreads();
        lmem[idx] += t;
        __syncthreads();
      }
      histogram[digit] = lmem[idx - 1];
      __syncthreads();
    }

    if (i < block_stop) {
      int address;
      address = histogram[key_4.x] + l_scanned_seeds[key_4.x] +
                l_block_counts[key_4.x];
      out[address] = val_4.x;
      histogram[key_4.x]++;

      address = histogram[key_4.y] + l_scanned_seeds[key_4.y] +
                l_block_counts[key_4.y];
      out[address] = val_4.y;
      histogram[key_4.y]++;

      address = histogram[key_4.z] + l_scanned_seeds[key_4.z] +
                l_block_counts[key_4.z];
      out[address] = val_4.z;
      histogram[key_4.z]++;

      address = histogram[key_4.w] + l_scanned_seeds[key_4.w] +
                l_block_counts[key_4.w];
      out[address] = val_4.w;
      histogram[key_4.w]++;
    }

    __syncthreads();
    if (lid == local_range - 1) {
      for (int q = 0; q < 16; q++) {
        l_block_counts[q] += histogram[q];
      }
    }
    __syncthreads();

    window += local_range;
    i += local_range;
  }
}
