#define BLOCK_SIZE 256

extern "C" __global__ void damage_of_node(
    const int n,
    const int* __restrict__ nlist,
    const int* __restrict__ family,
    int* __restrict__ n_neigh,
    double* __restrict__ damage)
{
  __shared__ int local_cache[BLOCK_SIZE];

  const int local_id = threadIdx.x;
  const int local_size = blockDim.x;
  const int nid = blockIdx.x;
  const int global_id = nid * local_size + local_id;

  if (global_id < n) {
    local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0;
  } else {
    local_cache[local_id] = 0;
  }

  __syncthreads();

  for (int i = local_size / 2; i > 0; i /= 2) {
    if (local_id < i) {
      local_cache[local_id] += local_cache[local_id + i];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    const int neighbours = local_cache[0];
    n_neigh[nid] = neighbours;
    damage[nid] = 1.0 - static_cast<double>(neighbours) / static_cast<double>(family[nid]);
  }
}

extern "C" __global__ void damage_of_node_optimized(
    const int n,
    const int* __restrict__ nlist,
    const int* __restrict__ family,
    int* __restrict__ n_neigh,
    double* __restrict__ damage)
{
  __shared__ int local_cache[BLOCK_SIZE];

  const int local_id = threadIdx.x;
  const int local_size = blockDim.x;
  const int nid = blockIdx.x;
  const int global_id = nid * local_size + local_id;

  if (global_id < n) {
    local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0;
  } else {
    local_cache[local_id] = 0;
  }

  __syncthreads();

  for (int i = local_size / 2; i > 0; i /= 2) {
    if (local_id < i) {
      local_cache[local_id] += local_cache[local_id + i];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    const int sum = local_cache[0];
    n_neigh[nid] = sum;
    damage[nid] = 1.0 - static_cast<double>(sum) / static_cast<double>(family[nid]);
  }
}
