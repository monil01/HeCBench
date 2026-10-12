#include <hip/hip_runtime.h>

#define MAX_KERNEL_THREADS 256

typedef float vtype;

template<typename T>
__device__ T parallel_prefix_sum(const int n, const int* ind, const T* w)
{
  T sum = 0.0;
  T last;
  int mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x);
  for (int i = threadIdx.x; i < mn; i += blockDim.x) {
    bool valid = i < n;
    last = __shfl(sum, blockDim.x - 1, blockDim.x);
    sum = valid ? w[ind[i]] : 0.0;
    for (int j = 1; j < blockDim.x; j *= 2) {
      T v = __shfl_up(sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    sum += last;
  }
  last = __shfl(sum, blockDim.x - 1, blockDim.x);
  return last;
}

template<bool weighted, typename T>
__device__ void jaccard_row_sum_impl(const int n, const int* __restrict__ csrPtr,
                                       const int* __restrict__ csrInd,
                                       const T* __restrict__ w,
                                       T* __restrict__ work)
{
  for (int row = threadIdx.y + blockIdx.y * blockDim.y;
       row < n; row += gridDim.y * blockDim.y) {
    int start = csrPtr[row];
    int end = csrPtr[row + 1];
    int length = end - start;
    if (weighted) {
      T sum = parallel_prefix_sum(length, csrInd + start, w);
      if (threadIdx.x == 0) work[row] = sum;
    } else {
      work[row] = static_cast<T>(length);
    }
  }
}

template<bool weighted, typename T>
__device__ void jaccard_is_opt_impl(const int n, const int e,
                                      const int* __restrict__ csrPtr,
                                      const int* __restrict__ csrInd,
                                      const T* __restrict__ v,
                                      const T* __restrict__ work,
                                      T* __restrict__ weight_i,
                                      T* __restrict__ weight_s)
{
  for (int row = threadIdx.z + blockIdx.z * blockDim.z;
       row < n; row += gridDim.z * blockDim.z) {
    for (int j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y;
         j < csrPtr[row + 1]; j += gridDim.y * blockDim.y) {
      int col = csrInd[j];
      int Ni = csrPtr[row + 1] - csrPtr[row];
      int Nj = csrPtr[col + 1] - csrPtr[col];
      int ref = (Ni < Nj) ? row : col;
      int cur = (Ni < Nj) ? col : row;

      weight_s[j] = work[row] + work[col];

      if (threadIdx.x == 0) {
        T local_sum = 0;
        int i_ptr = csrPtr[ref];
        int j_ptr = csrPtr[cur];
        int ref_end = csrPtr[ref + 1];
        int cur_end = csrPtr[cur + 1];

        while (i_ptr < ref_end && j_ptr < cur_end) {
          int ref_col = csrInd[i_ptr];
          int cur_col = csrInd[j_ptr];
          if (ref_col == cur_col) {
            T ref_val = weighted ? v[ref_col] : static_cast<T>(1.0);
            local_sum += ref_val;
            i_ptr++;
            j_ptr++;
          } else if (ref_col < cur_col) {
            i_ptr++;
          } else {
            j_ptr++;
          }
        }
        if (local_sum != 0)
          atomicAdd(&weight_i[j], local_sum);
      }
    }
  }
}

extern "C" __global__ void jaccard_jw(const int e, const vtype* __restrict__ csrVal,
                                      const vtype gamma,
                                      const vtype* __restrict__ weight_i,
                                      const vtype* __restrict__ weight_s,
                                      vtype* __restrict__ weight_j)
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x;
       j < e; j += gridDim.x * blockDim.x) {
    vtype Wi = weight_i[j];
    vtype Ws = weight_s[j];
    weight_j[j] = (gamma * csrVal[j]) * (Wi / (Ws - Wi));
  }
}

template<bool weighted>
__device__ void jaccard_fill_impl(vtype* w, const int e, const vtype value)
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x;
       j < e; j += gridDim.x * blockDim.x) {
    w[j] = weighted ? static_cast<vtype>(j + 1) / e : value;
  }
}

extern "C" __global__ void jaccard_fill_true(vtype* w, const int e, const vtype value)
{
  jaccard_fill_impl<true>(w, e, value);
}

extern "C" __global__ void jaccard_fill_false(vtype* w, const int e, const vtype value)
{
  jaccard_fill_impl<false>(w, e, value);
}

extern "C" __global__ void jaccard_row_sum_true(const int n, const int* csrPtr,
                                                const int* csrInd, const vtype* w,
                                                vtype* work)
{
  jaccard_row_sum_impl<true, vtype>(n, csrPtr, csrInd, w, work);
}

extern "C" __global__ void jaccard_row_sum_false(const int n, const int* csrPtr,
                                                 const int* csrInd, const vtype* w,
                                                 vtype* work)
{
  jaccard_row_sum_impl<false, vtype>(n, csrPtr, csrInd, w, work);
}

extern "C" __global__ void jaccard_is_opt_true(const int n, const int e,
                                               const int* csrPtr, const int* csrInd,
                                               const vtype* v, const vtype* work,
                                               vtype* weight_i, vtype* weight_s)
{
  jaccard_is_opt_impl<true, vtype>(n, e, csrPtr, csrInd, v, work, weight_i, weight_s);
}

extern "C" __global__ void jaccard_is_opt_false(const int n, const int e,
                                                const int* csrPtr, const int* csrInd,
                                                const vtype* v, const vtype* work,
                                                vtype* weight_i, vtype* weight_s)
{
  jaccard_is_opt_impl<false, vtype>(n, e, csrPtr, csrInd, v, work, weight_i, weight_s);
}
