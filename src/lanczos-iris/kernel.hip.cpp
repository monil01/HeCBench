#include <hip/hip_runtime.h>
#define THREADS_PER_BLOCK 256
template <typename T>
__device__ inline void dot_product_kernel(const int N, const T *x, const T *y,
    T *z) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T result[THREADS_PER_BLOCK];

    if (index < N) {
        result[threadIdx.x] = x[index] * y[index];
    } else {
        result[threadIdx.x] = 0;
    }
    __syncthreads();

    int half = THREADS_PER_BLOCK / 2;
    while (half > 0) {
        if (threadIdx.x < half) {
            result[threadIdx.x] += result[threadIdx.x + half];
        }
        __syncthreads();
        half /= 2;
    }

    if (threadIdx.x == 0) {
        z[blockIdx.x] = result[0];
    }
}

/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
__device__ inline void multiply_inplace_kernel(const int N, T *x, const T k) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        x[index] = x[index] * k;
    }
}



/**
 * @brief   Cuda kernel function for vector saxpy in place(y += a * x).
 *
 * @param   N   The vector size.
 * @param   y   The output vector.
 * @param   x   The input vector.
 * @param   a   The value to multiply.
 */
template <typename T>
__device__ inline void saxpy_inplace_kernel(const int N, T *y, const T *x,
    const T a) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        y[index] += a * x[index];
    }
}


/**
 * @brief   Cuda kernel function for warp sparse matrix multiplication.
 *
 * @param   group_size  The number of threads used to calculate one row.
 * @param   rows        The row number of the matrix.
 * @param   begin_row   The row to begin from in this kernel launch.
 * @param   row_ptr     Row pointers in the CSR matrix.
 * @param   col_ind     Column indexes in the CSR matrix.
 * @param   values      Data values in the CSR matrix.
 * @param   x           The input vector x to multiply.
 * @param   y           The output vector y.
 */
template <typename T>
__device__ inline void warp_multiply_kernel(const int group_size, const int rows,
    const int begin_row, const int *row_ptr, const int *col_ind,
    const T *values, const T *x, T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int r = index / group_size + begin_row;
    int lane = index % group_size;
    __shared__ volatile T result[THREADS_PER_BLOCK];

    result[threadIdx.x] = 0;
    if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start + lane; i < end; i+= group_size) {
            result[threadIdx.x] += values[i] * x[col_ind[i]];
        }
        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
            if (lane < half) {
                result[threadIdx.x] += result[threadIdx.x + half];
            }
            half /= 2;
        }
        if (lane == 0) {
            y[r] = result[threadIdx.x];
        }
    }
}

extern "C" __global__ void dot_f32(int n,const float*x,const float*y,float*z){dot_product_kernel<float>(n,x,y,z);}
extern "C" __global__ void mul_f32(int n,float*x,float k){multiply_inplace_kernel<float>(n,x,k);}
extern "C" __global__ void saxpy_f32(int n,float*y,const float*x,float a){saxpy_inplace_kernel<float>(n,y,x,a);}
extern "C" __global__ void spmv_f32(int group,int rows,int begin,const int*rp,const int*ci,const float*a,const float*x,float*y){warp_multiply_kernel<float>(group,rows,begin,rp,ci,a,x,y);}
extern "C" __global__ void dot_f64(int n,const double*x,const double*y,double*z){dot_product_kernel<double>(n,x,y,z);}
extern "C" __global__ void mul_f64(int n,double*x,double k){multiply_inplace_kernel<double>(n,x,k);}
extern "C" __global__ void saxpy_f64(int n,double*y,const double*x,double a){saxpy_inplace_kernel<double>(n,y,x,a);}
extern "C" __global__ void spmv_f64(int group,int rows,int begin,const int*rp,const int*ci,const double*a,const double*x,double*y){warp_multiply_kernel<double>(group,rows,begin,rp,ci,a,x,y);}
