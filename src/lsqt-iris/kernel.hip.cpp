#include <hip/hip_runtime.h>
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#include <math.h>
#ifdef USE_SP
using lsqt_real=float;
#else
using lsqt_real=double;
#endif
#define BLOCK_SIZE 256
extern "C" __global__ void gpu_set_zero(int number_of_elements,
  lsqt_real* __restrict__ g_state_real,
  lsqt_real* __restrict__ g_state_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_elements) {
    g_state_real[n] = 0;
    g_state_imag[n] = 0;
  }
}

extern "C" __global__ void gpu_copy_state(
  const int N,
  const lsqt_real* __restrict__ in_real,
  const lsqt_real* __restrict__ in_imag,
        lsqt_real* __restrict__ out_real,
        lsqt_real* __restrict__ out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    out_real[n] = in_real[n];
    out_imag[n] = in_imag[n];
  }
}

extern "C" __global__ void gpu_add_state(
  const int n,
  const lsqt_real*__restrict__ in_real,
  const lsqt_real*__restrict__ in_imag,
        lsqt_real*__restrict__ out_real,
        lsqt_real*__restrict__ out_imag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_real[i] += in_real[i];
    out_imag[i] += in_imag[i];
  }
}

extern "C" __global__ void gpu_apply_sz(
  const int n,
  const lsqt_real* __restrict__ in_real,
  const lsqt_real* __restrict__ in_imag,
        lsqt_real* __restrict__ out_real,
        lsqt_real* __restrict__ out_imag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (i % 2 == 0) {
      out_real[i] = in_real[i];
      out_imag[i] = in_imag[i];
    } else {
      out_real[i] = -in_real[i];
      out_imag[i] = -in_imag[i];
    }
  }
}

__device__ void warp_reduce(volatile lsqt_real* s, int t)
{
  lsqt_real v = s[t];
  v += s[t + 32];   __syncwarp();
  s[t] = v;         __syncwarp();
  v += s[t + 16];   __syncwarp();
  s[t] = v;         __syncwarp();
  v += s[t + 8];    __syncwarp();
  s[t] = v;         __syncwarp();
  v += s[t + 4];    __syncwarp();
  s[t] = v;         __syncwarp();
  v += s[t + 2];    __syncwarp();
  s[t] = v;         __syncwarp();
  v += s[t + 1];    __syncwarp();
  s[t] = v;
}

extern "C" __global__ void gpu_find_inner_product_1(
  const int number_of_atoms,
  const lsqt_real* __restrict__ g_final_state_real,
  const lsqt_real* __restrict__ g_final_state_imag,
  const lsqt_real* __restrict__ g_random_state_real,
  const lsqt_real* __restrict__ g_random_state_imag,
        lsqt_real* __restrict__ g_inner_product_real,
        lsqt_real* __restrict__ g_inner_product_imag,
  const int g_offset)
{
  int tid = threadIdx.x;
  int n = blockIdx.x * blockDim.x + tid;
  int m;
  lsqt_real a, b, c, d;
  __shared__ lsqt_real s_data_real[BLOCK_SIZE];
  __shared__ lsqt_real s_data_imag[BLOCK_SIZE];
  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;

  if (n < number_of_atoms) {
    a = g_final_state_real[n];
    b = g_final_state_imag[n];
    c = g_random_state_real[n];
    d = g_random_state_imag[n];
    s_data_real[tid] = (a * c + b * d);
    s_data_imag[tid] = (b * c - a * d);
  }
  __syncthreads();

/*
  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
*/

  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_real[blockIdx.x + g_offset] = s_data_real[0];
    g_inner_product_imag[blockIdx.x + g_offset] = s_data_imag[0];
  }
}

extern "C" __global__ void gpu_find_inner_product_2(
  const int number_of_atoms,
  const lsqt_real* __restrict__ g_inner_product_1_real,
  const lsqt_real* __restrict__ g_inner_product_1_imag,
        lsqt_real* __restrict__ g_inner_product_2_real,
        lsqt_real* __restrict__ g_inner_product_2_imag)
{
  //<<<para.number_of_energy_points, BLOCK_SIZE)>>>
  int tid = threadIdx.x;
  int patch, n, m;

  __shared__ lsqt_real s_data_real[BLOCK_SIZE];
  __shared__ lsqt_real s_data_imag[BLOCK_SIZE];
  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;
  int number_of_blocks = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  int number_of_patches = (number_of_blocks - 1) / BLOCK_SIZE + 1;

  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * BLOCK_SIZE;
    if (n < number_of_blocks) {
      m = blockIdx.x * number_of_blocks + n;
      s_data_real[tid] += g_inner_product_1_real[m];
      s_data_imag[tid] += g_inner_product_1_imag[m];
    }
  }
  __syncthreads();

/*
  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
*/

  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_2_real[blockIdx.x] = s_data_real[0];
    g_inner_product_2_imag[blockIdx.x] = s_data_imag[0];
  }
}

extern "C" __global__ void gpu_apply_hamiltonian(
  const int number_of_atoms,
  const lsqt_real energy_max,
  const  int* __restrict__ g_neighbor_number,
  const  int* __restrict__ g_neighbor_list,
  const lsqt_real* __restrict__ g_potential,
  const lsqt_real* __restrict__ g_hopping_real,
  const lsqt_real* __restrict__ g_hopping_imag,
  const lsqt_real* __restrict__ g_state_in_real,
  const lsqt_real* __restrict__ g_state_in_imag,
        lsqt_real* __restrict__ g_state_out_real,
        lsqt_real* __restrict__ g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = g_potential[n] * g_state_in_real[n]; // on-site
    lsqt_real temp_imag = g_potential[n] * g_state_in_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_in_real[index_2];
      lsqt_real d = g_state_in_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    g_state_out_real[n] = temp_real;
    g_state_out_imag[n] = temp_imag;
  }
}

extern "C" __global__ void gpu_apply_commutator(
  int number_of_atoms,
  lsqt_real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  lsqt_real* g_hopping_real,
  lsqt_real* g_hopping_imag,
  lsqt_real* g_xx,
  lsqt_real* g_state_in_real,
  lsqt_real* g_state_in_imag,
  lsqt_real* g_state_out_real,
  lsqt_real* g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = 0.0;
    lsqt_real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_in_real[index_2];
      lsqt_real d = g_state_in_imag[index_2];
      lsqt_real xx = g_xx[index_1];
      temp_real -= (a * c - b * d) * xx;
      temp_imag -= (a * d + b * c) * xx;
    }
    g_state_out_real[n] = temp_real / energy_max; // scale
    g_state_out_imag[n] = temp_imag / energy_max; // scale
  }
}

extern "C" __global__ void gpu_apply_current(
  const int number_of_atoms,
  const  int* __restrict__ g_neighbor_number,
  const  int* __restrict__ g_neighbor_list,
  const lsqt_real* __restrict__ g_hopping_real,
  const lsqt_real* __restrict__ g_hopping_imag,
  const lsqt_real* __restrict__ g_xx,
  const lsqt_real* __restrict__ g_state_in_real,
  const lsqt_real* __restrict__ g_state_in_imag,
        lsqt_real* __restrict__ g_state_out_real,
        lsqt_real* __restrict__ g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = 0.0;
    lsqt_real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_in_real[index_2];
      lsqt_real d = g_state_in_imag[index_2];
      temp_real += (a * c - b * d) * g_xx[index_1];
      temp_imag += (a * d + b * c) * g_xx[index_1];
    }
    g_state_out_real[n] = +temp_imag;
    g_state_out_imag[n] = -temp_real;
  }
}

extern "C" __global__ void gpu_chebyshev_01(
  const int number_of_atoms,
  const lsqt_real* __restrict__ g_state_0_real,
  const lsqt_real* __restrict__ g_state_0_imag,
  const lsqt_real* __restrict__ g_state_1_real,
  const lsqt_real* __restrict__ g_state_1_imag,
        lsqt_real* __restrict__ g_state_real,
        lsqt_real* __restrict__ g_state_imag,
  const lsqt_real b0,
  const lsqt_real b1,
  const int direction)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real bessel_0 = b0;
    lsqt_real bessel_1 = b1 * direction;
    g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
    g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
  }
}

extern "C" __global__ void gpu_chebyshev_2(
  const int number_of_atoms,
  const lsqt_real energy_max,
  const  int* __restrict__ g_neighbor_number,
  const  int* __restrict__ g_neighbor_list,
  const lsqt_real* __restrict__ g_potential,
  const lsqt_real* __restrict__ g_hopping_real,
  const lsqt_real* __restrict__ g_hopping_imag,
  const lsqt_real* __restrict__ g_state_0_real,
  const lsqt_real* __restrict__ g_state_0_imag,
  const lsqt_real* __restrict__ g_state_1_real,
  const lsqt_real* __restrict__ g_state_1_imag,
        lsqt_real* __restrict__ g_state_2_real,
        lsqt_real* __restrict__ g_state_2_imag,
        lsqt_real* __restrict__ g_state_real,
        lsqt_real* __restrict__ g_state_imag,
  const lsqt_real bessel_m,
  const int label)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    lsqt_real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_1_real[index_2];
      lsqt_real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    switch (label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_real;
        g_state_imag[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_real;
        g_state_imag[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_imag;
        g_state_imag[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_imag;
        g_state_imag[n] += bessel_m * temp_real;
        break;
      }
    }
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}

extern "C" __global__ void gpu_chebyshev_1x(
  const int number_of_atoms,
  const lsqt_real* __restrict__ g_state_1x_real,
  const lsqt_real* __restrict__ g_state_1x_imag,
        lsqt_real* __restrict__ g_state_real,
        lsqt_real* __restrict__ g_state_imag,
  const lsqt_real g_bessel_1)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real b1 = g_bessel_1;
    g_state_real[n] = +b1 * g_state_1x_imag[n];
    g_state_imag[n] = -b1 * g_state_1x_real[n];
  }
}

extern "C" __global__ void gpu_chebyshev_2x(
  const int number_of_atoms,
  const lsqt_real energy_max,
  const  int* __restrict__ g_neighbor_number,
  const  int* __restrict__ g_neighbor_list,
  const lsqt_real* __restrict__ g_potential,
  const lsqt_real* __restrict__ g_hopping_real,
  const lsqt_real* __restrict__ g_hopping_imag,
  const lsqt_real* __restrict__ g_xx,
  const lsqt_real* __restrict__ g_state_0_real,
  const lsqt_real* __restrict__ g_state_0_imag,
  const lsqt_real* __restrict__ g_state_0x_real,
  const lsqt_real* __restrict__ g_state_0x_imag,
  const lsqt_real* __restrict__ g_state_1_real,
  const lsqt_real* __restrict__ g_state_1_imag,
  const lsqt_real* __restrict__ g_state_1x_real,
  const lsqt_real* __restrict__ g_state_1x_imag,
        lsqt_real* __restrict__ g_state_2_real,
        lsqt_real* __restrict__ g_state_2_imag,
        lsqt_real* __restrict__ g_state_2x_real,
        lsqt_real* __restrict__ g_state_2x_imag,
        lsqt_real* __restrict__ g_state_real,
        lsqt_real* __restrict__ g_state_imag,
  const lsqt_real g_bessel_m,
  const int g_label)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = g_potential[n] * g_state_1_real[n];    // on-site
    lsqt_real temp_imag = g_potential[n] * g_state_1_imag[n];    // on-site
    lsqt_real temp_x_real = g_potential[n] * g_state_1x_real[n]; // on-site
    lsqt_real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];

      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_1_real[index_2];
      lsqt_real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping

      lsqt_real cx = g_state_1x_real[index_2];
      lsqt_real dx = g_state_1x_imag[index_2];
      temp_x_real += a * cx - b * dx; // hopping
      temp_x_imag += a * dx + b * cx; // hopping

      lsqt_real xx = g_xx[index_1];
      temp_x_real -= (a * c - b * d) * xx; // hopping
      temp_x_imag -= (a * d + b * c) * xx; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;

    temp_x_real /= energy_max; // scale
    temp_x_imag /= energy_max; // scale
    temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
    temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
    g_state_2x_real[n] = temp_x_real;
    g_state_2x_imag[n] = temp_x_imag;

    lsqt_real bessel_m = g_bessel_m;
    switch (g_label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_x_real;
        g_state_imag[n] += bessel_m * temp_x_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_x_real;
        g_state_imag[n] -= bessel_m * temp_x_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_x_imag;
        g_state_imag[n] -= bessel_m * temp_x_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_x_imag;
        g_state_imag[n] += bessel_m * temp_x_real;
        break;
      }
    }
  }
}

extern "C" __global__ void gpu_kernel_polynomial(
  const int number_of_atoms,
  const lsqt_real energy_max,
  const  int* __restrict__ g_neighbor_number,
  const  int* __restrict__ g_neighbor_list,
  const lsqt_real* __restrict__ g_potential,
  const lsqt_real* __restrict__ g_hopping_real,
  const lsqt_real* __restrict__ g_hopping_imag,
  const lsqt_real* __restrict__ g_state_0_real,
  const lsqt_real* __restrict__ g_state_0_imag,
  const lsqt_real* __restrict__ g_state_1_real,
  const lsqt_real* __restrict__ g_state_1_imag,
        lsqt_real* __restrict__ g_state_2_real,
        lsqt_real* __restrict__ g_state_2_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    lsqt_real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    lsqt_real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      lsqt_real a = g_hopping_real[index_1];
      lsqt_real b = g_hopping_imag[index_1];
      lsqt_real c = g_state_1_real[index_2];
      lsqt_real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
