#include <cuda_runtime.h>
#define THREADS 64
#define sos_width 6
template<typename T>
__device__ void sosfilt(
    const int n_signals,
    const int n_samples,
    const int n_sections,
    const int zi_width,
    const T *__restrict__ sos,
    const T *__restrict__ zi,
          T *__restrict__ x_in)
{
  __shared__ T smem[THREADS * (1 + 2 + sos_width)];
  T *s_out = reinterpret_cast<T *>( smem );
  T *s_zi = reinterpret_cast<T *>( &s_out[n_sections] ) ;
  T *s_sos = reinterpret_cast<T *>( &s_zi[n_sections * zi_width] ) ;

  const int tx = threadIdx.x;
  const int ty = blockIdx.x;

  // Reset shared memory
  s_out[tx] = 0;

  // Load zi
  for ( int i = 0; i < zi_width; i++ ) {
    s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
  }

  // Load SOS
#pragma unroll
  for ( int i = 0; i < sos_width; i++ ) {
    s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
  }

  const int load_size = n_sections - 1 ;
  const int unload_size = n_samples - load_size ;

  __syncthreads( );

  if ( ty < n_signals ) {
    T temp, x_n;

    // Loading phase
    for ( int n = 0; n < load_size; n++ ) {

      x_n = (tx == 0) ? x_in[ty * n_samples + n] : s_out[tx - 1];

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      __syncthreads( );

      s_out[tx] = temp;

      __syncthreads( );
    }

    // Processing phase
    for ( int n = load_size; n < n_samples; n++ ) {
      x_n = (tx == 0) ? x_in[ty * n_samples + n] : s_out[tx - 1];

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      __syncthreads( );

      if ( tx < load_size ) {
        s_out[tx] = temp;
      } else {
        x_in[ty * n_samples + ( n - load_size )] = temp;
      }

      __syncthreads( );
    }

    // Unloading phase
    for ( int n = 0; n < n_sections; n++ ) {
      // retire threads that are less than n
      if ( tx > n ) {
        x_n = s_out[tx - 1];

        // Use direct II transposed structure
        temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] =
          s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;
      }

      __syncthreads( );

      if ( tx > n ) {
        if ( tx < load_size ) {
          s_out[tx] = temp;
        } else {
          x_in[ty * n_samples + ( n + unload_size )] = temp;
        }
      }
      __syncthreads( );
    }
  }
}

extern "C" __global__ void sosfilt_float(const int a,const int b,const int c,const int d,const float* e,const float* f,float* g) { sosfilt<float>(a,b,c,d,e,f,g); }
extern "C" __global__ void sosfilt_double(const int a,const int b,const int c,const int d,const double* e,const double* f,double* g) { sosfilt<double>(a,b,c,d,e,f,g); }
