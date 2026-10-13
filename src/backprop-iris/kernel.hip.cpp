#include <hip/hip_runtime.h>
#include "backprop.h"
extern "C" __global__ void kernel_layerforward(
  const float*__restrict__ input,
        float*__restrict__ input_weights,
        float*__restrict__ hidden_partial_sum,
  const int hid) 
{
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // gridDim.y << gridDim.x
  int by = blockIdx.y; 
  int tx = threadIdx.x; 
  int ty = threadIdx.y;

  int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

  int index_in = HEIGHT * by + ty + 1;

  if ( tx == 0 )
    input_node[ty] = input[index_in] ;
  __syncthreads();

  weight_matrix[ty * WIDTH + tx] =  input_weights[index];
  __syncthreads();

  weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];
  __syncthreads();

  for ( int i = 1 ; i <= HEIGHT ; i=i*2){
    int power_two = i; 

    if( ty % power_two == 0 )
      weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];

    __syncthreads();

  }

  input_weights[index] =  weight_matrix[ty * WIDTH + tx];

  __syncthreads();

  if ( tx == 0 ) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
  }
}
extern "C" __global__ void kernel_adjust_weights (
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{
  int by = blockIdx.y; 
  int tx = threadIdx.x; 
  int ty = threadIdx.y;

  int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  __syncthreads();

  if (ty == 0 && by ==0){
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}
