#include <hip/hip_runtime.h>
#include <limits.h>
#include <cmath>
#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

extern "C" __global__ void
kernel_find_index (
    const float*__restrict__ arrayX,
    const float*__restrict__ arrayY,
    const float*__restrict__ CDF,
    const float*__restrict__ u,
          float*__restrict__ xj,
          float*__restrict__ yj,
    const int Nparticles)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < Nparticles){
    int index = -1;
    int x;

    for(x = 0; x < Nparticles; x++){
      if(CDF[x] >= u[i]){
        index = x;
        break;
      }
    }
    if(index == -1){
      index = Nparticles-1;
    }

    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
}

extern "C" __global__ void
kernel_likelihood (
    float*__restrict__ arrayX, 
    float*__restrict__ arrayY, 
    const float*__restrict__ xj,
    const float*__restrict__ yj,
    int*__restrict__ ind,
    const int*__restrict__ objxy,
    float*__restrict__ likelihood,
    const unsigned char*__restrict__ I,
    float*__restrict__ weights,
    int*__restrict__ seed,
    float*__restrict__ partial_sums,
    const int Nparticles,
    const int countOnes,
    const int IszY,
    const int Nfr,
    const int k,
    const int max_size)
{
  __shared__ float weights_local[BLOCK_SIZE];

  int block_id = blockIdx.x; 
  int thread_id = threadIdx.x;
  int i = block_id * blockDim.x + thread_id;
  int y;
  int indX, indY;
  float u, v;

  if(i < Nparticles){
    arrayX[i] = xj[i]; 
    arrayY[i] = yj[i]; 

    weights[i] = 1.0f / ((float) (Nparticles)); 
    seed[i] = (A*seed[i] + C) % M;
    u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    v = fabsf(seed[i]/((float)M));
    arrayX[i] += 1.0f + 5.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));

    seed[i] = (A*seed[i] + C) % M;
    u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    v = fabsf(seed[i]/((float)M));
    arrayY[i] += -2.0f + 2.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));
  }

  __syncthreads();


  if(i < Nparticles)
  {
    for(y = 0; y < countOnes; y++){

      int iX = arrayX[i];
      int iY = arrayY[i];
      int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
      int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
      indX = rnd_iX + objxy[y*2 + 1];
      indY = rnd_iY + objxy[y*2];

      ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
      if(ind[i*countOnes + y] >= max_size)
        ind[i*countOnes + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for(int x = 0; x < countOnes; x++)
      likelihoodSum += ((I[ind[i*countOnes + x]] - 100) * (I[ind[i*countOnes + x]] - 100) -
          (I[ind[i*countOnes + x]] - 228) * (I[ind[i*countOnes + x]] - 228)) / 50.0f;
    likelihood[i] = likelihoodSum/countOnes-SCALE_FACTOR;

    weights[i] = weights[i] * expf(likelihood[i]); //Donnie Newell - added the missing exponential function call

  }

  weights_local[thread_id] = (i < Nparticles) ? weights[i] : 0.f;

  __syncthreads();

  for(unsigned int s=BLOCK_SIZE/2; s>0; s>>=1)
  {
    if(thread_id < s)
    {
      weights_local[thread_id] += weights_local[thread_id + s];
    }
    __syncthreads();
  }
  if(thread_id == 0)
  {
    partial_sums[block_id] = weights_local[0];
  }
}

extern "C" __global__ void
kernel_normalize_weights (
    float* __restrict__ weights,
    const float* __restrict__ partial_sums,
    float* __restrict__ CDF,
    float* __restrict__ u,
    int* __restrict__ seed,
    const int Nparticles )
{
  __shared__ float u1;
  __shared__ float sumWeights;
  int local_id = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(0 == local_id)
    sumWeights = partial_sums[0];
  __syncthreads();
  if(i < Nparticles) {
    weights[i] = weights[i]/sumWeights;
  }
  __syncthreads();
  if(i == 0) {
    CDF[0] = weights[0];
    for(int x = 1; x < Nparticles; x++){
      CDF[x] = weights[x] + CDF[x-1];
    }

    seed[i] = (A*seed[i] + C) % M;
    float p = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    float q = fabsf(seed[i]/((float)M));
    u[0] = (1.0f/((float)(Nparticles))) * 
      (sqrtf(-2.0f*log(p))*cosf(2.0f*PI*q));
    // do this to allow all threads in all blocks to use the same u1
  }
  __syncthreads();
  if(0 == local_id)
    u1 = u[0];

  __syncthreads();
  if(i < Nparticles)
  {
    u[i] = u1 + i/((float)(Nparticles));
  }
}

extern "C" __global__ void
kernel_sum (float* partial_sums, const int Nparticles)
{
  int x;
  float sum = 0;
  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (x = 0; x < num_blocks; x++) {
    sum += partial_sums[x];
  }
  partial_sums[0] = sum;
}
