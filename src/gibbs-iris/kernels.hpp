/* 
   Generate a single Gamma distributed random variable by the Marsoglia 
   algorithm (George Marsaglia, Wai Wan Tsang; 2001).
 */
__device__ float rgamma(hiprandState *state, float a, float b)
{
  float d = a - 1.f / 3.f;
  float c = 1.f / sqrtf(9.f * d);
  bool flag = true;
  float V;

  while (flag) {
    // Generate a standard normal random variable
    float Z = hiprand_normal(state);
    if (Z > -1.f / c) {
      V = powf(1.f + c * Z, 3.f);
      float U = hiprand_uniform(state);
      flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
    }
  }
  return d * V / b;
}

/* 
   Initializes GPU random number generators 
 */
extern "C" __global__ void setup_kernel(hiprandState *state, int num_sample, unsigned int seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < num_sample)
    /* Each thread gets same seed, a different sequence number, no offset */
    hiprand_init(seed, id, 0, &state[id]);
}

/*
   Sample each theta from the appropriate gamma distribution
 */
extern "C" __global__ void sample_theta(
    hiprandState *__restrict__ state, 
    float *__restrict__ theta,
    const int *__restrict__ y,
    const float *__restrict__ n, 
    float a, float b, int num_sample)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < num_sample) {
    hiprandState *s = state + id;
    const float hyperA = a + y[id];
    const float hyperB = b + n[id];
    theta[id] = (hyperA < 1.f) ?
      rgamma(s, hyperA + 1.f, hyperB) * powf(hiprand_uniform(s), 1.f / hyperA) : 
      rgamma(s, hyperA, hyperB);
  }
}

/* 
   Sampling of a and b require the sum and product of all theta 
   values. This function performs parallel summations of 
   flat values and logs of theta for many blocks of length
   THREADS_PER_BLOCK_ADD. The CPU will then sum the block
   sums (see main method);    
 */
extern "C" __global__ void sum_blocks(const float *__restrict__ theta,
                                 float *__restrict__ flat_sums, 
                                 float *__restrict__ log_sums,
                                   int num_sample)
{
  __shared__ float flats[THREADS_PER_BLOCK_ADD];
  __shared__ float logs[THREADS_PER_BLOCK_ADD];

  int id = threadIdx.x + blockIdx.x * blockDim.x;

  flats[threadIdx.x] = (id < num_sample) ? theta[id] : 0.f;
  logs[threadIdx.x] = (id < num_sample) ? logf( theta[id] ) : 0.f;

  __syncthreads();

  int i = blockDim.x / 2;
  while(i != 0){
    if(threadIdx.x < i){
      flats[threadIdx.x] += flats[threadIdx.x + i];
      logs[threadIdx.x] += logs[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }

  if(threadIdx.x == 0){
    flat_sums[blockIdx.x] = flats[0];
    log_sums[blockIdx.x] = logs[0];
  }
}
