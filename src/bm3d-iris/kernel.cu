#include <cuda_runtime.h>
#define bm_shfl_down(value,offset) __shfl_down_sync(0xFFFFFFFF,value,offset,32)

#include "params.hpp"
#include "indices.hpp"


// Nearest lower power of 2
__device__ __inline__ uint flp2 (uint x)
{
  return (0x80000000u >> __clz(x));
}

//Computes the squared difference between two numbers
template<typename T>
__device__ __inline__ T L2p2(const T i1, const T i2)
{
  T diff = i1 - i2;
  return diff*diff;
}

/*
   Adds new patch to patch stack (only N most similar are kept)
Note: Stack is just an array, not FIFO
 */
__device__
void add_to_matched_image(
    uint *stack,         //IN/OUT: Stack of N patches matched to current reference patch
    uchar *num_patches_in_stack,//IN/OUT: Number of patches in stack
    const uint value,       //IN: [..DIFF(ushort)..|..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
    const Params & params    //IN: Denoising parameters
    )
{
  //stack[*num_patches_in_stack-1] is most similar (lowest number)
  int k;

  uchar num = (*num_patches_in_stack);
  if (num < params.N) //add new value
  {
    k = num++;
    while(k > 0 && value > stack[k-1])
    {
      stack[k] = stack[k-1];
      --k;
    }

    stack[k] = value;
    *num_patches_in_stack = num;
  }
  else if (value >= stack[0])
    return;
  else //delete highest value and add new
  {
    k = 1;
    while (k < params.N && value < stack[k])
    {
      stack[k-1] = stack[k];
      k++;
    }
    stack[k-1] = value;
  }
}

/*
   Block-matching algorithm
   For each processed reference patch it finds maximaly N similar patches that pass the distance threshold and stores them to the g_stacks array.
   It also returns the number of them for each reference patch in g_num_patches_in_stack.
   Used denoising parameters: n,k,N,T,p
Division: Kernel handles gridDim.y lines starting with the line passed in argument. Each block handles warpSize reference patches in line.
Each thread process one reference patch. All the warps of a block process the same reference patches.
 */
extern "C" __global__ void block_matching(
    const  uchar* __restrict image, //IN: Original image
    ushort* __restrict g_stacks,         //OUT: For each reference patch contains addresses of similar patches (patch is adressed by top left corner) [..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
    uint* __restrict g_num_patches_in_stack,  //OUT: For each reference patch contains number of similar patches
    const uint2 image_dim,      //IN: Image dimensions
    const uint2 stacks_dim,      //IN: Size of area, where reference patches could be located
    const Params params,      //IN: Denoising parameters
    const uint2 start_point)    //IN: Address of the top-left reference patch of a batch
{
  //One block is processing warpSize patches (because each warp is computing distance of same warpSize patches from different displaced patches)
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  int num_warps = blockDim.x/warpSize;

  //p_block denotes reference rectangle on which current cuda block is computing
  uint p_rectangle_width = ((warpSize-1) * params.p) + params.k;
  uint p_rectangle_start = start_point.x + blockIdx.x * warpSize * params.p;

  //Shared arrays
  __shared__ uint s_data[38504 / sizeof(uint)];
  uint *s_diff = s_data; //SIZE: p_rectangle_width*num_warps
  uint *s_stacks = &s_data[p_rectangle_width*num_warps]; //SIZE: params.N*num_warps*warpSize
  uchar *s_patches_in_stack = (uchar*)&s_data[num_warps*(p_rectangle_width + params.N*warpSize)]; //SIZE: num_warps*warpSize
  uchar *s_image_p = (uchar*)&s_patches_in_stack[num_warps*warpSize]; //SIZE: p_rectangle_width*params.k

  s_diff += idx2(0, wid, p_rectangle_width);

  //Initialize s_patches_in_stack to zero
  s_patches_in_stack[ idx2(tid, wid, warpSize) ] = 0;

  int2 p; //Address of reference patch
  int2 q; //Address of patch against which the difference is computed

  p.x = p_rectangle_start + (tid*params.p);
  p.y = start_point.y + (blockIdx.y*params.p);

  //Ensure, that the bottom most patches will be taken as reference patches regardless the p parameter.
  if (p.y >= stacks_dim.y && p.y < stacks_dim.y + params.p - 1)
    p.y = stacks_dim.y - 1;
  else if (p.y >= stacks_dim.y) return;

  //Ensure, that the right most patches will be taken as reference patches regardless the p parameter.
  uint inner_p_x = tid*params.p;
  if (p.x >= stacks_dim.x && p.x < stacks_dim.x + params.p - 1)
  {
    inner_p_x -= (p.x - (stacks_dim.x - 1));
    p.x = stacks_dim.x - 1;
  }

  //Load reference patches needed by actual block to shared memory
  for(int i = threadIdx.x; i < p_rectangle_width*params.k; i+=blockDim.x)
  {
    int sx = i % p_rectangle_width;
    int sy = i / p_rectangle_width;
    if (p_rectangle_start+sx >= image_dim.x) continue;
    s_image_p[i] = image[idx2(p_rectangle_start+sx,p.y+sy,image_dim.x)];
  }

  __syncthreads();

  //scale difference so that it can fit ushort
  uint shift = (__clz(params.Tn) < 16u) ? 16u - (uint)__clz(params.Tn) : 0;


  //Ensure that displaced patch coordinates (q) will be positive
  int2 from;
  from.y = (p.y - (int)params.n < 0) ? -p.y : -(int)params.n;
  from.x = (((int)p_rectangle_start) - (int)params.n < 0) ? -((int)p_rectangle_start) : -(int)params.n;
  from.x += wid;

  //For each displacement (x,y) in n neighbourhood
  for(int y = from.y; y <= (int)params.n; ++y)
  {
    q.y = p.y + y;
    if (q.y >= stacks_dim.y) break;

    for(int x = from.x; x <= (int)params.n; x += num_warps)
    {
      //Reference patch is always the most similar to itself (there is no need to copute it)
      if (x == 0 && y == 0) continue;

      //Each warp is computing the same patch with slightly different displacement.
      //Compute distance of reference patch p from current patch q which is dispaced by (x+tid,y)

      //q_block denotes displaced rectangle which is processed by the current warp
      uint q_rectangle_start = p_rectangle_start + x;
      q.x = q_rectangle_start + inner_p_x;

      //Compute distance for each column of reference patch
      for(uint i = tid; i < p_rectangle_width && p_rectangle_start+i < image_dim.x &&
                        q_rectangle_start+i < image_dim.x; i+=warpSize)
      {
        uint dist = 0;
        for(uint iy = 0; iy < params.k; ++iy)
        {
          dist += L2p2((int)s_image_p[ idx2(i, iy, p_rectangle_width) ],
                       (int)image[ idx2(q_rectangle_start+i, q.y+iy, image_dim.x) ]);
        }
        s_diff[i] = dist;
      }

      if (p.x >= stacks_dim.x || q.x >= stacks_dim.x) continue;

      //Sum column distances to obtain patch distance
      uint diff = 0;
      for (uint i = 0; i < params.k; ++i)
        diff += s_diff[inner_p_x + i];

      //Distance threshold
      if(diff < params.Tn)
      {
        uint loc_y = (uint)((q.y - p.y) & 0xFF); //relative location y (-127 to 127)
        uint loc_x = (uint)((q.x - p.x) & 0xFF); //relative location x (-127 to 127)
        diff >>= shift;
        diff <<= 16u; // [..DIFF(ushort)..|..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
        diff |= (loc_y << 8u);
        diff |= loc_x;

        //Add current patch to s_stacks
        add_to_matched_image(
            &s_stacks[ params.N * idx2(tid, wid, warpSize) ],
            &s_patches_in_stack[ idx2(tid, wid, warpSize) ],
            diff,
            params
            );
      }
    }
  }

  __syncthreads();

  uint batch_size = gridDim.x*warpSize;
  uint block_address_x = blockIdx.x*warpSize+tid;

  if (wid > 0) return;
  //Select N most similar patches for each reference patch from stacks in shared memory and save them to global memory
  //Each thread represents one reference patch
  //Each thread will find N most similar blocks in num_warps stacks (which were computed by different warps) and save them into global memory
  //In shared memory the most similar patch is at the end, in global memory the order does not matter
  //DEV: performance impact cca 8%
  if (p.x >= stacks_dim.x) return;

  int j;
  for (j = 0; j < params.N; ++j)
  {
    uint count = 0;
    uint minIdx = 0;
    uint minVal = 0xFFFFFFFF; //INF

    //Finds patch with minimal value of remaining
    for (int i = minIdx; i < num_warps; ++i)
    {
      count = (uint)s_patches_in_stack[ idx2(tid, i, warpSize) ];
      if (count == 0) continue;

      uint newMinVal = s_stacks[ idx3(count-1,tid,i,params.N,warpSize) ];
      if (newMinVal < minVal)
      {
        minVal = newMinVal;
        minIdx = i;
      }
    }
    if (minVal == 0xFFFFFFFF) break; //All stacks are empty

    //Remove patch from shared stack
    s_patches_in_stack[ idx2(tid, minIdx, warpSize) ]--;

    //Adds patch to stack in global memory
    g_stacks[idx3(j, block_address_x, blockIdx.y, params.N, batch_size)] = (ushort)(minVal & 0xFFFF);
  }
  //Save to the global memory the number of similar patches rounded to the nearest lower power of two
  g_num_patches_in_stack[ idx2(block_address_x ,blockIdx.y, batch_size) ] = flp2((uint)j+1)-1;
}



#include <float.h>
#include <stdio.h>


#include "indices.hpp"
#include "params.hpp"


// Kernels used for collaborative filtering and aggregation

//Sum the passed values in a warp to the first thread of this warp.
template<typename T>
__device__ inline T warpReduceSum(T val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += bm_shfl_down(val,offset);
  return val;
}


//Sum the passed values in a block to the first thread of a block.
template<typename T>
__inline__ __device__ float blockReduceSum(T* shared, T val, int tid, int tcount)
{
  int lane = tid % warpSize;
  int wid = tid / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (tid < tcount / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

//Returns absolute value of the passed real number raised to the power of two
__device__ __forceinline__
float abspow2(float & a)
{
  return a * a;
}


//Integer logarithm base 2.
template <typename IntType>
__device__ __inline__ uint ilog2(IntType n)
{
  uint l;
  for (l = 0; n; n >>= 1, ++l);
  return l;
}


//Orthogonal transformation.
template <typename T>
__device__ __inline__ void rotate(T& a, T& b)
{
  T tmp;
  tmp = a;
  a = tmp + b;
  b = tmp - b;
}


//Fast Walsh-Hadamard transform.
template <typename T>
__device__ __inline__ void fwht(T *data, uint n)
{
  unsigned l2 = ilog2(n) - 1;
  for ( uint i = 0; i < l2; ++i )
  {
    for (uint j = 0; j < n; j += (1 << (i + 1)))
    for (uint k = 0; k < (uint)(1 << i); ++k)
      rotate(data[j + k], data[j + k + (uint)(1 << i)]);
  }
}

//Based on blockIdx it computes the addresses to the arrays in global memory
__device__ inline void get_block_addresses(
  const uint2 & start_point,    //IN: first reference patch of a batch
  const uint & patch_stack_size,  //IN: maximal size of a 3D group
  const uint2 & stacks_dim,    //IN: Size of area, where reference patches could be located
  const Params & params,      //IN: Denoising parameters
  uint2 & outer_address,      //OUT: Coordinetes of reference patch in the image
  uint & start_idx)        //OUT: Address of a first element of the 3D group in stacks array
{
  //One block handles one patch_stack, data are in array one after one.
  start_idx = patch_stack_size * idx2(blockIdx.x,blockIdx.y,gridDim.x);

  outer_address.x = start_point.x + (blockIdx.x * params.p);
  outer_address.y = start_point.y + (blockIdx.y * params.p);

  //Ensure, that the bottom most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.y >= stacks_dim.y && outer_address.y < stacks_dim.y + params.p - 1)
    outer_address.y = stacks_dim.y - 1;
  //Ensure, that the right most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.x >= stacks_dim.x && outer_address.x < stacks_dim.x + params.p - 1)
    outer_address.x = stacks_dim.x - 1;
}

/*
Gather patches form image based on matching stored in 3D array stacks
Used parameters: p,k,N
Division: One block handles one patch_stack, threads match to the pixels of a patch
*/
extern "C" __global__ void get_block(
    const uint2 start_point,                       //IN: first reference patch of a batch
    const uchar* __restrict image,                 //IN: image
    const ushort* __restrict stacks,               //IN: array of adresses of similar patches
    const uint* __restrict g_num_patches_in_stack, //IN: numbers of patches in 3D groups
    float* __restrict patch_stack,                 //OUT: assembled 3D groups
    const uint2 image_dim,                         //IN: image dimensions
    const uint2 stacks_dim,                        //IN: dimensions limiting addresses of reference patches
    const Params params)                           //IN: denoising parameters
{


  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point,  params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);

  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

  patch_stack += startidx;

  const ushort* z_ptr = &stacks[ idx3(0, blockIdx.x, blockIdx.y, params.N,  gridDim.x) ];

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ];

  patch_stack[ idx3(threadIdx.x, threadIdx.y, 0, params.k, params.k) ] = (float)(image[ idx2(outer_address.x+threadIdx.x, outer_address.y+threadIdx.y, image_dim.x)]);
  for(uint i = 0; i < num_patches; ++i)
  {
    int x = (int)((signed char)(z_ptr[i] & 0xFF));
    int y = (int)((signed char)((z_ptr[i] >> 8) & 0xFF));
    patch_stack[ idx3(threadIdx.x, threadIdx.y, i+1, params.k, params.k) ] = (float)(image[ idx2(outer_address.x+x+threadIdx.x, outer_address.y+y+threadIdx.y, image_dim.x)]);
  }
}

/*
1) Do the Walsh-Hadamard 1D transform on the z axis of 3D stack.
2) Treshold every pixel and count the number of non-zero coefficients
3) Do the inverse Walsh-Hadamard 1D transform on the z axis of 3D stack.
Used parameters: L3D,N,k,p
Division: Each block delas with one transformed patch stack. (number of threads in block should be k*k)
*/
extern "C" __global__ void hard_treshold_block(
  const uint2 start_point,                        //IN: first reference patch of a batch
  float* __restrict patch_stack,                  //IN/OUT: 3D groups with thransfomed patches
  float* __restrict  w_P,                         //OUT: weight of each 3D group
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  uint2 stacks_dim,                               //IN: dimensions limiting addresses of reference patches
  const Params params,                            //IN: denoising parameters
  const uint sigma                                //IN: noise variance
)
{
  __shared__ float data[8 * 8 * 17];

  int paramN = params.N+1;
  uint tcount = blockDim.x*blockDim.y;
  uint tid = idx2(threadIdx.x, threadIdx.y, blockDim.x);
  uint patch_stack_size = tcount * paramN;

  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point, patch_stack_size, stacks_dim, params, outer_address, startidx);

  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ]+1; //+1 for the reference patch.
  float* s_patch_stack = data + (tid * (num_patches+1)); //+1 for avoiding bank conflicts //TODO:sometimes
  patch_stack = patch_stack + startidx + tid;

  //Load to the shared memory
  for(uint i = 0; i < num_patches; ++i)
    s_patch_stack[i] = patch_stack[ i*tcount ];

  //1D Transform
  fwht(s_patch_stack, num_patches);

  //Hard-thresholding + counting of nonzero coefficients
  uint nonzero = 0;
  float threshold = params.L3D * sqrtf((float)(num_patches * sigma));
  for(int i = 0; i < num_patches; ++i)
  {
    if (fabsf(s_patch_stack[ i ]) < threshold)
    {
      s_patch_stack[ i ] = 0.0f;
    }
    else
      ++nonzero;
  }

  //Inverse 1D Transform
  fwht(s_patch_stack, num_patches);

  //Normalize and save to global memory
  for (uint i = 0; i < num_patches; ++i)
  {
    patch_stack[ i*tcount ] = s_patch_stack[i] / num_patches;
  }

  //Reuse the shared memory for 32 partial sums
  __syncthreads();
  uint* shared = (uint*)data;
  //Sum the number of non-zero coefficients for a 3D group
  nonzero = blockReduceSum<uint>(shared, nonzero, tid, tcount);

  //Save the weight of a 3D group (1/nonzero coefficients)
  if (tid == 0)
  {
    if (nonzero < 1) nonzero = 1;
    w_P[ idx2(blockIdx.x, blockIdx.y, gridDim.x ) ] = 1.0f/(float)nonzero;
  }
}

/*
Fills two buffers: numerator and denominator in order to compute weighted average of pixels
Used parameters: k,N,p
Division: Each block delas with one transformed patch stack.
*/
extern "C" __global__ void aggregate_block(
  const uint2 start_point,                        //IN: first reference patch of a batch
  const float* __restrict patch_stack,            //IN: 3D groups with thransfomed patches
  const float* __restrict w_P,                    //IN: weight for each 3D group
  const ushort* __restrict stacks,                //IN: array of adresses of similar patches
  const float* __restrict kaiser_window,          //IN: kaiser window
  float* __restrict numerator,                    //IN/OUT: numerator aggregation buffer (have to be initialized to 0)
  float* __restrict denominator,                  //IN/OUT: denominator aggregation buffer (have to be initialized to 0)
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  const uint2 image_dim,                          //IN: image dimensions
  const uint2 stacks_dim,                         //IN: dimensions limiting addresses of reference patches
  const Params params                             //IN: denoising parameters
)
{
  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point, params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);

  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

  patch_stack += startidx;

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ]+1;

  float wp = w_P[ idx2(blockIdx.x, blockIdx.y, gridDim.x ) ];

  const ushort* z_ptr = &stacks[ idx3(0, blockIdx.x, blockIdx.y, params.N,  gridDim.x) ];

  float kaiser_value = kaiser_window[ idx2(threadIdx.x, threadIdx.y, params.k) ];

  for(uint z = 0; z < num_patches; ++z)
  {
    int x = 0;
    int y = 0;
    if (z > 0) {
      x = (int)((signed char)(z_ptr[z-1] & 0xFF));
      y = (int)((signed char)((z_ptr[z-1] >> 8) & 0xFF));
    }

    float value = ( patch_stack[ idx3(threadIdx.x, threadIdx.y, z, params.k, params.k) ]);
    int idx = idx2(outer_address.x + x + threadIdx.x, outer_address.y + y + threadIdx.y, image_dim.x);
    atomicAdd(numerator + idx, value * kaiser_value * wp);
    atomicAdd(denominator + idx, kaiser_value * wp);
  }
}

/*
Divide numerator with denominator and round result to image_o
*/
extern "C" __global__ void aggregate_final(
  const float* __restrict numerator,    //IN: numerator aggregation buffer
  const float* __restrict denominator,  //IN: denominator aggregation buffer
  const uint2 image_dim,                //IN: image dimensions
  uchar*__restrict result)              //OUT: image estimate
{
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= image_dim.x || idy >= image_dim.y) return;

  int value = lrintf(numerator[ idx2(idx,idy,image_dim.x) ] / denominator[ idx2(idx,idy,image_dim.x) ] );
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  result[ idx2(idx,idy,image_dim.x) ] = (uchar)value;
}




#include "indices.hpp"
/*
 * Based on dct8x8_kernel2.cu provided in CUDA samples form NVIDIA Corporation.
 *
 * Provide functions to compute many 2D DCT and 2D IDCT of size 8x8
*/


#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.


/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

#define BLOCK_SIZE          8

/**
*  Width of macro-block
*/
#define KER2_BLOCK_WIDTH          128


/**
*  Height of macro-block
*/
#define KER2_BLOCK_HEIGHT         8


/**
*  Stride of shared memory buffer (2nd kernel)
*/
#define KER2_SMEMBLOCK_STRIDE     (KER2_BLOCK_WIDTH+1)


/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void InplaceDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float X07P = (*Vect0) + (*Vect7);
    float X16P = (*Vect1) + (*Vect6);
    float X25P = (*Vect2) + (*Vect5);
    float X34P = (*Vect3) + (*Vect4);

    float X07M = (*Vect0) - (*Vect7);
    float X61M = (*Vect6) - (*Vect1);
    float X25M = (*Vect2) - (*Vect5);
    float X43M = (*Vect4) - (*Vect3);

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    (*Vect0) = C_norm * (X07P34PP + X16P25PP);
    (*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    (*Vect4) = C_norm * (X07P34PP - X16P25PP);
    (*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    (*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    (*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    (*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    (*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}


/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void InplaceIDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float Y04P   = (*Vect0) + (*Vect4);
    float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
    float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

    float Y04M   = (*Vect0) - (*Vect4);
    float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
    float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

    (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}


/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

extern "C" __global__ void DCT2D8x8(float *__restrict dst, const float *__restrict src, const uint size)
{
  __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

  if (blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (threadIdx.y+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

  int offset = threadIdx.y * (BLOCK_SIZE*BLOCK_SIZE) + threadIdx.x;

  //Get macro-block address
  src += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;
  dst += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;

  //8x1 blocks in one macro-block (threadIdx.y - index of block inside the macro-block)
  //Get the first element of the column in the block with index threadIdx.y
  src += offset;
  dst += offset;

  float *bl_ptr = block + offset;

#pragma unroll

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE]; //Load column to the shared mem

  //process rows
  InplaceDCTvector(bl_ptr - threadIdx.x + BLOCK_SIZE * threadIdx.x, 1);

  //process columns
  InplaceDCTvector(bl_ptr, BLOCK_SIZE);

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}


/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

extern "C" __global__ void IDCT2D8x8(float *__restrict dst, const float *__restrict src, const uint size)
{
  __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

  if (blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (threadIdx.y+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

  int offset = threadIdx.y * (BLOCK_SIZE*BLOCK_SIZE) + threadIdx.x;

  src += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;
  dst += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;

  src += offset;
  dst += offset;

  float *bl_ptr = block + offset;

#pragma unroll

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE];

  //process rows
  InplaceIDCTvector(bl_ptr - threadIdx.x + BLOCK_SIZE * threadIdx.x, 1);

  //process columns
      InplaceIDCTvector(bl_ptr, BLOCK_SIZE);

  for (unsigned int i = 0; i < BLOCK_SIZE; i++)
    dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}

