extern "C" __global__ void waxby_kernel(
        const int n,
        const MINIFE_SCALAR alpha,
        const MINIFE_SCALAR *xcoefs,
        const MINIFE_SCALAR beta,
        const MINIFE_SCALAR *ycoefs,
        MINIFE_SCALAR *wcoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

extern "C" __global__ void yaxby_kernel(
        const int n,
        const MINIFE_SCALAR alpha,
        const MINIFE_SCALAR *xcoefs,
        const MINIFE_SCALAR beta,
        MINIFE_SCALAR *ycoefs )
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

extern "C" __global__ void wxby_kernel(
        const int n,
        const MINIFE_SCALAR *xcoefs,
        const MINIFE_SCALAR beta,
        const MINIFE_SCALAR *ycoefs,
        MINIFE_SCALAR *wcoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

extern "C" __global__ void yxby_kernel(
        const int n,
        const MINIFE_SCALAR *xcoefs,
        const MINIFE_SCALAR beta,
        MINIFE_SCALAR *ycoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

extern "C" __global__ void wx_kernel(
        const int n,
        const MINIFE_SCALAR *xcoefs,
        MINIFE_SCALAR *wcoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = xcoefs[idx];
    }

extern "C" __global__ void dyx_kernel(
        const int n,
        const MINIFE_SCALAR *xcoefs,
        MINIFE_SCALAR *ycoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += xcoefs[idx];
    }

extern "C" __global__ void wax_kernel(
        const int n,
        const MINIFE_SCALAR alpha,
        const MINIFE_SCALAR *xcoefs,
        MINIFE_SCALAR *wcoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx];
    }

extern "C" __global__ void dyax_kernel(
        const int n,
        const MINIFE_SCALAR alpha,
        const MINIFE_SCALAR *xcoefs,
        MINIFE_SCALAR *ycoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += alpha * xcoefs[idx];
    }

extern "C" __global__ void dot_kernel(const MINIFE_LOCAL_ORDINAL n,
        const MINIFE_SCALAR* x,
        const MINIFE_SCALAR* y,
              MINIFE_SCALAR* d)
    {
      MINIFE_SCALAR sum=0;
      for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<n;idx+=gridDim.x*blockDim.x) {
        sum+=x[idx]*y[idx];
      }

      //Do a shared memory reduction on the dot product
      __shared__ MINIFE_SCALAR red[256];
      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = 256 / 2; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }

      //save partial dot products
      if(threadIdx.x==0) d[blockIdx.x]=sum;
    }

extern "C" __global__ void final_reduce(MINIFE_SCALAR *d) {
      MINIFE_SCALAR sum = d[threadIdx.x];
      __shared__ MINIFE_SCALAR red[256];

      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = 256 / 2; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }
      //save final dot product at the front
      if(threadIdx.x==0) d[0]=sum;
    }

extern "C" __global__ void matvec_kernel(const MINIFE_LOCAL_ORDINAL rows_size,
        const MINIFE_LOCAL_ORDINAL *Arowoffsets,
        const MINIFE_GLOBAL_ORDINAL *Acols,
        const MINIFE_SCALAR *Acoefs,
        const MINIFE_SCALAR *xcoefs,
        MINIFE_SCALAR *ycoefs)
    {
      MINIFE_LOCAL_ORDINAL row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < rows_size) {
        MINIFE_GLOBAL_ORDINAL row_start = Arowoffsets[row];
        MINIFE_GLOBAL_ORDINAL row_end   = Arowoffsets[row+1];
        MINIFE_SCALAR sum = 0;

        // Use the unroll factor in the OpenMP program 
#pragma unroll 27
        for(MINIFE_GLOBAL_ORDINAL i = row_start; i < row_end; ++i) {
          sum += Acoefs[i] * xcoefs[Acols[i]];
        }
        ycoefs[row] = sum;
      }
    }
