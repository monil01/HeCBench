#include "float2math.h"
extern "C" __global__ void scaledata(float2 *d_array, const size_t array_size, const float factor, size_t offset_0) {
  d_array += offset_0/sizeof(*d_array);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    d_array[id] *= factor;
  }
}

extern "C" __global__ void arrayadd_complex(float2 *array_c, const float2 *array_a, const float2 *array_b, const size_t array_size, float alpha, float beta, size_t offset_0, size_t offset_1, size_t offset_2) {
  array_c += offset_0/sizeof(*array_c);
  array_a += offset_1/sizeof(*array_a);
  array_b += offset_2/sizeof(*array_b);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    array_c[id] = alpha*array_a[id] + beta*array_b[id];
  }
}

extern "C" __global__ void arrayadd_real(float *array_c, const float *array_a, const float *array_b, const size_t array_size, float alpha, float beta, size_t offset_0, size_t offset_1, size_t offset_2) {
  array_c += offset_0/sizeof(*array_c);
  array_a += offset_1/sizeof(*array_a);
  array_b += offset_2/sizeof(*array_b);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    array_c[id] = alpha*array_a[id] + beta*array_b[id];
  }
}

extern "C" __global__ void arraydot(float2 *array_c, const float2 *array_a, const float2 *array_b, const size_t array_size, float alpha, size_t offset_0, size_t offset_1, size_t offset_2) {
  array_c += offset_0/sizeof(*array_c);
  array_a += offset_1/sizeof(*array_a);
  array_b += offset_2/sizeof(*array_b);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    // CUDA source uses complex operator*. HIP's built-in float2 operator*
    // is componentwise, so call the original complex formula explicitly.
    array_c[id] = cmul(alpha*array_a[id], array_b[id]);
  }
}

extern "C" __global__ void arrayreal(float2 *d_out, const float2 *d_in, size_t array_size, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    d_out[id] = make_float2(fmaxf(d_in[id].x, 0.f));
  }
}

extern "C" __global__ void update_r(float2 *__restrict__ d_r, float2 *__restrict__ d_tmp, const float2 *d_imgb, const size_t N, const float sigma, size_t offset_0, size_t offset_1, size_t offset_2) {
  d_r += offset_0/sizeof(*d_r);
  d_tmp += offset_1/sizeof(*d_tmp);
  d_imgb += offset_2/sizeof(*d_imgb);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N; id += blockDim.x * gridDim.x)
  {
    d_tmp[id] = d_tmp[id] - d_imgb[id];
    d_r[id] = (d_r[id] + sigma*d_tmp[id])/(1.f+sigma);
  }
}

extern "C" __global__ void grad_xx(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - rows*ndyn; id += blockDim.x * gridDim.x)
  {
    // (the index inside a group) + (the index of the group) * offset
    size_t idx = id%(rows*(cols-1)) + id/(rows*(cols-1))*(rows*cols);

    d_out[idx] = d_in[idx+rows] - d_in[idx];
  }
}

extern "C" __global__ void grad_xx_bound(float2 *d_array, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0) {
  d_array += offset_0/sizeof(*d_array);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*ndyn; id += blockDim.x * gridDim.x)
    {
      size_t idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

      d_array[idx] = make_float2(0.f);
    }
}

extern "C" __global__ void grad_yy(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - cols*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id%(rows-1) + id/(rows-1)*rows;

    d_out[idx] = d_in[idx+1] - d_in[idx];
  }
}

extern "C" __global__ void grad_yy_bound(float2 *d_array, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0) {
  d_array += offset_0/sizeof(*d_array);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < cols*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id*rows + (rows-1);

    d_array[idx] = make_float2(0.f);
  }
}

extern "C" __global__ void grad_tt(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, float mu, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - rows*cols; id += blockDim.x * gridDim.x)
  {
    d_out[id] = (d_in[id+rows*cols] - d_in[id])/mu;
  }
}

extern "C" __global__ void grad_tt_bound(float2 *d_array, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0) {
  d_array += offset_0/sizeof(*d_array);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*cols; id += blockDim.x * gridDim.x)
  {
    d_array[id + (ndyn-1)*rows*cols] = make_float2(0.f);
  }
}

extern "C" __global__ void proj_p(float2 *d_p, const float2 *d_tmp, const float2 *d_wbar, const size_t N, float sigma, float alpha, size_t offset_0, size_t offset_1, size_t offset_2) {
  d_p += offset_0/sizeof(*d_p);
  d_tmp += offset_1/sizeof(*d_tmp);
  d_wbar += offset_2/sizeof(*d_wbar);

  float absp, denom;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N; id += blockDim.x * gridDim.x)
  {
    d_p[id] = d_p[id] - sigma*(d_tmp[id] + d_wbar[id]);
    d_p[id+N] = d_p[id+N] - sigma*(d_tmp[id+N] + d_wbar[id+N]);
    d_p[id+2*N] = d_p[id+2*N] - sigma*(d_tmp[id+2*N] + d_wbar[id+2*N]);

    absp = sqrtf(abs(d_p[id])*abs(d_p[id]) + abs(d_p[id+N])*abs(d_p[id+N])
      + abs(d_p[id+2*N])*abs(d_p[id+2*N]));
    denom = fmaxf(1.f, absp/alpha);

    d_p[id] = d_p[id]/denom;
    d_p[id+N] = d_p[id+N]/denom;
    d_p[id+2*N] = d_p[id+2*N]/denom;
  }
}

extern "C" __global__ void grad_xx_adj(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - 2*rows*ndyn; id += blockDim.x * gridDim.x)
    {
      size_t idx = id%(rows*(cols-2)) + id/(rows*(cols-2))*(rows*cols) + rows;

      d_out[idx] = d_in[idx] - d_in[idx-rows];
    }
}

extern "C" __global__ void grad_xx_adj_zero(float2 *d_out, const float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id%rows + id/rows*(rows*cols);

    d_out[idx] = d_in[idx];
  }
}

extern "C" __global__ void grad_xx_adj_bound(float2 *d_out, const float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id%rows + id/rows*(rows*cols) + rows*(cols-1);

    d_out[idx] = -1.f*(d_in[idx-rows]);
  }
}

extern "C" __global__ void grad_yy_adj(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - 2*cols*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id%(rows-2) + id/(rows-2)*rows + 1;

    d_out[idx] = d_in[idx] - d_in[idx-1];
  }
}

extern "C" __global__ void grad_yy_adj_zero(float2 *d_out, const float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < cols*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id*rows;

    d_out[idx] = d_in[idx];
  }
}

extern "C" __global__ void grad_yy_adj_bound(float2 *d_out, const float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < cols*ndyn; id += blockDim.x * gridDim.x)
  {
    size_t idx = id*rows + (rows-1);

    d_out[idx] = -1.f*d_in[idx-1];
  }
}

extern "C" __global__ void grad_tt_adj(float2 *d_out, const float2 *d_in, const size_t N, const size_t rows, const size_t cols, const size_t ndyn, float mu, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  size_t stride = rows*cols;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N - 2*rows*cols; id += blockDim.x * gridDim.x)
  {
    d_out[id+stride] = (d_in[id+stride] - d_in[id])/mu;
  }
}

extern "C" __global__ void grad_tt_adj_zero(float2 *d_out, const float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*cols; id += blockDim.x * gridDim.x)
  {
    size_t idx = id;
    d_out[idx] = d_in[idx];
  }
}

extern "C" __global__ void grad_tt_adj_bound(float2 *d_out, float2 *d_in, const size_t rows, const size_t cols, const size_t ndyn, size_t offset_0, size_t offset_1) {
  d_out += offset_0/sizeof(*d_out);
  d_in += offset_1/sizeof(*d_in);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < rows*cols; id += blockDim.x * gridDim.x)
  {
    size_t idx = id + rows*cols*(ndyn-1);
    d_out[idx] = -1.f*d_in[idx-rows*cols];
  }
}

extern "C" __global__ void proj_q(float2 *__restrict__ d_q, float2 *__restrict__ d_tmp, const size_t N, const float sigma, const float alpha, size_t offset_0, size_t offset_1) {
  d_q += offset_0/sizeof(*d_q);
  d_tmp += offset_1/sizeof(*d_tmp);

  float absq, denom;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N; id += blockDim.x * gridDim.x)
    {
      d_tmp[id] = d_tmp[id+3*N];
      d_tmp[id+N] = d_tmp[id+4*N];
      d_tmp[id+2*N] = d_tmp[id+5*N];
      d_tmp[id+3*N] = (d_tmp[id+6*N] + d_tmp[id+7*N])/2.f;
      d_tmp[id+4*N] = (d_tmp[id+8*N] + d_tmp[id+9*N])/2.f;
      d_tmp[id+5*N] = (d_tmp[id+10*N] + d_tmp[id+11*N])/2.f;

      d_q[id] = d_q[id] - sigma*d_tmp[id];
      d_q[id+N] = d_q[id+N] - sigma*d_tmp[id+N];
      d_q[id+2*N] = d_q[id+2*N] - sigma*d_tmp[id+2*N];
      d_q[id+3*N] = d_q[id+3*N] - sigma*d_tmp[id+3*N];
      d_q[id+4*N] = d_q[id+4*N] - sigma*d_tmp[id+4*N];
      d_q[id+5*N] = d_q[id+5*N] - sigma*d_tmp[id+5*N];

      absq = sqrtf(abs(d_q[id])*abs(d_q[id])
                 + abs(d_q[id+N])*abs(d_q[id+N])
                 + abs(d_q[id+2*N])*abs(d_q[id+2*N])
                 + 2*abs(d_q[id+3*N])*abs(d_q[id+3*N])
                 + 2*abs(d_q[id+4*N])*abs(d_q[id+4*N])
                 + 2*abs(d_q[id+5*N])*abs(d_q[id+5*N]));
      denom = fmaxf(1.f, absq/alpha);

      d_q[id] = d_q[id]/denom;
      d_q[id+N] = d_q[id+N]/denom;
      d_q[id+2*N] = d_q[id+2*N]/denom;
      d_q[id+3*N] = d_q[id+3*N]/denom;
      d_q[id+4*N] = d_q[id+4*N]/denom;
      d_q[id+5*N] = d_q[id+5*N]/denom;
    }
}

extern "C" __global__ void update_s(float2 *d_imgs, const float2 *d_tmp, const float2 *d_imgz, const size_t N, const float tau, size_t offset_0, size_t offset_1, size_t offset_2) {
  d_imgs += offset_0/sizeof(*d_imgs);
  d_tmp += offset_1/sizeof(*d_tmp);
  d_imgz += offset_2/sizeof(*d_imgz);

  float2 divp;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N; id += blockDim.x * gridDim.x)
  {
    divp = d_tmp[id] + d_tmp[id+N] + d_tmp[id+2*N];
    d_imgs[id] = d_imgs[id] - tau*(d_imgz[id] + divp);
  }
}

extern "C" __global__ void update_w(float2 *d_w, const float2 *d_tmp, const float2 *d_p, const size_t N, const float tau, size_t offset_0, size_t offset_1, size_t offset_2) {
  d_w += offset_0/sizeof(*d_w);
  d_tmp += offset_1/sizeof(*d_tmp);
  d_p += offset_2/sizeof(*d_p);

  float2 divq;
  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < N; id += blockDim.x * gridDim.x)
  {
    divq = d_tmp[id+3*N] + d_tmp[id+4*N] + d_tmp[id+5*N];
    d_w[id] = d_w[id] - tau*(divq - d_p[id]);

    divq = d_tmp[id+6*N] + d_tmp[id+7*N] + d_tmp[id+8*N];
    d_w[id+N] = d_w[id+N] - tau*(divq - d_p[id+N]);

    divq = d_tmp[id+9*N] + d_tmp[id+10*N] + d_tmp[id+11*N];
    d_w[id+2*N] = d_w[id+2*N] - tau*(divq - d_p[id+2*N]);
  }
}

extern "C" __global__ void shrink(float2 *__restrict__ d_array2, float *__restrict__ d_array, const float beta, const int array_size, size_t offset_0, size_t offset_1) {
  d_array2 += offset_0/sizeof(*d_array2);
  d_array += offset_1/sizeof(*d_array);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    d_array[id] = fmaxf(d_array[id] - beta, 0.f);
    d_array2[id] = make_float2(d_array[id]);
  }
}

extern "C" __global__ void arrayabs(float *d_array, const float2 *d_array2, const size_t array_size, size_t offset_0, size_t offset_1) {
  d_array += offset_0/sizeof(*d_array);
  d_array2 += offset_1/sizeof(*d_array2);

  for (size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       id < array_size; id += blockDim.x * gridDim.x)
  {
    d_array[id] = abs(d_array2[id]);
  }
}

extern "C" __global__ void copy_bytes(unsigned char* dst,const unsigned char* src,size_t n,size_t od,size_t os){
 for(size_t i=threadIdx.x+blockIdx.x*blockDim.x;i<n;i+=blockDim.x*gridDim.x)dst[od+i]=src[os+i];
}
extern "C" __global__ void zero_bytes(unsigned char* dst,size_t n,size_t od){
 for(size_t i=threadIdx.x+blockIdx.x*blockDim.x;i<n;i+=blockDim.x*gridDim.x)dst[od+i]=0;
}
