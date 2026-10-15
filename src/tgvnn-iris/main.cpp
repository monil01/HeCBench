/*
  This file is part of the TGV package (https://github.com/chixindebaoyu/tgvnn).

  The MIT License (MIT)

  Copyright (c) Dong Wang

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <chrono>

#include <hipfft/hipfft.h>
#include <hipsolver/hipsolver.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "float2math.h"
#include "ra.h"
#include "dmem_ops.hpp"
#define TGVNN_IRIS 1
#include "observe.hpp"

//Configuration parameters
int blocksize = 128;
int gridsize = 128;
#include "launches.hpp"

//HIP global variables
hipStream_t stream[2];

hipblasStatus_t hipblas_status;
int hipblas_handle;

hipsolverStatus_t cusolver_status;
int cusolver_handle;

int fft2_plan; // Host compatibility token; actual plan is callback-owned.

// Useful kernels
void
gpuAssert(hipError_t code, const char *file, int line)
{
    if (code != hipSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code),
            file, line);
        getchar();
        exit(code);
    }
}
#define cuTry(ans) gpuAssert((ans), __FILE__, __LINE__);

void *
mallocAssert(const size_t n, const char *file, int line)
{
    void *p = malloc(n);
    if (p == NULL)
    {
        printf("Malloc failed from %s:%d", file, line);
        abort();
    }
    return p;
}
#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);

void
freeAssert(void *p, const char *file, int line)
{
    if (p == NULL)
    {
        printf("Free failed from %s:%d", file, line);
        abort();
    }
}
#define safe_free(x) freeAssert(x, __FILE__, __LINE__);

void
print_usage()
{
    fprintf(stderr, "Dynamic MRI reconstruction using Primal-Dual algorithm\n");
    fprintf(stderr, "Decomposition of second order TGV and nuclear norm\n");
    fprintf(stderr, "Usage: runtgv [OPTION] <img.ra> <mask.ra>\n");
    fprintf(stderr, "\t-o, --output <output.ra>\t recon output RA file\n");
    fprintf(stderr, "\t-i, --iter n\t\t\t iteration number\n");
    fprintf(stderr, "\t-a, --alpha n\t\t\t parameter for TGV\n");
    fprintf(stderr, "\t-b, --beta n\t\t\t parameter for nuclear norm\n");
    fprintf(stderr, "\t-s, --sigma n\t\t\t dual stepsize\n");
    fprintf(stderr, "\t-t, --tau n\t\t\t primal stepsize\n");
    fprintf(stderr, "\t-m, --mu n\t\t\t temporal stepsize\n");
    fprintf(stderr, "\t-G, --gridsize n\t\t set GPU gridsize\n");
    fprintf(stderr, "\t-B, --blocksize n\t\t set GPU blocksize\n");
    fprintf(stderr, "\t-h\t\t\t\t show this help\n");
}

void
save_rafile(float2 *h_out, const char *out_path,
  const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3)
{
    // save out
    ra_t out;
    out.flags = 0;
    out.eltype = RA_TYPE_COMPLEX;
    out.elbyte = sizeof(float2);
    out.size = dim0 * dim1 * dim2 * dim3 * sizeof(float2);
    out.ndims = 4;
    out.dims = (uint64_t *) safe_malloc(out.ndims * sizeof(uint64_t));
    out.dims[0] = dim0;
    out.dims[1] = dim1;
    out.dims[2] = dim2;
    out.dims[3] = dim3;
    out.data = (uint8_t *) h_out;
    ra_write(&out, out_path);
    ra_free(&out);
}



float
compute_alpha (float alpha0, float alpha1, int iter, int index)
{
  return expf(float(index)/float(iter)*logf(alpha1) +
          (float(iter)-float(index))/float(iter)*logf(alpha0));
}

float
compute_maxmag (float2 *d_array, const size_t array_size)
{
  long max_idx;
  hipblas_status = tgv_imax(hipblas_handle, array_size, d_array, 1, &max_idx);
  if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
     printf ("tgv_imax failed:%d\n", hipblas_status);
     exit(EXIT_FAILURE);
  }

  float2 *h_tmp;
  h_tmp = (float2*)safe_malloc(sizeof(float2));
  cuTry(tgv_dmem::copy(h_tmp, d_array+max_idx-1, sizeof(float2),
        hipMemcpyDeviceToHost));

  return abs(h_tmp[0].x);
}









void
fft2_init(const int rows, const int cols, const int ndyn)
{
  // setup FFT plan, ndyn = # timepoints
  const int rank = 2;
  int idist = 1, odist = 1, istride = ndyn, ostride = ndyn;
  int n[rank] = {cols, rows};

  tgv_fft_plan(&fft2_plan, rank, n,
      NULL, istride, idist,
      NULL, ostride, odist,
      HIPFFT_C2C, ndyn);
}

void
forward (float2 *d_out, float2 *d_in, float2 *d_mask,
  const size_t N, const size_t rows, const size_t cols)
{
  tgv_fft_exec(fft2_plan, d_in, d_out, HIPFFT_FORWARD);
  launch_scaledata(d_out, N, 1.f / sqrtf(rows*cols));
  launch_arraydot(d_out, d_out, d_mask, N);
}

void
backward (float2 *d_out, float2 *d_in, float2 *d_mask,
  const size_t N, const size_t rows, const size_t cols)
{
  float2 *d_tmp;
  cuTry(tgv_dmem::alloc((void **)&d_tmp, N * sizeof(float2)));

  cuTry(tgv_dmem::copy(d_tmp, d_in, N * sizeof(float2),
        hipMemcpyDeviceToDevice, stream[0]));

  launch_arraydot(d_tmp, d_tmp, d_mask, N);
  tgv_fft_exec(fft2_plan, d_tmp, d_out, HIPFFT_BACKWARD);
  launch_scaledata(d_out, N, 1.f / sqrtf(rows*cols));

  cuTry(tgv_dmem::release(d_tmp));
}



void
update_r (float2 *d_r, float2 *d_lbar, float2 *d_sbar,
  float2 *d_imgb, float2 *d_tmp, float2 *d_mask, const float sigma,
  const size_t N, const size_t rows, const size_t cols)
{
  launch_arrayadd(d_tmp, d_lbar, d_sbar, N, 1.f, 1.f);
  forward(d_tmp, d_tmp, d_mask, N, rows, cols);
  launch_update_r(d_r, d_tmp, d_imgb, N, sigma);
}













void
grad (float2 *d_out, float2 *d_in, const size_t N,
    const size_t rows, const size_t cols, const size_t ndyn,
    float mu, char mode)
{
  switch (mode)
  {
    case 'x':
      launch_grad_xx(d_out, d_in, N, rows, cols, ndyn);
      launch_grad_xx_bound(d_out, N, rows, cols, ndyn);
      break;
    case 'y':
      launch_grad_yy(d_out, d_in, N, rows, cols, ndyn);
      launch_grad_yy_bound(d_out, N, rows, cols, ndyn);
      break;
    case 't':
      launch_grad_tt(d_out, d_in, N, rows, cols, ndyn, mu);
      launch_grad_tt_bound(d_out, N, rows, cols, ndyn);
      break;
    default:
      printf("Please select x, y or z!\n");
      break;
  }
}





















void
grad_adj (float2 *d_out, float2 *d_in, const size_t N,
  const size_t rows, const size_t cols, const size_t ndyn,
  float mu, char mode)
{
  switch (mode)
  {
    case 'x':
      launch_grad_xx_adj(d_out, d_in, N, rows, cols, ndyn);
      launch_grad_xx_adj_zero(d_out, d_in, rows, cols, ndyn);
      launch_grad_xx_adj_bound(d_out, d_in, rows, cols, ndyn);
      break;
    case 'y':
      launch_grad_yy_adj(d_out, d_in, N, rows, cols, ndyn);
      launch_grad_yy_adj_zero(d_out, d_in, rows, cols, ndyn);
      launch_grad_yy_adj_bound(d_out, d_in, rows, cols, ndyn);
      break;
    case 't':
      launch_grad_tt_adj(d_out, d_in, N, rows, cols, ndyn, mu);
      launch_grad_tt_adj_zero(d_out, d_in, rows, cols, ndyn);
      launch_grad_tt_adj_bound(d_out, d_in, rows, cols, ndyn);
      break;
    default:
      printf("Please select x, y or z!\n");
      break;
  }
}











float
compute_ser(float2 *d_array, float2 *d_img, const size_t array_size)
{
  float *d_diff, *d_tmp, ser, diff_norm, img_norm;

  cuTry(tgv_dmem::alloc((void **)&d_diff, array_size * sizeof(float)));
  cuTry(tgv_dmem::alloc((void **)&d_tmp, array_size * sizeof(float)));

  launch_arrayabs(d_diff, d_array, array_size);
  launch_arrayabs(d_tmp, d_img, array_size);
  launch_arrayadd(d_diff, d_diff, d_tmp, array_size, 1.f, -1.f);

  hipblas_status = tgv_nrm2(hipblas_handle, array_size, d_tmp, 1, &img_norm);
  assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

  hipblas_status = tgv_nrm2(hipblas_handle, array_size, d_diff, 1, &diff_norm);
  assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

  ser = -20.f*log10f(diff_norm/img_norm);

  cuTry(tgv_dmem::release(d_diff));
  cuTry(tgv_dmem::release(d_tmp));

  return ser;
}

void
tgv_cs(float2 *d_imgl, float2 *d_imgs, float2 *h_img, float2 *h_mask,
      const size_t N, const size_t rows, const size_t cols, const size_t ndyn,
      float alpha, float beta, float mu,
      float tau, float sigma, float reduction, int iter)
{
  // Setup cuda handles
  hipblas_status = tgv_blas_create(&hipblas_handle);
  cusolver_status = tgv_solver_create(&cusolver_handle);

  // Read image and mask
  float2 *d_img, *d_imgz, *d_imgb, *d_mask;

  cuTry(tgv_dmem::alloc((void **) &d_img, N * sizeof(float2)));
  cuTry(tgv_dmem::copy(d_img, h_img, N * sizeof(float2),
          hipMemcpyHostToDevice, stream[0]));

  cuTry(tgv_dmem::alloc((void **) &d_mask, N * sizeof(float2)));
  cuTry(tgv_dmem::copy(d_mask, h_mask, N * sizeof(float2),
          hipMemcpyHostToDevice, stream[0]));

  // Scale the data to [0,1]
  float img_max = compute_maxmag(d_img, N);
  launch_scaledata(d_img, N, 1.f / img_max);

  // Sample image using mask and generate zerofilled image
  cuTry(tgv_dmem::alloc((void **) &d_imgz, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **) &d_imgb, N * sizeof(float2)));

  tgv_observe("normalized",d_img,N*sizeof(float2));
  fft2_init(rows, cols, ndyn);

  forward(d_imgb, d_img, d_mask, N, rows, cols);
  tgv_observe("sampled",d_imgb,N*sizeof(float2));
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);
  tgv_observe("zerofill",d_imgz,N*sizeof(float2));

  // Allocate arrays
  float2 *d_w; // Primal variables
  float2 *d_p, *d_q, *d_r; // Dual variables
  float2 *d_lbar, *d_sbar, *d_wbar; // Intermediate variables
  float2 *d_lold, *d_sold, *d_wold; // Old variables
  float2 *d_tmp; // Temporal variable (grad, grad_adj ect)

  cuTry(tgv_dmem::alloc((void **)&d_w, 3 * N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_p, 3 * N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_q, 6 * N *sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_r, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_lbar, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_sbar, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_wbar, 3 * N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_lold, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_sold, N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_wold, 3 * N * sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_tmp, 12 * N * sizeof(float2)));

  // Initialization for variables
  launch_arrayreal(d_imgl, d_imgz, N);
  cuTry(tgv_dmem::zero(d_imgs, 0.f, N * sizeof(float2)));
  cuTry(tgv_dmem::zero(d_w, 0.f, 3 * N * sizeof(float2)));

  cuTry(tgv_dmem::copy(d_sbar, d_imgs, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
  cuTry(tgv_dmem::copy(d_lbar, d_imgl, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
  cuTry(tgv_dmem::copy(d_wbar, d_w, 3 * N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));

  cuTry(tgv_dmem::zero(d_p, 0.f, 3 * N * sizeof(float2)));
  cuTry(tgv_dmem::zero(d_q, 0.f, 6 * N * sizeof(float2)));
  cuTry(tgv_dmem::zero(d_r, 0.f, N * sizeof(float2)));

  cuTry(tgv_dmem::zero(d_tmp, 0.f, 12 * N * sizeof(float2)));

  // Initialization for alpha
  float alpha0 = sqrtf(2)*alpha;
  float alpha1 = alpha;

  float alpha00 = alpha0;
  float alpha10 = alpha1;
  float alpha01 = alpha0*reduction;
  float alpha11 = alpha1*reduction;

  // Initialization for SVD
  int lwork, lda, ldu, lds, ldvt, *devInfo;

  cusolver_status = tgv_svd_size(cusolver_handle,
      rows*cols, ndyn, &lwork);
  assert(cusolver_status == HIPSOLVER_STATUS_SUCCESS);

  lda = rows*cols;
  ldu = rows*cols;
  lds = fminf(ndyn, rows*cols);
  ldvt = ndyn;

  cuTry(tgv_dmem::alloc((void **)&devInfo, sizeof(int)));

  float2 *d_lu, *d_ls2, *d_lvt, *d_lsvt, *d_work, a, b;
  float *d_ls, *d_rwork;

  cuTry(tgv_dmem::alloc((void **)&d_lu, ldu*lds*sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_ls, lds*sizeof(float)));
  cuTry(tgv_dmem::alloc((void **)&d_ls2, lds*sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_lvt, lds*ldvt*sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_lsvt, lds*ldvt*sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_work, lwork*sizeof(float2)));
  cuTry(tgv_dmem::alloc((void **)&d_rwork, (lds-1)*sizeof(float)));

  a = make_float2(1.f);
  b = make_float2(0.f);

  // Run the main loop
  printf("Running recon ...\n");
  for (int i = 1; i <= iter; i++)
  {
    // update alpha
    alpha0 = compute_alpha(alpha00, alpha01, iter, i);
    alpha1 = compute_alpha(alpha10, alpha11, iter, i);

    cuTry(tgv_dmem::copy(d_lold, d_imgl, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
    cuTry(tgv_dmem::copy(d_sold, d_imgs, N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));
    cuTry(tgv_dmem::copy(d_wold, d_w, 3 * N * sizeof(float2),
          hipMemcpyDeviceToDevice, stream[0]));

    // update r
    update_r(d_r, d_lbar, d_sbar, d_imgb, d_tmp, d_mask,
      sigma, N, rows, cols);

    // update p
    grad(d_tmp, d_sbar, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+N, d_sbar, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+2*N, d_sbar, N, rows, cols, ndyn, mu, 't');

    launch_proj_p(d_p, d_tmp, d_wbar, N, sigma, alpha1);

    // update q
    grad_adj(d_tmp+3*N, d_wbar, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+4*N, d_wbar+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+5*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+6*N, d_wbar, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+7*N, d_wbar+N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+8*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+9*N, d_wbar, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+10*N, d_wbar+N, N, rows, cols, ndyn, mu, 't');
    grad_adj(d_tmp+11*N, d_wbar+2*N, N, rows, cols, ndyn, mu, 'y');

    launch_proj_q(d_q, d_tmp, N, sigma, alpha0);

    // update s
    backward(d_imgz, d_r, d_mask, N, rows, cols);

    grad_adj(d_tmp, d_p, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+N, d_p+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+2*N, d_p+2*N, N, rows, cols, ndyn, mu, 't');

    launch_update_s(d_imgs, d_tmp, d_imgz, N, tau);

    // update w
    grad(d_tmp+3*N, d_q, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+4*N, d_q+3*N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+5*N, d_q+4*N, N, rows, cols, ndyn, mu, 't');
    grad(d_tmp+6*N, d_q+3*N, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+7*N, d_q+N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+8*N, d_q+5*N, N, rows, cols, ndyn, mu, 't');
    grad(d_tmp+9*N, d_q+4*N, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+10*N, d_q+5*N, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+11*N, d_q+2*N, N, rows, cols, ndyn, mu, 't');

    launch_update_w(d_w, d_tmp, d_p, N, tau);

    // update l
    launch_arrayadd(d_imgl, d_imgl, d_imgz, N, 1.f, -1.f*tau);

    if(i==1)tgv_observe("prox_input",d_imgl,N*sizeof(float2));
    cusolver_status = tgv_svd(cusolver_handle, 'S', 'S',
      rows*cols, ndyn, d_imgl, lda, d_ls, d_lu, ldu, d_lvt, ldvt,
      d_work, lwork, d_rwork, devInfo);
    assert(cusolver_status == HIPSOLVER_STATUS_SUCCESS);

    launch_shrink(d_ls2, d_ls, beta, lds);

    hipblas_status = tgv_dgmm(hipblas_handle, HIPBLAS_SIDE_LEFT,
      lds, ldvt, d_lvt, ldvt, d_ls2, 1, d_lsvt, ldvt);
    assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

    hipblas_status = tgv_gemm(hipblas_handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
      ldu, ldvt, lds, &a, d_lu, ldu, d_lsvt, ldvt, &b, d_imgl, ldu);
    assert(hipblas_status == HIPBLAS_STATUS_SUCCESS);

    if(i==1)tgv_observe("prox_output",d_imgl,N*sizeof(float2));
    if(i==1)tgv_observe("singular",d_ls,lds*sizeof(float));
    // update intermediate variables
    launch_arrayadd(d_sbar, d_imgs, d_sold, N, 2.f, -1.f);
    launch_arrayadd(d_wbar, d_w, d_wold, 3*N, 2.f, -1.f);
    launch_arrayadd(d_lbar, d_imgl, d_lold, N, 2.f, -1.f);
  }

  // Compute SER
  float2 *d_imgr;
  cuTry(tgv_dmem::alloc((void **)&d_imgr, N * sizeof(float2)));

  launch_arrayadd(d_imgr, d_imgl, d_imgs, N, 1.f, 1.f);
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);

  float ser_imgz, ser_imgr;
  ser_imgz = compute_ser(d_imgz, d_img, N);
  ser_imgr = compute_ser(d_imgr, d_img, N);

  printf("The SER of zerofill: %.2f dB\n", ser_imgz);
  printf("The SER of recon:    %.2f dB\n", ser_imgr);

  // Free arrays and destroy handles
  cuTry(tgv_dmem::release(d_img));
  cuTry(tgv_dmem::release(d_imgb));
  cuTry(tgv_dmem::release(d_imgz));
  cuTry(tgv_dmem::release(d_imgr));
  cuTry(tgv_dmem::release(d_mask));

  cuTry(tgv_dmem::release(d_w));
  cuTry(tgv_dmem::release(d_p));
  cuTry(tgv_dmem::release(d_q));
  cuTry(tgv_dmem::release(d_r));
  cuTry(tgv_dmem::release(d_lbar));
  cuTry(tgv_dmem::release(d_sbar));
  cuTry(tgv_dmem::release(d_wbar));
  cuTry(tgv_dmem::release(d_lold));
  cuTry(tgv_dmem::release(d_sold));
  cuTry(tgv_dmem::release(d_wold));
  cuTry(tgv_dmem::release(d_tmp));

  cuTry(tgv_dmem::release(d_lu));
  cuTry(tgv_dmem::release(d_ls));
  cuTry(tgv_dmem::release(d_ls2));
  cuTry(tgv_dmem::release(d_lvt));
  cuTry(tgv_dmem::release(d_lsvt));
  cuTry(tgv_dmem::release(d_work));
  cuTry(tgv_dmem::release(d_rwork));
  cuTry(tgv_dmem::release(devInfo));

  tgv_blas_destroy(hipblas_handle);
  tgv_solver_destroy(cusolver_handle);
}

int
main (int argc, char *argv[])
{
    // Setup default values
    const char *in_img, *in_mask, *out_img;
    in_img = "../tgvnn-cuda/data/pincat.ra";
    in_mask = "../../tgvnn-cuda/data/pincat_mask.ra";
    out_img = "pincat_recon.ra";

    float alpha = 0.004;
    float beta = 0.5;
    float sigma = 0.25;
    float tau = 0.25;
    float mu = 1.f;
    float reduction = 0.5; // No need to change this
    int iter = 500;

    // Read command line
    struct option long_options[] =
    {
        {"iter", 1, NULL, 'i'},
        {"alpha", 1, NULL, 'a'},
        {"beta", 1, NULL, 'b'},
        {"sigma", 1, NULL, 's'},
        {"tau", 1, NULL, 't'},
        {"mu", 1, NULL, 'm'},
        {"output", 1, NULL, 'o'},
        {"gridsize", 1, NULL, 'G'},
        {"blocksize", 1, NULL, 'B'},
        {"help", 0, 0, 'h'}
    };

    extern int optind;
    opterr = 0;
    int option_index = 0;
    int c;
    while ((c =
      getopt_long(argc, argv, "i:a:b:s:t:m:o:G:B:h",
        long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'i':
                iter = atoi(optarg);
                break;
            case 'a':
                alpha = atof(optarg);
                break;
            case 'b':
                beta = atof(optarg);
                break;
            case 's':
                sigma = atof(optarg);
                break;
            case 't':
                tau = atof(optarg);
                break;
            case 'm':
                mu = atof(optarg);
            case 'o':
                out_img = optarg;
                break;
            case 'G':
                gridsize = atoi(optarg);
                break;
            case 'B':
                blocksize = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return 1;
        }
    }

    int ra_count = 0;
    const char *tmp;
    while (optind <= argc-1)
    {
      tmp = argv[optind];

      if (strstr(tmp, ".ra") != NULL)
      {
        if (ra_count == 0)
        {
          in_img = tmp;
          ra_count ++;
          optind ++;
        }
        else if (ra_count == 1)
        {
          in_mask = tmp;
          ra_count ++;
          optind ++;
        }
        else
        {
          break;
        }
      }
      else
      {
        optind ++;
      }
    }

    // Read input data and mask
    printf("Reading input ...\n");
    printf("Input image: %s\n", in_img);
    printf("Input mask:  %s\n", in_mask);
    printf("gridsize:  %d\n", gridsize);
    printf("blocksize: %d\n", blocksize);
    printf("alpha: %.4f\n", alpha);
    printf("beta:  %.4f\n", beta);
    printf("sigma: %.4f\n", sigma);
    printf("tau:   %.4f\n", tau);
    printf("iter:  %d\n", iter);

    ra_t ra_img, ra_mask;

    ra_read(&ra_img, in_img);
    ra_read(&ra_mask, in_mask);

    float2 *h_img = (float2*)ra_img.data;
    float2 *h_mask = (float2*)ra_mask.data;

    size_t rows = ra_img.dims[0];
    size_t cols = ra_img.dims[1];
    size_t ndyn = ra_img.dims[2];
    size_t N = ndyn*rows*cols;

    printf("rows = %lu, cols = %lu, ndyn = %lu, N = %lu\n",
      rows, cols, ndyn, N);
    printf("\n");

    // Run TGV+NN

    // Setup HIP
    if(rows<3||cols<3||ndyn<3||ndyn>rows*cols||N>INT_MAX||gridsize<1||blocksize<1||iter<1) {
      fprintf(stderr,"Unsupported TGV dimensions/launch/iteration domain\n");return 1;
    }
    setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);
    iris::Platform platform;
    tgv_dmem::check(platform.init(&argc,&argv,1));

    float2 *d_imgl, *d_imgs;
    cuTry(tgv_dmem::alloc((void **) &d_imgl, N * sizeof(float2)));
    cuTry(tgv_dmem::alloc((void **) &d_imgs, N * sizeof(float2)));

    auto start = std::chrono::steady_clock::now();

    tgv_cs(d_imgl, d_imgs, h_img, h_mask, N, rows, cols, ndyn,
           alpha, beta, mu, tau, sigma, reduction, iter);
    tgv_dmem::check(iris_synchronize());

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total execution time of tgv: %f (s)\n", time * 1e-9f);

    // Save results
    printf("Saving output ...\n");
    printf("Output: %s\n", out_img);

    float2 *h_out;
    h_out = (float2 *)safe_malloc(2 * N * sizeof(float2));
    cuTry(tgv_dmem::copy(h_out, d_imgl, N * sizeof(float2),
          hipMemcpyDeviceToHost, stream[0]));
    cuTry(tgv_dmem::copy(h_out+N, d_imgs, N * sizeof(float2),
          hipMemcpyDeviceToHost, stream[0]));
    save_rafile(h_out, out_img, rows, cols, ndyn, 2);

    // Free memory and destory cuda stream
    ra_free(&ra_img);
    ra_free(&ra_mask);
    safe_free(h_out);
    cuTry(tgv_dmem::release(d_imgl));
    cuTry(tgv_dmem::release(d_imgs));

    tgv_dmem::finish();
    if(iris_error_count())std::abort();
    tgv_dmem::check(platform.finalize());

    return 0;
}
