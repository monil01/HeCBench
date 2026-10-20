void
observed_tgv_cs(float2 *d_imgl, float2 *d_imgs, float2 *h_img, float2 *h_mask,
      const size_t N, const size_t rows, const size_t cols, const size_t ndyn,
      float alpha, float beta, float mu,
      float tau, float sigma, float reduction, int iter)
{
  // Setup cuda handles
  cublas_status = cublasCreate(&cublas_handle);
  cusolver_status = cusolverDnCreate(&cusolver_handle);

  // Read image and mask
  float2 *d_img, *d_imgz, *d_imgb, *d_mask;

  cuTry(cudaMalloc((void **) &d_img, N * sizeof(float2)));
  cuTry(cudaMemcpyAsync(d_img, h_img, N * sizeof(float2),
          cudaMemcpyHostToDevice, stream[0]));

  cuTry(cudaMalloc((void **) &d_mask, N * sizeof(float2)));
  cuTry(cudaMemcpyAsync(d_mask, h_mask, N * sizeof(float2),
          cudaMemcpyHostToDevice, stream[0]));

  observe_inputs(d_img,d_mask,N);

  // Scale the data to [0,1]
  float img_max = compute_maxmag(d_img, N);
  scaledata<<<gridsize, blocksize>>>(d_img, N, 1.f / img_max);

  // Sample image using mask and generate zerofilled image
  cuTry(cudaMalloc((void **) &d_imgz, N * sizeof(float2)));
  cuTry(cudaMalloc((void **) &d_imgb, N * sizeof(float2)));

  fft2_init(rows, cols, ndyn);

  forward(d_imgb, d_img, d_mask, N, rows, cols);
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);

  // Allocate arrays
  float2 *d_w; // Primal variables
  float2 *d_p, *d_q, *d_r; // Dual variables
  float2 *d_lbar, *d_sbar, *d_wbar; // Intermediate variables
  float2 *d_lold, *d_sold, *d_wold; // Old variables
  float2 *d_tmp; // Temporal variable (grad, grad_adj ect)

  cuTry(cudaMalloc((void **)&d_w, 3 * N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_p, 3 * N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_q, 6 * N *sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_r, N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_lbar, N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_sbar, N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_wbar, 3 * N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_lold, N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_sold, N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_wold, 3 * N * sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_tmp, 12 * N * sizeof(float2)));

  // Initialization for variables
  arrayreal<<<gridsize, blocksize>>>(d_imgl, d_imgz, N);
  cuTry(cudaMemset(d_imgs, 0.f, N * sizeof(float2)));
  cuTry(cudaMemset(d_w, 0.f, 3 * N * sizeof(float2)));

  cuTry(cudaMemcpyAsync(d_sbar, d_imgs, N * sizeof(float2),
          cudaMemcpyDeviceToDevice, stream[0]));
  cuTry(cudaMemcpyAsync(d_lbar, d_imgl, N * sizeof(float2),
          cudaMemcpyDeviceToDevice, stream[0]));
  cuTry(cudaMemcpyAsync(d_wbar, d_w, 3 * N * sizeof(float2),
          cudaMemcpyDeviceToDevice, stream[0]));

  cuTry(cudaMemset(d_p, 0.f, 3 * N * sizeof(float2)));
  cuTry(cudaMemset(d_q, 0.f, 6 * N * sizeof(float2)));
  cuTry(cudaMemset(d_r, 0.f, N * sizeof(float2)));

  cuTry(cudaMemset(d_tmp, 0.f, 12 * N * sizeof(float2)));

  // Initialization for alpha
  float alpha0 = sqrtf(2)*alpha;
  float alpha1 = alpha;

  float alpha00 = alpha0;
  float alpha10 = alpha1;
  float alpha01 = alpha0*reduction;
  float alpha11 = alpha1*reduction;

  // Initialization for SVD
  int lwork, lda, ldu, lds, ldvt, *devInfo;

  cusolver_status = cusolverDnCgesvd_bufferSize(cusolver_handle,
      rows*cols, ndyn, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  lda = rows*cols;
  ldu = rows*cols;
  lds = fminf(ndyn, rows*cols);
  ldvt = ndyn;

  cuTry(cudaMalloc((void **)&devInfo, sizeof(int)));

  float2 *d_lu, *d_ls2, *d_lvt, *d_lsvt, *d_work, a, b;
  float *d_ls, *d_rwork;

  cuTry(cudaMalloc((void **)&d_lu, ldu*lds*sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_ls, lds*sizeof(float)));
  cuTry(cudaMalloc((void **)&d_ls2, lds*sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_lvt, lds*ldvt*sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_lsvt, lds*ldvt*sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_work, lwork*sizeof(float2)));
  cuTry(cudaMalloc((void **)&d_rwork, (lds-1)*sizeof(float)));

  a = make_float2(1.f);
  b = make_float2(0.f);

  // Run the main loop
  for (int i = 1; i <= iter; i++)
  {
    // update alpha
    alpha0 = compute_alpha(alpha00, alpha01, iter, i);
    alpha1 = compute_alpha(alpha10, alpha11, iter, i);

    cuTry(cudaMemcpyAsync(d_lold, d_imgl, N * sizeof(float2),
            cudaMemcpyDeviceToDevice, stream[0]));
    cuTry(cudaMemcpyAsync(d_sold, d_imgs, N * sizeof(float2),
            cudaMemcpyDeviceToDevice, stream[0]));
    cuTry(cudaMemcpyAsync(d_wold, d_w, 3 * N * sizeof(float2),
            cudaMemcpyDeviceToDevice, stream[0]));

    // update r
    update_r(d_r, d_lbar, d_sbar, d_imgb, d_tmp, d_mask,
      sigma, N, rows, cols);

    if (i==3) observe_trace("r",d_r,N);

    // update p
    grad(d_tmp, d_sbar, N, rows, cols, ndyn, mu, 'x');
    grad(d_tmp+N, d_sbar, N, rows, cols, ndyn, mu, 'y');
    grad(d_tmp+2*N, d_sbar, N, rows, cols, ndyn, mu, 't');

    proj_p<<<gridsize, blocksize>>>(d_p, d_tmp, d_wbar, N, sigma, alpha1);

    if (i==3) observe_trace("p",d_p,3*N);

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

    proj_q<<<gridsize, blocksize>>>(d_q, d_tmp, N, sigma, alpha0);

    if (i==3) observe_trace("q",d_q,6*N);

    // update s
    backward(d_imgz, d_r, d_mask, N, rows, cols);

    grad_adj(d_tmp, d_p, N, rows, cols, ndyn, mu, 'x');
    grad_adj(d_tmp+N, d_p+N, N, rows, cols, ndyn, mu, 'y');
    grad_adj(d_tmp+2*N, d_p+2*N, N, rows, cols, ndyn, mu, 't');

    update_s<<<gridsize, blocksize>>>(d_imgs, d_tmp, d_imgz, N, tau);

    if (i==3) observe_trace("s",d_imgs,N);

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

    update_w<<<gridsize, blocksize>>>(d_w, d_tmp, d_p, N, tau);

    if (i==3) observe_trace("w",d_w,3*N);

    // update l
    arrayadd<<<gridsize, blocksize>>>(d_imgl, d_imgl, d_imgz, N, 1.f, -1.f*tau);

    cusolver_status = cusolverDnCgesvd(cusolver_handle, 'S', 'S',
      rows*cols, ndyn, d_imgl, lda, d_ls, d_lu, ldu, d_lvt, ldvt,
      d_work, lwork, d_rwork, devInfo);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    shrink<<<gridsize, blocksize>>>(d_ls2, d_ls, beta, lds);

    cublas_status = cublasCdgmm_64(cublas_handle, CUBLAS_SIDE_LEFT,
      lds, ldvt, d_lvt, ldvt, d_ls2, 1, d_lsvt, ldvt);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    cublas_status = cublasCgemm_64(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
      ldu, ldvt, lds, &a, d_lu, ldu, d_lsvt, ldvt, &b, d_imgl, ldu);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);

    if (i==3) observe_trace("l",d_imgl,N);

    // update intermediate variables
    arrayadd<<<gridsize, blocksize>>>(d_sbar, d_imgs, d_sold, N, 2.f, -1.f);
    arrayadd<<<gridsize, blocksize>>>(d_wbar, d_w, d_wold, 3*N, 2.f, -1.f);
    arrayadd<<<gridsize, blocksize>>>(d_lbar, d_imgl, d_lold, N, 2.f, -1.f);
    if(i==3){ observe_trace("lb",d_lbar,N);observe_trace("sb",d_sbar,N);observe_trace("wb",d_wbar,3*N);observe_trace("tmp",d_tmp,12*N); }
  }

  // Compute SER
  float2 *d_imgr;
  cuTry(cudaMalloc((void **)&d_imgr, N * sizeof(float2)));

  arrayadd<<<gridsize, blocksize>>>(d_imgr, d_imgl, d_imgs, N, 1.f, 1.f);
  backward(d_imgz, d_imgb, d_mask, N, rows, cols);

  float ser_imgz, ser_imgr;
  ser_imgz = compute_ser(d_imgz, d_img, N);
  ser_imgr = compute_ser(d_imgr, d_img, N);


  observe_final(N,lds,{{"img",d_img,N},{"mask",d_mask,N},{"imgz",d_imgz,N},{"imgb",d_imgb,N},{"imgl",d_imgl,N},{"imgs",d_imgs,N},{"w",d_w,3*N},{"p",d_p,3*N},{"q",d_q,6*N},{"r",d_r,N},{"lbar",d_lbar,N},{"sbar",d_sbar,N},{"wbar",d_wbar,3*N},{"lold",d_lold,N},{"sold",d_sold,N},{"wold",d_wold,3*N},{"tmp",d_tmp,12*N},{"imgr",d_imgr,N}},d_ls,d_ls2,d_lu,d_lvt,d_lsvt,devInfo,ldu,ldvt,ser_imgz,ser_imgr);

  // Free arrays and destroy handles
  cuTry(cudaFree(d_img));
  cuTry(cudaFree(d_imgb));
  cuTry(cudaFree(d_imgz));
  cuTry(cudaFree(d_imgr));
  cuTry(cudaFree(d_mask));

  cuTry(cudaFree(d_w));
  cuTry(cudaFree(d_p));
  cuTry(cudaFree(d_q));
  cuTry(cudaFree(d_r));
  cuTry(cudaFree(d_lbar));
  cuTry(cudaFree(d_sbar));
  cuTry(cudaFree(d_wbar));
  cuTry(cudaFree(d_lold));
  cuTry(cudaFree(d_sold));
  cuTry(cudaFree(d_wold));
  cuTry(cudaFree(d_tmp));

  cuTry(cudaFree(d_lu));
  cuTry(cudaFree(d_ls));
  cuTry(cudaFree(d_ls2));
  cuTry(cudaFree(d_lvt));
  cuTry(cudaFree(d_lsvt));
  cuTry(cudaFree(d_work));
  cuTry(cudaFree(d_rwork));
  cuTry(cudaFree(devInfo));

  cublasDestroy(cublas_handle);
  cusolverDnDestroy(cusolver_handle);
}
