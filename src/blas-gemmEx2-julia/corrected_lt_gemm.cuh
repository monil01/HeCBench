template <typename T, typename S, typename CT>
bool study_lt_gemm(
    const int m, const int n, const int k,
    T *A, T *B, S *C,
    int lda, int ldb, int ldc,
    const CT *alpha, const CT *beta, int compute32F_mode)
{
  bool status = true;
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatmulPreference_t preference;
  cublasLtMatrixLayout_t ALayout;
  cublasLtMatrixLayout_t BLayout;
  cublasLtMatrixLayout_t CLayout;
  cublasLtMatmulHeuristicResult_t heuristic;

  // create the operation descriptor
  //cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasOperation_t opNoTranspose = CUBLAS_OP_N;

  cudaDataType_t AType, CType, SType;
  cublasComputeType_t ComputeType;
  if (std::is_same<T, double>::value) {
    AType = SType = CType = CUDA_R_64F;
    ComputeType = CUBLAS_COMPUTE_64F;
  } else if (std::is_same<T, float>::value) {
    AType = SType = CType = CUDA_R_32F;
    switch (compute32F_mode) {
      case 0: ComputeType = CUBLAS_COMPUTE_32F;
      case 1: ComputeType = CUBLAS_COMPUTE_32F_FAST_16F;
      case 2: ComputeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      case 3: ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      default: ComputeType = CUBLAS_COMPUTE_32F;
    }
  } else if (std::is_same<T, __half>::value) {
    AType = CType = CUDA_R_16F;
    if (std::is_same<CT, __half>::value) {
      SType = CUDA_R_16F;
      ComputeType = CUBLAS_COMPUTE_16F;
    }
    else {
      SType = CUDA_R_32F;
      ComputeType = CUBLAS_COMPUTE_32F;
    }
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    AType = CType = CUDA_R_16BF;
    SType = CUDA_R_32F;
    ComputeType = CUBLAS_COMPUTE_32F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = CUDA_R_8I;
    CType = SType = CUDA_R_32I;
    ComputeType = CUBLAS_COMPUTE_32I;
  } else {
    printf("Not supported data type.");
    return false;
  }
  
  // compute type and scale type
  cublasCheck(cublasLtMatmulDescCreate(&operationDesc, ComputeType, SType));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
              CUBLASLT_MATMUL_DESC_TRANSA, &opNoTranspose, sizeof(opNoTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(
              operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));

  // define matrix layouts
  cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, AType, m, k, lda));
  cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, AType, k, n, ldb));
  cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CType, m, n, ldc));

  // create a preference handle with specified max workspace
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // find a suitable algorithm
  int returnedResults = 0;
  cublasCheck(cublasLtMatmulAlgoGetHeuristic(handle, operationDesc,
      ALayout, BLayout, CLayout, CLayout,
      preference, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
      printf("No cuBLASLt algorithm found.\n");
      status = false;
      goto cleanups;
  }

  // call the matmul
  cublasCheck(cublasLtMatmul(handle, operationDesc,
      alpha, A, ALayout, B, BLayout, beta,
      C, CLayout, C, CLayout, &heuristic.algo,
      cublaslt_workspace, cublaslt_workspace_size, 0));

  // wait for matmul
  cudaDeviceSynchronize();

  cleanups:
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));

  return status;
}

