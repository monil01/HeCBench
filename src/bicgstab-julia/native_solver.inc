// Source-exact arithmetic; one iteration-counter assignment added.
double study_native_solver(
                 int                  verbose,
                 cublasHandle_t       cublasHandle,
                 cusparseHandle_t     cusparseHandle,
                 int                  m,
                 cusparseSpMatDescr_t matA,
                 cusparseSpMatDescr_t matM_lower,
                 cusparseSpMatDescr_t matM_upper,
                 Vec                  d_B,
                 Vec                  d_X,
                 Vec                  d_R0,
                 Vec                  d_R,
                 Vec                  d_P,
                 Vec                  d_P_aux,
                 Vec                  d_S,
                 Vec                  d_S_aux,
                 Vec                  d_V,
                 Vec                  d_T,
                 Vec                  d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // Create opaque data structures that holds analysis data between calls
    double              coeff_tmp = 1.0; // defined named analysis coefficient
    size_t              bufferSizeL, bufferSizeU;
    void*               d_bufferL, *d_bufferU;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
    CHECK_CUDA( cudaMalloc(&d_bufferL, bufferSizeL) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, CUDA_R_64F,
                        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )

    // Calculate UPPER buffersize
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrU) )
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        &bufferSizeU) )
    CHECK_CUDA( cudaMalloc(&d_bufferU, bufferSizeU) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        d_bufferU) )
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA( cudaMemcpy(d_R0.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //    (b) compute R = -A * X0 + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R0.vec,
                                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1,
                             &delta) )
    delta_prev = delta;
    // R = R0
    CHECK_CUDA( cudaMemcpy(d_R.ptr, d_R0.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R0.ptr, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    if (verbose) printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual

    for (int i = 1; i <= maxIterations; i++) {
        study_iterations = i; // named measurement instrumentation only
        if (verbose) printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        CHECK_CUDA( cudaMemcpy(d_P.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        if (i > 1) {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_R.ptr, 1,
                                     &delta) )
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_V.ptr, 1,
                                      d_P.ptr, 1) )
            //    (b) P = beta * P
            CHECK_CUBLAS( cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1) )
            //    (c) P = R + P
            CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &one, d_R.ptr, 1,
                                      d_P.ptr, 1) )
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CHECK_CUDA( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) )
        CHECK_CUDA( cudaMemset(d_P_aux.ptr, 0x0, m * sizeof(double)) )
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_P.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_P_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) )
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P_aux.vec, &zero, d_V.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) denominator = R'0 * V
        double denominator;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R0.ptr, 1, d_V.ptr, 1,
                                 &denominator) )
        alpha = delta / denominator;
        if (verbose) PRINT_INFO(delta)
        if (verbose) PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &alpha, d_P_aux.ptr, 1,
                                  d_X.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_CUDA( cudaMemcpy(d_S.ptr, d_R.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_alpha, d_V.ptr, 1,
                                  d_S.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_S.ptr, 1, &nrm_S) )
        if (verbose) PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double));
        cudaMemset(d_S_aux.ptr, 0x0, m * sizeof(double));
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_S.vec, d_tmp.vec,
                                           CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_S_aux.vec, CUDA_R_64F,
                                           CUSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU))
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S_aux.vec, &zero, d_T.vec,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_S.ptr, 1,
                                 &omega_num) )
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_T.ptr, 1,
                                 &omega_den) )
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        if (verbose) PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &omega, d_S_aux.ptr, 1,
                                  d_X.ptr, 1) )
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_CUDA( cudaMemcpy(d_R.ptr, d_S.ptr, m * sizeof(double),
                               cudaMemcpyDeviceToDevice) )
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_omega, d_T.ptr, 1,
                                  d_R.ptr, 1) )
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
        if (verbose) PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    if (verbose) printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                           cudaMemcpyDeviceToDevice) )
    // R = -A * X + R
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // check ||R||
    CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrL) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrU) )
    CHECK_CUDA( cudaFree(d_bufferL) )
    CHECK_CUDA( cudaFree(d_bufferU) )
    return nrm_R;
}

