/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#include "utils.h"

#define CHECK_HIP(func)                                                       \
{                                                                              \
    hipError_t status = (func);                                               \
    if (status != hipSuccess) {                                               \
        printf("HIP API failed at line %d with error: %s (%d)\n",             \
               __LINE__, hipGetErrorString(status), status);                  \
        abort();                                                   \
    }                                                                          \
}

#define CHECK_HIPBLAS(func)                                                     \
{                                                                              \
    hipblasStatus_t status = (func);                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                     \
        printf("HIPBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        abort();                                                   \
    }                                                                          \
}

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            abort();                                                           \
        }                                                                                 \
    }


#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

typedef struct VecStruct {
    hipsparseDnVecDescr_t vec;
    double*              ptr;
} Vec;


double gpu_BiCGStab(
                 int                  verbose,
                 hipblasHandle_t       hipblasHandle,
                 hipsparseHandle_t     hipsparseHandle,
                 int                  m,
                 hipsparseSpMatDescr_t matA,
                 hipsparseSpMatDescr_t matM_lower,
                 hipsparseSpMatDescr_t matM_upper,
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
                 void* d_bufferL, void* d_bufferU, const double* zeros,
                 int                  maxIterations,
                 double               tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    //--------------------------------------------------------------------------
    // Create opaque data structures that holds analysis data between calls
    double              coeff_tmp = 1.0;
    size_t              bufferSizeL, bufferSizeU;
    hipsparseSpSVDescr_t spsvDescrL, spsvDescrU;
    CHECK_HIPSPARSE( hipsparseSpSV_createDescr(&spsvDescrL) )
    CHECK_HIPSPARSE( hipsparseSpSV_bufferSize(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
    CHECK_HIPSPARSE( hipsparseSpSV_analysis(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_lower, d_P.vec, d_tmp.vec, HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )

    // Calculate UPPER buffersize
    CHECK_HIPSPARSE( hipsparseSpSV_createDescr(&spsvDescrU) )
    CHECK_HIPSPARSE( hipsparseSpSV_bufferSize(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        &bufferSizeU) )
    CHECK_HIPSPARSE( hipsparseSpSV_analysis(
                        hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &coeff_tmp, matM_upper, d_tmp.vec, d_P_aux.vec,
                        HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescrU,
                        d_bufferU) )
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_B.ptr, 1, d_R0.ptr, 1) )
    //    (b) compute R = -A * X0 + R
    CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, matA, d_X.vec, &one, d_R0.vec,
                                 HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                 d_bufferMV) )
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_R0.ptr, 1,
                             &delta) )
    delta_prev = delta;
    // R = R0
    CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_R0.ptr, 1, d_R.ptr, 1) )
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R0.ptr, 1, &nrm_R) )
    double threshold = tolerance * nrm_R;
    if (verbose) printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual

    for (int i = 1; i <= maxIterations; i++) {
        if (verbose) printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_R.ptr, 1, d_P.ptr, 1) )
        if (i > 1) {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_R.ptr, 1,
                                     &delta) )
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_omega, d_V.ptr, 1,
                                      d_P.ptr, 1) )
            //    (b) P = beta * P
            CHECK_HIPBLAS( hipblasDscal(hipblasHandle, m, &beta, d_P.ptr, 1) )
            //    (c) P = R + P
            CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &one, d_R.ptr, 1,
                                      d_P.ptr, 1) )
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, zeros, 1, d_tmp.ptr, 1) )
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, zeros, 1, d_P_aux.ptr, 1) )
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_P.vec, d_tmp.vec,
                                           HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_P_aux.vec, HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) )
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_P_aux.vec, &zero, d_V.vec,
                                     HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) denominator = R'0 * V
        double denominator;
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_R0.ptr, 1, d_V.ptr, 1,
                                 &denominator) )
        alpha = delta / denominator;
        if (verbose) PRINT_INFO(delta)
        if (verbose) PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &alpha, d_P_aux.ptr, 1,
                                  d_X.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_R.ptr, 1, d_S.ptr, 1) )
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_alpha, d_V.ptr, 1,
                                  d_S.ptr, 1) )
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        double nrm_S;
        CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_S.ptr, 1, &nrm_S) )
        if (verbose) PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, zeros, 1, d_tmp.ptr, 1) )
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, zeros, 1, d_S_aux.ptr, 1) )
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_lower, d_S.vec, d_tmp.vec,
                                           HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrL) )
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        CHECK_HIPSPARSE( hipsparseSpSV_solve(hipsparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, matM_upper, d_tmp.vec,
                                           d_S_aux.vec, HIP_R_64F,
                                           HIPSPARSE_SPSV_ALG_DEFAULT,
                                           spsvDescrU) )
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                     matA, d_S_aux.vec, &zero, d_T.vec,
                                     HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                                     d_bufferMV) )
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_T.ptr, 1, d_S.ptr, 1,
                                 &omega_num) )
        //    (c) omega_den = (A * S_aux, A * S_aux)
        CHECK_HIPBLAS( hipblasDdot(hipblasHandle, m, d_T.ptr, 1, d_T.ptr, 1,
                                 &omega_den) )
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        if (verbose) PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &omega, d_S_aux.ptr, 1,
                                  d_X.ptr, 1) )
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_S.ptr, 1, d_R.ptr, 1) )
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        CHECK_HIPBLAS( hipblasDaxpy(hipblasHandle, m, &minus_omega, d_T.ptr, 1,
                                  d_R.ptr, 1) )
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R.ptr, 1, &nrm_R) )
        if (verbose) PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    if (verbose) printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_HIPBLAS( hipblasDcopy(hipblasHandle, m, d_B.ptr, 1, d_R.ptr, 1) )
    // R = -A * X + R
    CHECK_HIPSPARSE( hipsparseSpMV(hipsparseHandle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                 matA, d_X.vec, &one, d_R.vec, HIP_R_64F,
                                 HIPSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // check ||R||
    CHECK_HIPBLAS( hipblasDnrm2(hipblasHandle, m, d_R.ptr, 1, &nrm_R) )
    //--------------------------------------------------------------------------
    CHECK_HIPSPARSE( hipsparseSpSV_destroyDescr(spsvDescrL) )
    CHECK_HIPSPARSE( hipsparseSpSV_destroyDescr(spsvDescrU) )
    return nrm_R;
}

//==============================================================================
//==============================================================================
