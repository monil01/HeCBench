#include "muon_common.h"
#include "dmem.hpp"
static inline void rgemm(GemmOp opA,GemmOp opB,int M,int N,int K,const bf16* A,const bf16* B,bf16* C) {
    port::launch("muon_gemm",dim3((N+15)/16,(M+15)/16),dim3(16,16),A,B,C,M,N,K,int(opA==GEMM_T),int(opB==GEMM_T));
}
int main(int argc, char* argv[])
{
  port::Session session(&argc, &argv);
  if (argc != 5) {
    printf("Usage: %s <rows> <cols> <repeat> <verify>\n", argv[0]);
    printf("  rows = out_features, cols = in_features of a 2D weight matrix\n");
    return 1;
  }

  const int P = atoi(argv[1]);
  const int Q = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int verify = atoi(argv[4]);

  const int N = P * Q;
  const int mm_ = std::min(P, Q); // rows of the Newton-Schulz working matrix
  const int nn_ = std::max(P, Q); // cols of the Newton-Schulz working matrix

  const size_t size_bytes = (size_t)N * sizeof(float);

  float *mom = (float*) malloc (size_bytes);
  float *g   = (float*) malloc (size_bytes);
  float *p   = (float*) malloc (size_bytes);
  float *r   = (float*) malloc (size_bytes);   // reference param
  float *rm  = (float*) malloc (size_bytes);   // reference momentum

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < N; i++) {
    rm[i] = mom[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  // Muon hyper-parameters
  const float beta = 0.95f;
  const float lr   = 0.02f;
  const float wd   = 0.f;     // weight decay (Muon default)
  const float scale = sqrtf(fmaxf(1.f, (float)P / (float)Q));

  float *d_p, *d_m, *d_g, *d_ss;
  bf16  *d_X, *d_A, *d_A2, *d_B, *d_BX;
  CHECK_HIP(port::allocate((void**)&d_p, size_bytes));
  CHECK_HIP(port::allocate((void**)&d_m, size_bytes));
  CHECK_HIP(port::allocate((void**)&d_g, size_bytes));
  CHECK_HIP(port::allocate((void**)&d_ss, sizeof(float)));
  CHECK_HIP(port::allocate((void**)&d_X,  sizeof(bf16) * N));          // mm_ x nn_ = N
  CHECK_HIP(port::allocate((void**)&d_A,  sizeof(bf16) * mm_ * mm_));
  CHECK_HIP(port::allocate((void**)&d_A2, sizeof(bf16) * mm_ * mm_));
  CHECK_HIP(port::allocate((void**)&d_B,  sizeof(bf16) * mm_ * mm_));
  CHECK_HIP(port::allocate((void**)&d_BX, sizeof(bf16) * N));          // mm_ x nn_ = N

  CHECK_HIP(port::copy(d_m, mom, size_bytes, port::h2d));
  CHECK_HIP(port::copy(d_g, g, size_bytes, port::h2d));
  CHECK_HIP(port::copy(d_p, p, size_bytes, port::h2d));



  const int tpb = 256;
  // vectorized (4 elem/thread) launch grids for the elementwise kernels
  const dim3 grid_N4 (((N + 3) / 4 + tpb - 1) / tpb);
  const dim3 grid_mm4 (((mm_ * mm_ + 3) / 4 + tpb - 1) / tpb);
  const dim3 block1 (tpb);
  const dim3 tblock2 (T_TILE, T_BROWS);   // tiled transpose block

  // launch grids for the fused transposing kernels (P>Q path)
  const dim3 gt_fwd ((Q + T_TILE - 1) / T_TILE, (P + T_TILE - 1) / T_TILE);
  const dim3 gt_bwd ((P + T_TILE - 1) / T_TILE, (Q + T_TILE - 1) / T_TILE);

  CHECK_HIP(iris_synchronize());
  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < repeat; it++) {
    // momentum update + bf16 conversion; for P>Q it is fused with the
    // transpose so X is produced directly as the (Q x P) working matrix.
    if (P > Q)
      port::launch("momentum_transpose",gt_fwd, tblock2,d_m, d_g, d_X, beta, P, Q);
    else
      port::launch("momentum_update",grid_N4, block1,d_m, d_g, d_X, beta, N);

    // normalize by Frobenius norm
    CHECK_HIP(port::zero(d_ss, 0, sizeof(float)));
    port::launch("sumsq_kernel",256, block1,d_X, d_ss, N);
    port::launch("normalize",grid_N4, block1,d_X, d_ss, N);

    // Newton-Schulz iterations (NS_A, NS_B, and NS_C defined in reference.h)
    for (int s = 0; s < NS_STEPS; s++) {
      // A=X * X^T
      rgemm(GEMM_N, GEMM_T, mm_, mm_, nn_, d_X, d_X, d_A);
      // A2=A * A
      rgemm(GEMM_N, GEMM_N, mm_, mm_, mm_, d_A, d_A, d_A2);
      // B = b*A + c*A2
      port::launch("combineB",grid_mm4, block1,d_A, d_A2, d_B, NS_B, NS_C, mm_ * mm_);
      // BX=B * X
      rgemm(GEMM_N, GEMM_N, mm_, nn_, mm_, d_B, d_X, d_BX);
      // X = a*X + BX
      port::launch("updateX",grid_N4, block1,d_X, d_BX, NS_A, N);
    }

    // spectral scaling + weight update; for P>Q it is fused with the
    // transpose so the (Q x P) working matrix updates p (P x Q) in place.
    if (P > Q)
      port::launch("transpose_param_update",gt_bwd, tblock2,d_p, d_X, lr, wd, scale, P, Q);
    else
      port::launch("param_update",grid_N4, block1,d_p, d_X, lr, wd, scale, N);
  }

  CHECK_HIP(iris_synchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average step execution time %f (ms)\n", time * 1e-6f / repeat);

  CHECK_HIP(port::copy(p, d_p, size_bytes, port::d2h));


  CHECK_HIP(port::release(d_p));
  CHECK_HIP(port::release(d_m));
  CHECK_HIP(port::release(d_g));
  CHECK_HIP(port::release(d_ss));
  CHECK_HIP(port::release(d_X));
  CHECK_HIP(port::release(d_A));
  CHECK_HIP(port::release(d_A2));
  CHECK_HIP(port::release(d_B));
  CHECK_HIP(port::release(d_BX));

  if (verify) {
    reference(repeat, r, rm, g, P, Q, beta, lr, wd);

    bool ok = true;
    for (int i = 0; i < N; i++) {
      if (!std::isfinite(p[i]) || fabsf(r[i] - p[i]) > 1e-2f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(mom);
  free(g);
  free(p);
  free(r);
  free(rm);
  return 0;
}
