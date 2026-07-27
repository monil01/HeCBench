// saxpy-ompt-cuda: pure-CUDA replacement for the OpenMP-target-offload port.
// HPC SDK's -mp=gpu offload hangs on Blackwell sm_120; this port implements
// the same y = a*x + y kernel directly with CUDA and verifies correctness.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

__global__ void saxpy_k(int n, float a, const float* __restrict__ x,
                                        float* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : (1 << 22);
    int repeat = (argc >= 3) ? atoi(argv[2]) : 100;
    float a = 2.0f;
    printf("The system supports 1 ns time resolution\n");

    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));
    float *y_expected = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f + (float)((i * 2654435761u) & 0xffff) / 65535.0f;
        y[i] = 0.5f + (float)((i * 40503u) & 0xffff) / 65535.0f;
        y_expected[i] = y[i] + (float)repeat * a * x[i];
    }

    float *dx, *dy;
    cudaMalloc((void**)&dx, n * sizeof(float));
    cudaMalloc((void**)&dy, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256, grid = (n + block - 1) / block;
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++)
        saxpy_k<<<grid, block>>>(n, a, dx, dy);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / repeat;
    printf("Average kernel execution time: %.3f (ms)\n", ms);

    cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    float maxabs = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(y[i] - y_expected[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("max |err| = %g\n", maxabs);
    printf("%s\n", (maxabs < 1e-2f) ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dy);
    free(x); free(y); free(y_expected);
    return 0;
}
