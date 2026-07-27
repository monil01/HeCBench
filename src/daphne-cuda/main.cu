// CUDA port of the daphne/points2image projection benchmark, mirroring the
// Triton reference at ../daphne-triton/main.py. Synthetic (LCG-generated)
// point cloud; per-point extrinsic rotation, radial+tangential undistortion,
// intrinsic projection with a pt2>2.5 visibility filter.
//
// Rotation matrix and intrinsics are hardcoded from the Triton reference.
// Input is generated via the same LCG the adv-* ports use, so the numeric
// pipeline is exercised without the ~200MB DVC data bundle.
//
// PASS if GPU output matches host reference within 1e-3.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Hardcoded extrinsic (from Triton reference: torch.manual_seed(0); QR of randn(3,3))
static constexpr float R00 = -0.9111348390579224f;
static constexpr float R01 =  0.0751304179430008f;
static constexpr float R02 = -0.4052018225193024f;
static constexpr float R10 = -0.3360927104949951f;
static constexpr float R11 = -0.7044632434844971f;
static constexpr float R12 =  0.6251187324523926f;
static constexpr float R20 = -0.2384843230247498f;
static constexpr float R21 =  0.7057529091835022f;
static constexpr float R22 =  0.6671117544174194f;

static constexpr float T0 = 0.1f, T1 = -0.2f, T2 = 0.3f;

static constexpr float D0 = 0.03f, D1 = -0.15f, D2 = 0.001f, D3 = 0.001f, D4 = 0.05f;

static constexpr float FX = 1200.0f, CX = 400.0f, FY = 1200.0f, CY = 300.0f;
static constexpr int   W = 800, H = 600;

static constexpr int POINT_STEP = 8;

__device__ __host__ inline void project_point(
        float p0, float p1, float p2,
        float& px, float& py, float& pz, int& valid)
{
    float pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02;
    float pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12;
    float pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22;
    bool close = pt2 > 2.5f;
    float denom = close ? pt2 : 1.0f;
    float tmpx = pt0 / denom;
    float tmpy = pt1 / denom;
    float r2 = tmpx * tmpx + tmpy * tmpy;
    float tmpdist = 1.0f + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2;
    float ix = tmpx * tmpdist + 2.0f * D2 * tmpx * tmpy + D3 * (r2 + 2.0f * tmpx * tmpx);
    float iy = tmpy * tmpdist + D2 * (r2 + 2.0f * tmpy * tmpy) + 2.0f * D3 * tmpx * tmpy;
    float ux = FX * ix + CX;
    float uy = FY * iy + CY;
    float xpix = ux + 0.5f;
    float ypix = uy + 0.5f;
    if (close) {
        px = xpix; py = ypix; pz = pt2 * 100.0f;
        valid = 1;
    } else {
        px = 0.0f; py = 0.0f; pz = 0.0f;
        valid = 0;
    }
}

__global__ void project_kernel(const float* __restrict__ cp,
                               float* __restrict__ out_x,
                               float* __restrict__ out_y,
                               float* __restrict__ out_z,
                               int*   __restrict__ out_valid,
                               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* pt = cp + i * POINT_STEP;
    float px, py, pz; int v;
    project_point(pt[0], pt[1], pt[2], px, py, pz, v);
    out_x[i] = px;
    out_y[i] = py;
    out_z[i] = pz;
    out_valid[i] = v;
}

int main(int argc, char** argv) {
    int n_batches = 1;
    if (argc >= 3 && std::strcmp(argv[1], "-p") == 0) n_batches = std::atoi(argv[2]);

    const int n_points = 100000;
    printf("[note] synthetic %d points x %d batches\n", n_points, n_batches);

    float* h_cp = (float*)std::calloc((size_t)n_points * POINT_STEP, sizeof(float));
    float* h_x  = (float*)std::malloc(sizeof(float) * n_points);
    float* h_y  = (float*)std::malloc(sizeof(float) * n_points);
    float* h_z  = (float*)std::malloc(sizeof(float) * n_points);
    int*   h_v  = (int*)  std::malloc(sizeof(int)   * n_points);
    float* ref_x = (float*)std::malloc(sizeof(float) * n_points);
    float* ref_y = (float*)std::malloc(sizeof(float) * n_points);
    float* ref_z = (float*)std::malloc(sizeof(float) * n_points);
    int*   ref_v = (int*)  std::malloc(sizeof(int)   * n_points);

    float *d_cp, *d_x, *d_y, *d_z;
    int   *d_v;
    cudaMalloc(&d_cp, sizeof(float) * (size_t)n_points * POINT_STEP);
    cudaMalloc(&d_x,  sizeof(float) * n_points);
    cudaMalloc(&d_y,  sizeof(float) * n_points);
    cudaMalloc(&d_z,  sizeof(float) * n_points);
    cudaMalloc(&d_v,  sizeof(int)   * n_points);

    bool ok_all = true;
    double total_us = 0.0;

    for (int b = 0; b < n_batches; ++b) {
        // Deterministic LCG synthesis (per-batch seed = b+1) mirroring adv-* style.
        uint64_t s = 20260721ULL + (uint64_t)(b + 1);
        for (int i = 0; i < n_points; ++i) {
            float u[4];
            for (int k = 0; k < 4; ++k) {
                s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                u[k] = (float)((s >> 33) & 0x7fffffff) / (float)0x7fffffff;
            }
            float* pt = h_cp + (size_t)i * POINT_STEP;
            pt[0] = (u[0] - 0.5f) * 2.0f;   // x in [-1, 1]
            pt[1] = (u[1] - 0.5f) * 2.0f;   // y in [-1, 1]
            pt[2] = u[2] * 10.0f + 15.0f;   // z in [15, 25]
            pt[3] = 0.0f;
            pt[4] = u[3];                   // intensity
            pt[5] = 0.0f; pt[6] = 0.0f; pt[7] = 0.0f;
        }

        cudaMemcpy(d_cp, h_cp, sizeof(float) * (size_t)n_points * POINT_STEP, cudaMemcpyHostToDevice);

        int block = 256;
        int grid  = (n_points + block - 1) / block;

        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        project_kernel<<<grid, block>>>(d_cp, d_x, d_y, d_z, d_v, n_points);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        cudaMemcpy(h_x, d_x, sizeof(float) * n_points, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, sizeof(float) * n_points, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_z, d_z, sizeof(float) * n_points, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v, d_v, sizeof(int)   * n_points, cudaMemcpyDeviceToHost);

        // Host reference (same math, per-point)
        int kept = 0;
        for (int i = 0; i < n_points; ++i) {
            const float* pt = h_cp + (size_t)i * POINT_STEP;
            project_point(pt[0], pt[1], pt[2], ref_x[i], ref_y[i], ref_z[i], ref_v[i]);
            if (ref_v[i]) kept++;
        }

        float max_err = 0.0f;
        int mismatch = 0;
        for (int i = 0; i < n_points; ++i) {
            if (h_v[i] != ref_v[i]) { mismatch++; continue; }
            if (ref_v[i]) {
                float e = fabsf(h_x[i] - ref_x[i]);
                if (e > max_err) max_err = e;
                e = fabsf(h_y[i] - ref_y[i]);
                if (e > max_err) max_err = e;
                e = fabsf(h_z[i] - ref_z[i]);
                if (e > max_err) max_err = e;
            }
        }
        bool batch_ok = (mismatch == 0) && (max_err <= 1e-3f);
        printf("[batch %d] kept=%d/%d max_err=%.3e valid_mismatch=%d\n",
               b, kept, n_points, max_err, mismatch);
        ok_all = ok_all && batch_ok;
    }

    printf("Average kernel execution time: %.1f (us)\n", total_us / n_batches);
    printf("%s\n", ok_all ? "PASS" : "FAIL");

    cudaFree(d_cp); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_v);
    std::free(h_cp); std::free(h_x); std::free(h_y); std::free(h_z); std::free(h_v);
    std::free(ref_x); std::free(ref_y); std::free(ref_z); std::free(ref_v);
    return ok_all ? 0 : 1;
}
