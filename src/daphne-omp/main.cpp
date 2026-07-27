// OpenMP CPU port of the daphne/points2image projection benchmark, mirroring
// the Triton reference at ../daphne-triton/main.py. Synthetic (LCG) point
// cloud; per-point extrinsic rotation, radial+tangential undistortion,
// intrinsic projection with a pt2>2.5 visibility filter.
//
// Uses #pragma omp parallel for (CPU multicore). HPC SDK's -mp=gpu codegen
// is broken on this box, so this port targets host-multithreaded OpenMP.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>

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
static constexpr int POINT_STEP = 8;

static inline void project_point(float p0, float p1, float p2,
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
    if (close) { px = xpix; py = ypix; pz = pt2 * 100.0f; valid = 1; }
    else       { px = 0.0f; py = 0.0f; pz = 0.0f; valid = 0; }
}

int main(int argc, char** argv) {
    int n_batches = 1;
    if (argc >= 3 && std::strcmp(argv[1], "-p") == 0) n_batches = std::atoi(argv[2]);

    const int n_points = 100000;
    printf("[note] synthetic %d points x %d batches\n", n_points, n_batches);

    float* cp   = (float*)std::calloc((size_t)n_points * POINT_STEP, sizeof(float));
    float* h_x  = (float*)std::malloc(sizeof(float) * n_points);
    float* h_y  = (float*)std::malloc(sizeof(float) * n_points);
    float* h_z  = (float*)std::malloc(sizeof(float) * n_points);
    int*   h_v  = (int*)  std::malloc(sizeof(int)   * n_points);
    float* ref_x = (float*)std::malloc(sizeof(float) * n_points);
    float* ref_y = (float*)std::malloc(sizeof(float) * n_points);
    float* ref_z = (float*)std::malloc(sizeof(float) * n_points);
    int*   ref_v = (int*)  std::malloc(sizeof(int)   * n_points);

    bool ok_all = true;
    double total_us = 0.0;

    for (int b = 0; b < n_batches; ++b) {
        uint64_t s = 20260721ULL + (uint64_t)(b + 1);
        for (int i = 0; i < n_points; ++i) {
            float u[4];
            for (int k = 0; k < 4; ++k) {
                s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                u[k] = (float)((s >> 33) & 0x7fffffff) / (float)0x7fffffff;
            }
            float* pt = cp + (size_t)i * POINT_STEP;
            pt[0] = (u[0] - 0.5f) * 2.0f;
            pt[1] = (u[1] - 0.5f) * 2.0f;
            pt[2] = u[2] * 10.0f + 15.0f;
            pt[4] = u[3];
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < n_points; ++i) {
            const float* pt = cp + (size_t)i * POINT_STEP;
            project_point(pt[0], pt[1], pt[2], h_x[i], h_y[i], h_z[i], h_v[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        int kept = 0;
        for (int i = 0; i < n_points; ++i) {
            const float* pt = cp + (size_t)i * POINT_STEP;
            project_point(pt[0], pt[1], pt[2], ref_x[i], ref_y[i], ref_z[i], ref_v[i]);
            if (ref_v[i]) kept++;
        }

        float max_err = 0.0f;
        int mismatch = 0;
        for (int i = 0; i < n_points; ++i) {
            if (h_v[i] != ref_v[i]) { mismatch++; continue; }
            if (ref_v[i]) {
                float e;
                e = fabsf(h_x[i] - ref_x[i]); if (e > max_err) max_err = e;
                e = fabsf(h_y[i] - ref_y[i]); if (e > max_err) max_err = e;
                e = fabsf(h_z[i] - ref_z[i]); if (e > max_err) max_err = e;
            }
        }
        bool batch_ok = (mismatch == 0) && (max_err <= 1e-3f);
        printf("[batch %d] kept=%d/%d max_err=%.3e valid_mismatch=%d\n",
               b, kept, n_points, max_err, mismatch);
        ok_all = ok_all && batch_ok;
    }

    printf("Average kernel execution time: %.1f (us)\n", total_us / n_batches);
    printf("%s\n", ok_all ? "PASS" : "FAIL");

    std::free(cp); std::free(h_x); std::free(h_y); std::free(h_z); std::free(h_v);
    std::free(ref_x); std::free(ref_y); std::free(ref_z); std::free(ref_v);
    return ok_all ? 0 : 1;
}
