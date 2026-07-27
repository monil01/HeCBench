/* saxpy-ompt-omp: OpenMP port of the saxpy-ompt HeCBench benchmark.
 *
 * The upstream benchmark measures y = a*x + y across several implementations
 * (OpenMP-target, cuBLAS, ...). This port parallelizes the saxpy loop with a
 * host-side `#pragma omp parallel for` and compares against an independently
 * seeded reference to PASS. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : (1 << 22);
    float a = 2.0f;
    float *x = (float*)malloc(n * sizeof(float));
    float *y = (float*)malloc(n * sizeof(float));
    float *yref = (float*)malloc(n * sizeof(float));
    if (!x || !y || !yref) { fprintf(stderr, "alloc fail\n"); return 1; }

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f + (float)((i * 2654435761u) & 0xffff) / 65535.0f;
        y[i] = 0.5f + (float)((i * 40503u) & 0xffff) / 65535.0f;
        yref[i] = a * x[i] + y[i];
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double mbps = (2.0 * n * sizeof(float)) / (dt * 1e6);
    printf("saxpy n=%d  elapsed=%.6f s  bandwidth=%.1f MB/s\n", n, dt, mbps);

    float maxabs = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(y[i] - yref[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("max |err| = %g\n", maxabs);
    printf("%s\n", (maxabs < 1e-5f) ? "PASS" : "FAIL");

    free(x); free(y); free(yref);
    return 0;
}
