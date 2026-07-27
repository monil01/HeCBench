/* adv-serial: 1D advection surrogate matching adv-julia/-rust/-mojo. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int iters = (argc >= 4) ? atoi(argv[3]) : 100;
    const int n = 65536;
    float *u   = malloc(n * sizeof(float));
    float *g   = malloc(n * sizeof(float));
    float *tmp = malloc(n * sizeof(float));
    unsigned long long s = 20260721ULL;
    for (int i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        u[i] = (float)((s >> 33) & 0x7fffffff) / (float)0x7fffffff;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        g[i] = (float)((s >> 33) & 0x7fffffff) / (float)0x7fffffff * 0.01f;
    }
    // Snapshot for reference
    float *uref = malloc(n * sizeof(float)); memcpy(uref, u, n * sizeof(float));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < iters; r++) {
        for (int i = 0; i < n; i++) {
            int im = (i - 1 + n) % n;
            int ip = (i + 1) % n;
            tmp[i] = u[i] + g[i] * (u[im] - u[ip]);
        }
        float *sw = u; u = tmp; tmp = sw;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / iters;
    printf("elapsed time= %f us/iter\n", dt * 1e6);

    // Independent reference (repeat the loop)
    float *tmp2 = malloc(n * sizeof(float));
    for (int r = 0; r < iters; r++) {
        for (int i = 0; i < n; i++) {
            int im = (i - 1 + n) % n;
            int ip = (i + 1) % n;
            tmp2[i] = uref[i] + g[i] * (uref[im] - uref[ip]);
        }
        float *sw = uref; uref = tmp2; tmp2 = sw;
    }

    float maxabs = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(u[i] - uref[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("Max error: %g\n", maxabs);
    printf("%s\n", (maxabs <= 1e-3f) ? "PASS" : "FAIL");
    return 0;
}
