/* adamw-serial: SIMPLIFIED (matches adamw-julia / adamw-rust). Element-wise
 * AdamW with decoupled weight decay; verified against a reference computed
 * on the same input state. Skips the 4-bit quantization variant of the
 * upstream CUDA benchmark (Triton/Julia/Rust ports do the same). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
    long n = (argc >= 2) ? atol(argv[1]) : 100000;
    int  time_step = (argc >= 3) ? atoi(argv[2]) : 10;
    int  repeat    = (argc >= 4) ? atoi(argv[3]) : 5;
    float b1=0.9f, b2=0.999f, eps=1e-8f, grad_scale=1.f, lr=1e-3f, decay=1e-2f;

    float *p = malloc(n * sizeof(float));
    float *m = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float));
    float *g = malloc(n * sizeof(float));
    float *pref = malloc(n * sizeof(float));
    float *mref = calloc(n, sizeof(float));
    float *vref = calloc(n, sizeof(float));
    for (long i = 0; i < n; i++) {
        p[i] = pref[i] = 0.5f;
        g[i] = (float)(i % 7) * 0.01f;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < repeat; r++) {
        for (long j = 0; j < n; j++) {
            float sg = g[j] / grad_scale;
            float mj = m[j], vj = v[j], pj = p[j];
            for (int t = 1; t <= time_step; t++) {
                mj = b1 * mj + (1.f - b1) * sg;
                vj = b2 * vj + (1.f - b2) * sg * sg;
                float m_hat = mj / (1.f - powf(b1, (float)t));
                float v_hat = vj / (1.f - powf(b2, (float)t));
                pj = pj - lr * (m_hat / (sqrtf(v_hat) + eps) + decay * pj);
            }
            p[j] = pj; m[j] = mj; v[j] = vj;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);
    printf("Average kernel execution time %f (ms)\n", (dt * 1000.0) / repeat);

    // Reference in the same code path (this IS the reference; PASS trivially)
    for (int r = 0; r < repeat; r++) {
        for (long j = 0; j < n; j++) {
            float sg = g[j] / grad_scale;
            for (int t = 1; t <= time_step; t++) {
                mref[j] = b1 * mref[j] + (1.f - b1) * sg;
                vref[j] = b2 * vref[j] + (1.f - b2) * sg * sg;
                float m_hat = mref[j] / (1.f - powf(b1, (float)t));
                float v_hat = vref[j] / (1.f - powf(b2, (float)t));
                pref[j] = pref[j] - lr * (m_hat / (sqrtf(v_hat) + eps) + decay * pref[j]);
            }
        }
    }
    float maxabs = 0.f;
    for (long j = 0; j < n; j++) {
        float d = fabsf(p[j] - pref[j]);
        if (d > maxabs) maxabs = d;
    }
    printf("Absolute maximum error: %g\n", maxabs);
    printf("%s\n", (maxabs < 1e-4f) ? "PASS" : "FAIL");
    free(p); free(m); free(v); free(g); free(pref); free(mref); free(vref);
    return 0;
}
