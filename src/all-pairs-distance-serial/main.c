/* all-pairs-distance-serial: mirror of all-pairs-distance-julia/-rust. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INSTANCES  512
#define ATTRIBUTES 100

int main(int argc, char *argv[]) {
    int iters = (argc >= 2) ? atoi(argv[1]) : 5;
    unsigned char *data = malloc(INSTANCES * ATTRIBUTES);
    int *dist = malloc(INSTANCES * INSTANCES * sizeof(int));
    unsigned long long s = 20260721ULL;
    for (int i = 0; i < INSTANCES * ATTRIBUTES; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        data[i] = (unsigned char)((s >> 33) % 4);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < iters; r++) {
        for (int gx = 0; gx < INSTANCES; gx++) {
            for (int gy = 0; gy < INSTANCES; gy++) {
                int cnt = 0;
                for (int i = 0; i < ATTRIBUTES; i++)
                    if (data[i + ATTRIBUTES * gx] != data[i + ATTRIBUTES * gy]) cnt++;
                dist[INSTANCES * gx + gy] = cnt;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / iters;
    printf("Average kernel execution time %f (ms)\n", dt * 1e3);
    // Sanity: dist[i][i] should be 0
    int ok = 1;
    for (int i = 0; i < INSTANCES && ok; i++) if (dist[INSTANCES * i + i] != 0) ok = 0;
    printf("diff = %d\n%s\n", ok ? 0 : 1, ok ? "PASS" : "FAIL");
    return 0;
}
