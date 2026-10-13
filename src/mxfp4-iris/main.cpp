#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include <vector>
#include <cstring>
#include "common.h"
#include "verify.h"
#include <iris/iris.hpp>

static void check(int status) {
    if (status != IRIS_SUCCESS) { fprintf(stderr, "IRIS failure: %d\n", status); exit(2); }
}

template<typename T>
void simulate(int numel, int niters, int device) {
    printf("\nSimulate conversion between %s and FP4\n",
           std::is_same_v<T, __half> ? "FP16" : "BF16");
    const size_t bytes = size_t(numel) * sizeof(T);
    std::vector<T> src(numel), dst(numel);
    const float v[] = {0.0, 0.5, 1.0, 1.5, 2, 3, 4,
                      -0.0, -0.5, -1.0, -1.5, -2, -3, -4,
                      0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5,
                      0.1, 0.3, 0.4, 0.6, 0.9,
                      -0.1, -0.3, -0.4, -0.6, -0.9};
    // IRIS initialization reseeds libc RNG; restore the source's seed here.
    srand(123);
    for (int i = 0; i < numel / 32; ++i) {
        for (int j = i*32; j < (i+1)*32; ++j)
            src[j] = v[rand() % (sizeof(v) / sizeof(v[0]))];
        int r = rand() % 32;
        src[i*32 + r] = r < 16 ? 6 : -6;
    }
    iris::DMem input(src.data(), bytes), output(dst.data(), bytes);
    size_t global = numel, local = numel % 128 == 0 ? 128 : 64;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niters; ++i) {
        iris::Task task;
        void* params[] = {&input, &output};
        int modes[] = {iris_r, iris_w};
        check(task.kernel(std::is_same_v<T, __half> ? "qdq_half" : "qdq_bfloat",
                          1, nullptr, &global, &local, 2, params, modes));
        if (i == niters - 1) check(task.flush_out(output));
        check(task.submit(device, nullptr, true));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / double(niters) / 1.0e6;
    double size = 2.0 * sizeof(T) * numel / 1.0e9;
    printf("size(GB):%.2f, average time(sec):%f, Bandwidth (GB/sec):%f\n", size, time, size/time);
    verify(src.data(), dst.data(), numel);
    // Optional observational evidence: exact output words, no numerical changes.
    if (const char* prefix = getenv("HECBENCH_DUMP_PREFIX")) {
        std::string filename = std::string(prefix) + (std::is_same_v<T,__half> ? ".half.bin" : ".bf16.bin");
        FILE* f = fopen(filename.c_str(), "wb");
        if (!f || fwrite(dst.data(), sizeof(T), numel, f) != size_t(numel)) { perror("output dump"); exit(3); }
        fclose(f);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) { printf("Usage: %s <number of elements> <repeat>\n", argv[0]); return 1; }
    const int numel = atoi(argv[1]), niters = atoi(argv[2]);
    if (numel <= 0 || numel % 64 || niters <= 0) {
        fprintf(stderr, "Expected a positive multiple of 64 and positive repeat count\n"); return 1;
    }
    iris::Platform platform;
    check(platform.init(nullptr, nullptr, true));
    int device = -1, backend = -1, type = 0;
    for (int d = 0; d < platform.ndevs(); ++d) {
        check(iris_device_info(d, iris_backend, &backend, nullptr));
        if (backend == iris_hip) { device = d; break; }
    }
    if (device < 0) { fprintf(stderr, "No IRIS HIP device\n"); return 2; }
    char name[256] = {};
    check(iris_device_info(device, iris_name, name, nullptr));
    check(iris_device_info(device, iris_type, &type, nullptr));
    printf("IRIS backend=%d type=%d device=%s\n", backend, type, name);
    simulate<__half>(numel, niters, device);
    simulate<__hip_bfloat16>(numel, niters, device);
    check(platform.finalize());
    return 0;
}
