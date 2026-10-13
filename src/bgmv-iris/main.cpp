#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <hip/hip_fp16.h>
#include <iris/iris.hpp>
#include "config.h"
#include "reference.h"

#define IRIS_CHECK(expr) do { int status = (expr); if (status != IRIS_SUCCESS) { \
    fprintf(stderr, "IRIS error %s:%d: %s => %d\n", __FILE__, __LINE__, #expr, status); \
    exit(1); } } while (0)

// IRIS gws counts work items: (T * block_size, output_columns) preserves
// the original CUDA grid (T, output_columns) and block (block_size, 1).
static void launch(iris::DMem& out, iris::DMem& in, iris::DMem& wt,
                   iris::DMem& ids, Config& cfg, int block_size, int device,
                   bool flush) {
    bool shrink = !strcmp(cfg.op, "bgmv_shrink");
    char symbol[64];
    snprintf(symbol, sizeof(symbol), shrink ? "bgmv_shrink%s_%d" : "bgmv_expand%s_%d",
             shrink && cfg.vectorize ? "_vec" : "", block_size);
    void* last = shrink ? static_cast<void*>(&cfg.scaling) : static_cast<void*>(&cfg.add_to_output);
    void* params[] = {&out, &in, &wt, &ids, &cfg.num_tokens, &cfg.hidden_size, &cfg.lora_rank, last};
    int info[] = {iris_rw, iris_r, iris_r, iris_r, sizeof(int), sizeof(int), sizeof(int),
                  shrink ? int(sizeof(float)) : int(sizeof(bool))};
    size_t gws[] = {size_t(cfg.num_tokens) * block_size,
                    size_t(shrink ? cfg.lora_rank : cfg.hidden_size)};
    size_t lws[] = {size_t(block_size), 1};
    iris::Task task;
    IRIS_CHECK(task.kernel(symbol, 2, nullptr, gws, lws, 8, params, info));
    if (flush) IRIS_CHECK(task.flush_out(out));
    IRIS_CHECK(task.submit(device, nullptr, true));
}

template<typename Input, typename Output>
static bool benchmark(Config& cfg, std::vector<Input>& input, std::vector<__half>& weights,
                      std::vector<int>& ids, std::vector<float>& reference, int device) {
    std::vector<Output> output(reference.size());
    iris::DMem mem_in(input.data(), input.size() * sizeof(Input));
    iris::DMem mem_wt(weights.data(), weights.size() * sizeof(__half));
    iris::DMem mem_ids(ids.data(), ids.size() * sizeof(int));
    iris::DMem mem_out(output.data(), output.size() * sizeof(Output));
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    bool pass = true;
    auto zero = [&]() {
        memset(output.data(), 0, output.size() * sizeof(Output));
        // Host zeros are authoritative; update invalidates all device copies.
        IRIS_CHECK(mem_out.update(output.data()));
    };
    for (int block_size : block_sizes) {
        zero();
        launch(mem_out, mem_in, mem_wt, mem_ids, cfg, block_size, device, true);
        float max_err = 0.f;
        for (size_t i = 0; i < output.size(); ++i) {
            float value = static_cast<float>(output[i]);
            float err = fabsf(value - reference[i]);
            if (!std::isfinite(value) || !std::isfinite(err)) err = INFINITY;
            max_err = fmaxf(max_err, err);
        }
        bool ok = max_err < 0.1f;
        pass &= ok;
        printf("block_size %4d | correctness check max_err = %e  => %s\n",
               block_size, max_err, ok ? "PASS" : "FAIL");
    }
    for (int block_size : block_sizes) {
        zero();
        IRIS_CHECK(iris_synchronize());
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.repeat; ++i)
            launch(mem_out, mem_in, mem_wt, mem_ids, cfg, block_size, device, false);
        IRIS_CHECK(iris_synchronize());
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        size_t bytes_read = input.size() * sizeof(Input) + weights.size() * sizeof(__half) + ids.size() * sizeof(int);
        size_t bytes_write = output.size() * sizeof(Output);
        double bw = double(bytes_read + bytes_write) * cfg.repeat / time;
        if (!strcmp(cfg.op, "bgmv_shrink"))
            printf("bgmv_shrink  | block_size %4d | avg latency: %.3f us  (over %d iters) | bandwidth (approx): %.1f GB/s\n",
                   block_size, time * 1e-3f / cfg.repeat, cfg.repeat, bw);
        else
            printf("bgmv_expand | block_size %4d | add_to_output=%s  avg latency: %.3f us | bandwidth (approx): %.1f GB/s\n",
                   block_size, cfg.add_to_output ? "true" : "false", time * 1e-3f / cfg.repeat, bw);
    }
    // Make the final timed output host-visible, including accumulation runs.
    iris::Task flush;
    IRIS_CHECK(flush.flush_out(mem_out));
    IRIS_CHECK(flush.submit(device, nullptr, true));
    return pass;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    bool shrink = !strcmp(cfg.op, "bgmv_shrink");
    if (!shrink && strcmp(cfg.op, "bgmv_expand")) {
        fprintf(stderr, "Unknown op '%s'\n", cfg.op);
        usage(argv[0]);
        return 1;
    }
    if (cfg.num_tokens <= 0 || cfg.hidden_size <= 0 || cfg.lora_rank <= 0 || cfg.num_loras <= 0 || cfg.repeat <= 0) {
        fprintf(stderr, "Dimensions and repeat must be positive\n");
        return 1;
    }
    const int T = cfg.num_tokens, H = cfg.hidden_size, R = cfg.lora_rank, NL = cfg.num_loras;
    printf("\n=== BGMV Benchmark ===\n");
    printf("  op          : %s\n  num_tokens  : %d\n  hidden_size : %d\n  lora_rank   : %d\n  num_loras   : %d\n  repeat      : %d\n  add_to_output  : %s\n",
           cfg.op, T, H, R, NL, cfg.repeat, cfg.add_to_output ? "true" : "false");
    printf("------------------------------------------------------------------\n");
    iris::Platform platform;
    IRIS_CHECK(platform.init(&argc, &argv, 1));
    int count = 0, device = -1;
    IRIS_CHECK(iris_device_count(&count));
    for (int i = 0; i < count; ++i) {
        int backend = 0, type = 0;
        char name[256] = {};
        IRIS_CHECK(iris_device_info(i, iris_backend, &backend, nullptr));
        IRIS_CHECK(iris_device_info(i, iris_type, &type, nullptr));
        IRIS_CHECK(iris_device_info(i, iris_name, name, nullptr));
        printf("IRIS device %d: name=%s backend=%d type=%d\n", i, name, backend, type);
        if (device < 0 && backend == iris_hip && (type & iris_amd)) device = i;
    }
    if (device < 0) {
        fprintf(stderr, "blocked: no-amd-gpu HIP device in IRIS\n");
        platform.finalize();
        return 2;
    }
    printf("Selected IRIS HIP AMD device %d\n", device);
    std::mt19937 rng(19937);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto random = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
    auto half = [](const std::vector<float>& f) {
        std::vector<__half> h(f.size());
        for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
        return h;
    };
    std::vector<int> ids(T);
    std::uniform_int_distribution<int> lora_dist(0, NL - 1);
    for (auto& x : ids) x = lora_dist(rng);
    auto input = random(size_t(T) * (shrink ? H : R));
    auto weights = random(size_t(NL) * R * H);
    auto weights_half = half(weights);
    std::vector<float> reference(size_t(T) * (shrink ? R : H), 0.f);
    bool pass;
    if (shrink) {
        auto input_half = half(input);
        ref_bgmv_shrink_cpu(reference.data(), input.data(), weights.data(), ids.data(), T, H, R, cfg.scaling);
        pass = benchmark<__half, float>(cfg, input_half, weights_half, ids, reference, device);
    } else {
        // Preserve the CUDA call: scaling converts to the reference's bool.
        // The expand device kernel itself does not apply scaling.
        ref_bgmv_expand_cpu(reference.data(), input.data(), weights.data(), ids.data(), T, H, R, cfg.scaling);
        pass = benchmark<float, __half>(cfg, input, weights_half, ids, reference, device);
    }
    printf("------------------------------------------------------------------\n");
    IRIS_CHECK(platform.finalize());
    return pass ? 0 : 1;
}
