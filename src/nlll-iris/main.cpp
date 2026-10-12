#include <iris/iris.hpp>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>

#include "../nlll-cuda/reference.h"

template <typename T>
float to_float(T v)
{
  return static_cast<float>(v);
}

template <>
float to_float<__half>(__half v)
{
  return __half2float(v);
}

template <typename T>
T from_float(float v)
{
  return static_cast<T>(v);
}

template <>
__half from_float<__half>(float v)
{
  return __float2half(v);
}

template <typename scalar_t>
const char* kernel_name(int threads)
{
  const bool is_half = std::is_same<scalar_t, __half>::value;
  switch (threads) {
    case 64: return is_half ? "nll_loss_half_64" : "nll_loss_float_64";
    case 128: return is_half ? "nll_loss_half_128" : "nll_loss_float_128";
    case 256: return is_half ? "nll_loss_half_256" : "nll_loss_float_256";
    case 512: return is_half ? "nll_loss_half_512" : "nll_loss_float_512";
    case 1024: return is_half ? "nll_loss_half_1024" : "nll_loss_float_1024";
    default: return nullptr;
  }
}

template <typename scalar_t, typename index_t>
int eval(const int64_t nframe,
         const int64_t kdim,
         const bool size_average,
         const int64_t ignore_index,
         const scalar_t r_output,
         const scalar_t r_total_weight,
         scalar_t* h_input,
         scalar_t* h_weights,
         index_t* h_target,
         const int repeat,
         const int threads)
{
  const int64_t input_size = nframe * kdim;
  const int64_t input_size_bytes = input_size * sizeof(scalar_t);
  const int64_t weights_size = nframe;
  const int64_t weights_size_bytes = weights_size * sizeof(scalar_t);
  const int64_t target_size = nframe;
  const int64_t target_size_bytes = target_size * sizeof(index_t);
  const int output_size_bytes = sizeof(scalar_t);

  scalar_t h_output = from_float<scalar_t>(0.0f);
  scalar_t h_total_weight = from_float<scalar_t>(0.0f);

  iris::DMem mem_output(&h_output, output_size_bytes);
  iris::DMem mem_total_weight(&h_total_weight, output_size_bytes);
  iris::DMem mem_input(h_input, input_size_bytes);
  iris::DMem mem_target(h_target, target_size_bytes);
  iris::DMem mem_weights(h_weights, weights_size_bytes);

  const size_t gws[1] = { static_cast<size_t>(threads) };
  const size_t lws[1] = { static_cast<size_t>(threads) };
  const char* name = kernel_name<scalar_t>(threads);
  if (name == nullptr) {
    std::fprintf(stderr, "unsupported thread block size: %d\n", threads);
    return 1;
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[] = { &mem_output, &mem_total_weight, &mem_input, &mem_target, &mem_weights,
                       const_cast<bool*>(&size_average), const_cast<int64_t*>(&nframe),
                       const_cast<int64_t*>(&kdim), const_cast<int64_t*>(&ignore_index) };
    int pinfo[] = { iris_w, iris_w, iris_r, iris_r, iris_r, sizeof(size_average),
                    sizeof(nframe), sizeof(kdim), sizeof(ignore_index) };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                9, params, pinfo);
    if (i + 1 == repeat) {
      task.flush_out(mem_output);
      task.flush_out(mem_total_weight);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::printf("\nThread block size: %d\n", threads);
  std::printf("Average execution time of nll loss forward kernel: %f (us)\n",
              (time * 1e-3f) / repeat);

  const float ho = to_float(h_output);
  const float ro = to_float(r_output);
  const float hw = to_float(h_total_weight);
  const float rw = to_float(r_total_weight);
  bool ok = true;
  if (std::fabs(ho - ro) > 1.f || std::fabs(hw - rw) > 1.f) {
    std::printf("%f %f %f %f\n", ho, ro, hw, rw);
    ok = false;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

template <typename scalar_t, typename index_t>
int driver(char** argv)
{
  const int64_t nframe = std::atol(argv[1]);
  const int64_t n_classes = std::atol(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const int64_t input_size = nframe * n_classes;
  const int64_t weights_size = nframe;
  const int64_t target_size = nframe;

  scalar_t* h_input = static_cast<scalar_t*>(std::malloc(input_size * sizeof(scalar_t)));
  scalar_t* h_weights = static_cast<scalar_t*>(std::malloc(weights_size * sizeof(scalar_t)));
  index_t* h_target = static_cast<index_t*>(std::malloc(target_size * sizeof(index_t)));

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> d1(-1.f, 1.f);
  std::uniform_int_distribution<index_t> d2(0, n_classes - 1);

  for (int64_t i = 0; i < input_size; i++) h_input[i] = from_float<scalar_t>(d1(g));
  for (int64_t i = 0; i < weights_size; i++) h_weights[i] = from_float<scalar_t>(d1(g));
  for (int64_t i = 0; i < target_size; i++) h_target[i] = d2(g);

  const bool size_average = true;
  const int64_t ignore_index = n_classes / 2;
  scalar_t r_output;
  scalar_t r_total_weight;

  reference<scalar_t, scalar_t, index_t>(&r_output, &r_total_weight, h_input, h_target,
                                         h_weights, size_average, nframe, n_classes,
                                         ignore_index);

  int errors = 0;
  const int thread_counts[] = { 64, 128, 256, 512, 1024 };
  for (int threads : thread_counts) {
    errors += eval<scalar_t, index_t>(nframe, n_classes, size_average, ignore_index,
                                      r_output, r_total_weight, h_input, h_weights,
                                      h_target, repeat, threads);
  }

  std::free(h_input);
  std::free(h_target);
  std::free(h_weights);
  return errors;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <minibatch size> <number of classes> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int errors = 0;
  std::printf("=========== Data type is FP32 ==========\n");
  errors += driver<float, int>(argv);

  std::printf("=========== Data type is FP16 ==========\n");
  errors += driver<__half, int>(argv);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}
