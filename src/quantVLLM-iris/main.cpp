#include <chrono>
#include <cmath>
#include <cstring>
#include "dmem_runtime.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hipcub/hipcub.hpp>
#include "utils.h"
#include "reference.h"
#include <string>
template<class T> std::string quant_name(const char* base) { return std::string(base) + (std::is_same_v<T,__half> ? "_f16" : std::is_same_v<T,hip_bfloat16> ? "_bf16" : "_f32"); }


template <typename scalar_t>
void static_scaled_int8_quant(int8_t *output,
                              const scalar_t *input,
                              int const hidden_size,
                              int const num_tokens,
                              float scale,
                              int32_t azp)
{
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  if (!azp)
    iris_launch(quant_name<scalar_t>("static_scaled_int8_quant_kernel").c_str(), grid, block, input, output, scale, hidden_size);
  else
    iris_launch(quant_name<scalar_t>("static_scaled_int8_azp_quant_kernel").c_str(), grid, block, input, output, scale, azp, hidden_size);
}

template <typename scalar_t>
void dynamic_scaled_int8_quant(int8_t *output,
                               const scalar_t *input,
                               int const hidden_size,
                               int const num_tokens,
                               float *scales,
                               int32_t *azp)
{
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  if (azp == nullptr) {
    iris_launch(quant_name<scalar_t>("dynamic_scaled_int8_quant_kernel").c_str(), grid, block, input, output, scales, hidden_size);
  } else {
    iris_launch(quant_name<scalar_t>("dynamic_scaled_int8_azp_quant_kernel").c_str(), grid, block, input, output, scales, azp, hidden_size);
  }
}

template <typename scalar_t>
void quant(int num_tokens, int hidden_size, int repeat) {
  
  scalar_t *d_input, *h_input;
  float *d_scale, *h_scale;
  int32_t *d_azp = nullptr, *h_azp;
  int8_t *d_output, *h_output, *h_output_r;


  size_t input_size_bytes = hidden_size * num_tokens * sizeof(scalar_t);
  size_t output_size_bytes = hidden_size * num_tokens * sizeof(int8_t);
  size_t scale_size_bytes = num_tokens * sizeof(float);
  size_t azp_size_bytes = num_tokens * sizeof(int32_t);

  srand(123);
  h_input = (scalar_t*) malloc (input_size_bytes);
  for (int i = 0; i < hidden_size * num_tokens; i++) {
    h_input[i] = rand() % 1000 - 300;
  }
  
  h_scale = (float*) malloc (scale_size_bytes);
  h_azp = (int32_t*) malloc (azp_size_bytes);
  h_output = (int8_t*) malloc (output_size_bytes);
  h_output_r = (int8_t*) malloc (output_size_bytes);

  iris_alloc(&d_input, input_size_bytes);
  iris_alloc(&d_output, output_size_bytes);
  iris_alloc(&d_scale, scale_size_bytes);

  iris_copy(d_input, h_input, input_size_bytes, iris_h2d);

  float scale = 0.1;
  int32_t error = 0;

  // static_scaled_int8_quant
  int32_t azp = 0;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    static_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, scale, azp);
  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of static_scaled_int8_quant kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  iris_copy(h_output, d_output, output_size_bytes, iris_d2h);
  static_scaled_int8_quant_reference(h_input, h_output_r, scale, num_tokens, hidden_size);
  error = memcmp(h_output, h_output_r, output_size_bytes);
  printf("Stage verification: %s\n", memcmp(h_output,h_output_r,output_size_bytes) ? "FAIL" : "PASS");

  // static_scaled_int8_quant_azp
  azp = 54;
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    static_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, scale, azp);
  iris_sync();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of static_scaled_int8_quant_azp kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  iris_copy(h_output, d_output, output_size_bytes, iris_d2h);
  static_scaled_int8_azp_quant_reference(h_input, h_output_r, scale, azp, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);
  printf("Stage verification: %s\n", memcmp(h_output,h_output_r,output_size_bytes) ? "FAIL" : "PASS");

  // dynamic_scaled_int8_quant
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    dynamic_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, d_scale, d_azp);
  iris_sync();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dynamic_scaled_int8_quant kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  iris_copy(h_output, d_output, output_size_bytes, iris_d2h);
  dynamic_scaled_int8_quant_reference(h_input, h_output_r, h_scale, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);
  printf("Stage verification: %s\n", memcmp(h_output,h_output_r,output_size_bytes) ? "FAIL" : "PASS");

  // dynamic_scaled_int8_quant_azp
  iris_alloc(&d_azp, azp_size_bytes);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    dynamic_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, d_scale, d_azp);
  iris_sync();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dynamic_scaled_int8_quant_azp kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  iris_copy(h_output, d_output, output_size_bytes, iris_d2h);
  dynamic_scaled_int8_azp_quant_reference(h_input, h_output_r, h_scale, h_azp, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);
  printf("Stage verification: %s\n", memcmp(h_output,h_output_r,output_size_bytes) ? "FAIL" : "PASS");

  printf("%s\n", error ? "FAIL" : "PASS");

  iris_free(d_input);
  iris_free(d_output);
  iris_free(d_scale);
  iris_free(d_azp);
  free(h_input);
  free(h_output);
  free(h_output_r);
  free(h_scale);
  free(h_azp);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of tokens> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_tokens = atoi(argv[1]);
  const int hidden_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Input type is FP16\n");
  quant<__half>(num_tokens, hidden_size, repeat);

  printf("Input type is BF16\n");
  quant<hip_bfloat16>(num_tokens, hidden_size, repeat);

  printf("Input type is FP32\n");
  quant<float>(num_tokens, hidden_size, repeat);
  iris_finish();
}
