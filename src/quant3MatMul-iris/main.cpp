// Modified for IRIS DMem backend symbol/host conversion, 2026-10-13. See Apache-2.0 LICENSE.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "iris_support.h"
#include <cstdint>
using half=uint16_t;

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  if(repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();
  bool all_ok=true;

  // Reference https://github.com/IST-DASLab/gptq/blob/main/test_kernel.py
  // Benchmarking OPT-175B FC2 matvec 
  const int m = 12288 * 4;
  const int n = 12288;
  const int mat_height = (m / 1024 * 96);  // 4608
  const int mat_width = n;
  const int mat_size = mat_height * mat_width;
  const int vec_size = 1 * m;
  const int mul_size = 1 * n;

  using scalar_t = float;

  // initialize the vectors 
  scalar_t *h_input = (scalar_t*) malloc (sizeof(scalar_t) * vec_size);
  half *h_input_half = (half*) malloc (sizeof(half) * vec_size);
  for (int i = 0; i < vec_size; i++) {
    h_input[i]=1.f;
    h_input_half[i]=0x3c00; // IEEE binary16 representation of 1.0, as native half input.
  }

  // initialize the weights 
  int *h_weight = (int*) malloc (sizeof(int) * mat_size);
  int p1 = 0b01001001001001001001001001001001;
  for (int i = 0; i < mat_height; i += 3) {
    for (int j = 0; j < mat_width; j++) {
      h_weight[i * mat_width + j] = p1;
      h_weight[(i+1) * mat_width + j] = p1 << 1;
      h_weight[(i+2) * mat_width + j] = p1 << 2;
    }
  }

  scalar_t *h_scale = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  for (int i = 0; i < mul_size; i++) {
    h_scale[i] = -1.f;
  }

  scalar_t *h_bias = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  for (int i = 0; i < mul_size; i++) {
    h_bias[i] = 0.f;
  }

  scalar_t *h_output = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  
  iris::DMem d_input(h_input,sizeof(float)*vec_size);
  iris::DMem d_input_half(h_input_half,sizeof(half)*vec_size);
  iris::DMem d_weight(h_weight,sizeof(int)*mat_size);
  iris::DMem d_scale(h_scale,sizeof(float)*mul_size), d_bias(h_bias,sizeof(float)*mul_size);
  iris::DMem d_output(h_output,sizeof(float)*mul_size);
  auto launch = [&](const char* name,iris::DMem& vec,iris::DMem& mat,
                    iris::DMem& mul,iris::DMem& scale,iris::DMem& bias,
                    int height,int width) {
    iris::Task task;
    size_t zero_local=256, zero_global=((width+255)/256)*zero_local;
    void* zero_args[]={&mul,&width}; int zero_info[]={iris_w,sizeof(width)};
    iris_check(task.kernel("reset_output",1,nullptr,&zero_global,&zero_local,2,zero_args,zero_info),"reset output");
    size_t lws[]={256,1}, gws[]={static_cast<size_t>((height+23)/24)*256,static_cast<size_t>((width+255)/256)};
    void* args[]={&vec,&mat,&mul,&scale,&bias,&height,&width};
    int info[]={iris_r,iris_r,iris_rw,iris_r,iris_r,sizeof(height),sizeof(width)};
    iris_check(task.kernel(name,2,nullptr,gws,lws,7,args,info),"matmul");
    iris_check(task.submit(device,nullptr,true),"HIP reset/matmul task");
  };
  auto vecquant3matmul = [&](iris::DMem& vec,iris::DMem& mat,iris::DMem& mul,
                             iris::DMem& scale,iris::DMem& bias,int height,int width) {
    launch("VecQuant3MatMulKernel",vec,mat,mul,scale,bias,height,width);
  };
  auto vecquant3matmul_faster = [&](iris::DMem& vec,iris::DMem& mat,iris::DMem& mul,
                                    iris::DMem& scale,iris::DMem& bias,int height,int width) {
    launch("VecQuant3MatMulKernelFaster",vec,mat,mul,scale,bias,height,width);
  };
  auto flush_output = [&]() {
    iris::Task task;
    iris_check(task.flush_out(d_output),"flush output");
    iris_check(task.submit(device,nullptr,true),"HIP flush");
  };

  // verify vecquant3matmul
  vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                  mat_height, mat_width);
  flush_output();
  bool ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  all_ok &= ok;

  // verify vecquant3matmul_faster
  vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                         mat_height, mat_width);
  flush_output();
  ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  all_ok &= ok;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                    mat_height, mat_width);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                           mat_height, mat_width);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the faster kernel: %f (us)\n", (time * 1e-3f) / repeat);


  free(h_input);
  free(h_input_half);
  free(h_weight);
  free(h_scale);
  free(h_bias);
  free(h_output);
  iris_check(platform.finalize(),"finalize");
  return all_ok ? 0 : 1;
}
