#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <iris/iris.hpp>
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>
#include "reference.h"
#include "common.h"

// GPU thread block size
#define TPB 1024

static int gpu_device=-1;
struct Param {void* ptr;int info;};
static Param param(iris::DMem& mem){return {&mem,iris_rw};}
template<class T> static Param param(const T& value){return {const_cast<T*>(&value),sizeof(T)};}
template<class... Args> static void launch(const char* name,size_t grid,size_t block,Args&&... args) {
  if(block<1 || block>1024) throw std::runtime_error("block size must be 1..1024");
  Param list[]={param(args)...};void* p[sizeof...(args)];int info[sizeof...(args)];
  for(size_t i=0;i<sizeof...(args);++i){p[i]=list[i].ptr;info[i]=list[i].info;}
  size_t g[]={grid*block},l[]={block};iris::Task task;
  if(task.kernel(name,1,nullptr,g,l,sizeof...(args),p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("kernel failed");
}
static void fetch(iris::DMem& mem,void* host) {
  iris::Task task;task.flush_out(mem);
  if(task.submit(gpu_device,nullptr,1) || mem.fetch(host)) throw std::runtime_error("fetch failed");
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3f) / repeat;
}

template<class T>
void validate_result(iris::DMem& device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  float* out_gpu = (float*)malloc(num_elements * sizeof(float));
  fetch(device_result,out_gpu);
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (!std::isfinite(out_gpu[i]) || !std::isfinite(cpu_reference[i]) || std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance) {
      printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
      nfaults++;

    }
  }

  printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
  if(const char* prefix=getenv("GROUPNORM_DUMP")) {
    static int index=0;
    std::string path=std::string(prefix)+"."+std::to_string(index++)+".bin";
    FILE* f=fopen(path.c_str(),"wb");if(!f||fwrite(out_gpu,sizeof(float),num_elements,f)!=num_elements)std::abort();fclose(f);
  }
  free(out_gpu);
  if(nfaults)std::exit(3);
}


// kernel launcher

void groupnorm_forward(
    iris::DMem& x, iris::DMem& weight, iris::DMem& bias,
    iris::DMem& out, iris::DMem& mean, iris::DMem& rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32);
    launch("groupnorm_forward_kernel",n_blocks,block_size,
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups
    );
}

void groupnorm_forward2(
    iris::DMem& x, iris::DMem& weight, iris::DMem& bias,
    iris::DMem& out, iris::DMem& mean, iris::DMem& rstd,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32);
    launch("groupnorm_forward_kernel2",n_blocks,block_size,
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups
    );
}

void groupnorm_backward(
    iris::DMem& dout, iris::DMem& x, iris::DMem& mean, iris::DMem& rstd, iris::DMem& weight,
    iris::DMem& dx, iris::DMem& dweight, iris::DMem& dbias,
    int B, int C, int H, int W, int n_groups
) {
    int img_size = H * W;
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32 * group_size);
    launch("groupnorm_backward_kernel",n_blocks,block_size,
        dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, group_size, n_groups
    );
}


// -----------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 7) {
      printf("Usage: %s <batch size> <number of channels> <height> <width> <number of groups> <repeat>\n", argv[0]);
      return 1;
    }

    uint64_t B = atoi(argv[1]);
    uint64_t C = atoi(argv[2]);
    uint64_t H = atoi(argv[3]);
    uint64_t W = atoi(argv[4]);
    uint64_t n_groups = atoi(argv[5]);
    int repeat = atoi(argv[6]);

    uint64_t img_size = H * W;

    srand(0);
    float *x = make_random_float(B * C * img_size);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);
    float *dout = make_random_float(B * C * img_size);

    float *out = (float*)malloc(B * C * img_size * sizeof(float));

    // reset values
    float *dx = (float*)calloc(B * C * img_size , sizeof(float));
    float *dweight = (float*)calloc(C , sizeof(float));
    float *dbias = (float*)calloc(C , sizeof(float));

    float *mean = (float*) malloc(B * n_groups * sizeof(float));
    float *rstd = (float*) malloc(B * n_groups * sizeof(float));

      iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;

    gpu_device=device;
    // device memory
    iris::DMem d_x(x,B * C * img_size * sizeof(float));
    iris::DMem d_weight(weight,C * sizeof(float));
    iris::DMem d_bias(bias,C * sizeof(float));
    // Independent GPU staging: reference computation must never initialize or
    // be overwritten by device gradients and validation flushes.
    std::vector<float> gpu_out(B*C*img_size),gpu_mean(B*n_groups),gpu_rstd(B*n_groups),
      gpu_dx(B*C*img_size),gpu_dweight(C),gpu_dbias(C);
    iris::DMem d_out(gpu_out.data(),B * C * img_size * sizeof(float));
    iris::DMem d_mean(gpu_mean.data(),B * n_groups * sizeof(float));
    iris::DMem d_rstd(gpu_rstd.data(),B * n_groups * sizeof(float));
    iris::DMem d_dout(dout,B * C * img_size * sizeof(float));
    iris::DMem d_dx(gpu_dx.data(),B * C * img_size * sizeof(float));
    iris::DMem d_dweight(gpu_dweight.data(),C * sizeof(float));
    iris::DMem d_dbias(gpu_dbias.data(),C * sizeof(float));
    printf("Checking forward pass\n");

    groupnorm_forward_ref(x, weight, bias, out, mean, rstd, B, C, img_size, n_groups);

    float fwd_acc = 1e-2;
    groupnorm_forward(d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(d_out, out, "out", B * C * img_size, fwd_acc);

    printf("Checking forward2 pass\n");

    groupnorm_forward2(d_x, d_weight, d_bias, d_out, d_mean, d_rstd, B, C, H, W, n_groups);
    validate_result(d_out, out, "out", B * C * img_size, fwd_acc);

    printf("Checking backward pass\n");

    groupnorm_backward_ref(dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, n_groups);

    groupnorm_backward(d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias, B, C, H, W, n_groups);
    float acc = 1e-2;
    printf("Checking dbias\n");
    validate_result(d_dbias, dbias, "dbias", C, acc);
    printf("Checking dweight\n");
    validate_result(d_dweight, dweight, "dweight", C, acc);
    printf("Checking dx\n");
    validate_result(d_dx, dx, "dx", B * C * img_size, 1.0f);
    printf("\n─────────────────────────────────────────────────────\n");

    printf("Forward pass benchmarks\n");
    float elapsed_time = benchmark_kernel(repeat, groupnorm_forward,
                                          d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                          B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Forward2 pass benchmarks\n");
    elapsed_time = benchmark_kernel(repeat, groupnorm_forward2,
                                    d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);

    printf("Backward pass benchmarks\n");
    elapsed_time = benchmark_kernel(repeat, groupnorm_backward,
                                    d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                    B, C, H, W, n_groups);
    printf("time %.4f us\n", elapsed_time);


    free(x);
    free(weight);
    free(bias);
    free(out);
    free(dout);
    free(dx);
    free(dweight);
    free(dbias);
    free(mean);
    free(rstd);

    platform.finalize();
    return 0;
}
