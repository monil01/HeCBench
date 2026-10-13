#include "hadamard_common.h"
#include "dmem.hpp"
#include <type_traits>
template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_launch(HadamardParamsBase& params, hipStream_t) {
    char name[64];
    const char* tag=std::is_same<input_t,float>::value ? "f32" : (std::is_same<input_t,__half>::value ? "f16" : "bf");
    snprintf(name,sizeof(name),"hadamard_%s_%d",tag,kLogN);
    port::launch(name,dim3(params.batch),dim3(kNThreads),params.x_ptr,params.out_ptr,
                 params.batch,params.dim,params.log_N,params.x_batch_stride,params.out_batch_stride,params.scale);
}

template<typename input_t>
void fast_hadamard_transform(HadamardParamsBase &params, hipStream_t stream) {
    if (params.log_N == 3) {
        fast_hadamard_transform_launch<1, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_launch<2, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_launch<4, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_launch<8, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_launch<16, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_launch<32, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_launch<32, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_launch<128, 10, input_t>(params, stream);
    } else if (params.log_N == 11) {
        fast_hadamard_transform_launch<256, 11, input_t>(params, stream);
    } else if (params.log_N == 12) {
        fast_hadamard_transform_launch<256, 12, input_t>(params, stream);
    } else if (params.log_N == 13) {
        fast_hadamard_transform_launch<256, 13, input_t>(params, stream);
    } else if (params.log_N == 14) {
        fast_hadamard_transform_launch<256, 14, input_t>(params, stream);
    } else if (params.log_N == 15) {
        fast_hadamard_transform_launch<256, 15, input_t>(params, stream);
    }
}


template <typename T>
void hadamard_transform(int batch_size, int dim, int repeat) {

    assert(dim % (8) == 0);
    assert(dim <= 32768);

    float scale = 1.f / std::sqrt(dim);
    int64_t numel = (int64_t)batch_size * dim;

    std::vector<T> h_x (numel);
    std::vector<T> h_out (numel);
    std::vector<T> r_out (numel);

    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int64_t i = 0; i < numel; ++i) {
      h_x[i] = T(dis(gen));
    }

    void *d_x, *d_out;
    GPU_CHECK(port::allocate((void**)&d_x, numel * sizeof(T)));
    GPU_CHECK(port::allocate((void**)&d_out, numel * sizeof(T)));

    GPU_CHECK(port::copy(d_x, h_x.data(), numel * sizeof(T), port::h2d));

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, d_x, d_out, scale);

    for (int i = 0; i < 30; i++) {
      fast_hadamard_transform<T>(params, 0);
    }
    GPU_CHECK(iris_synchronize());

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      fast_hadamard_transform<T>(params, 0);
    }

    GPU_CHECK(iris_synchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("batch size: %d | hidden dimension: %d | Average kernel execution time : %f (us)\n",
            batch_size, dim, (time * 1e-3f) / repeat);

    GPU_CHECK(port::copy(h_out.data(), d_out, numel * sizeof(T), port::d2h));

    reference(h_x, r_out, batch_size, dim, scale);

    bool ok = true;
    for (int64_t i = 0; i < numel; ++i) {
      if (std::fabs((float)h_out[i] - (float)r_out[i]) > 1e-3f) {
        printf("Mismatch at index %ld %f %f\n", i, (float)h_out[i], (float)r_out[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    GPU_CHECK(port::release(d_x));
    GPU_CHECK(port::release(d_out));
}

int main(int argc, char* argv[])
{
  port::Session session(&argc, &argv);
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  for (int dim = 8; dim <= 32768; dim = dim * 8) {
    hadamard_transform<float>(batch_size, dim, repeat);
    hadamard_transform<__half>(batch_size, dim, repeat);
    hadamard_transform<__hip_bfloat16>(batch_size, dim, repeat);
  }
  return 0;
}
