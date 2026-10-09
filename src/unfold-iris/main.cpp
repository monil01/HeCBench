#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nelem = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const int64_t size = 2;
  const int64_t step = 1;
  const int64_t grad_in_dim_stride = 1;
  const int64_t grad_in_last_dim_stride = 1;
  const int64_t grad_in_dim_size = nelem;
  const int64_t grad_out_dim_stride = 1;
  const int total_n_elems = static_cast<int>(grad_in_dim_size);

  using scalar_t = int;

  const size_t input_size_bytes = static_cast<size_t>(grad_in_dim_size) * sizeof(scalar_t);
  const size_t output_size_bytes = static_cast<size_t>(grad_in_dim_size) * sizeof(scalar_t);

  scalar_t* h_grad_in = static_cast<scalar_t*>(std::malloc(input_size_bytes));
  scalar_t* h_grad_out = static_cast<scalar_t*>(std::calloc(static_cast<size_t>(grad_in_dim_size), sizeof(scalar_t)));
  if (!h_grad_in || !h_grad_out) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_grad_in);
    std::free(h_grad_out);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int64_t i = 0; i < grad_in_dim_size; i++) {
    h_grad_in[i] = std::rand() % 256;
  }

  const int64_t h_idx_dim = 0;

  iris::DMem mem_grad_in(h_grad_in, input_size_bytes);
  iris::DMem mem_grad_out(h_grad_out, output_size_bytes);
  iris::DMem mem_idx_dim(const_cast<int64_t*>(&h_idx_dim), sizeof(h_idx_dim));

  const size_t n_threads = 64;
  const size_t n_elems_per_thread = 4;
  const size_t total_work_block = n_threads * n_elems_per_thread;
  const size_t blocks = (static_cast<size_t>(total_n_elems) + total_work_block - 1) / total_work_block;
  size_t gws[1] = { blocks * n_threads };
  size_t lws[1] = { n_threads };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[10] = {
      &mem_grad_out,
      &mem_grad_in,
      &mem_idx_dim,
      const_cast<int*>(&total_n_elems),
      const_cast<int64_t*>(&size),
      const_cast<int64_t*>(&step),
      const_cast<int64_t*>(&grad_in_dim_stride),
      const_cast<int64_t*>(&grad_in_last_dim_stride),
      const_cast<int64_t*>(&grad_in_dim_size),
      const_cast<int64_t*>(&grad_out_dim_stride)
    };
    int pinfo[10] = {
      iris_rw,
      iris_r,
      iris_r,
      sizeof(total_n_elems),
      sizeof(size),
      sizeof(step),
      sizeof(grad_in_dim_stride),
      sizeof(grad_in_last_dim_stride),
      sizeof(grad_in_dim_size),
      sizeof(grad_out_dim_stride)
    };
    task.kernel("unfold_backward_kernel", 1, nullptr, gws, lws, 10, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush_task;
  flush_task.flush_out(mem_grad_out);
  flush_task.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of unfold backward kernel: %f (us)\n", (time * 1e-3f) / repeat);

  mem_grad_out.fetch(h_grad_out);

  bool ok = true;
  for (int64_t i = 0; i < grad_in_dim_size; i++) {
    if (repeat * h_grad_in[i] != h_grad_out[i]) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(h_grad_in);
  std::free(h_grad_out);
  platform.finalize();
  return ok ? 0 : 1;
}
