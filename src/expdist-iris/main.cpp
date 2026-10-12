#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef block_size_x
#define block_size_x 32
#endif
#ifndef block_size_y
#define block_size_y 8
#endif
#ifndef tile_size_x
#define tile_size_x 4
#endif
#ifndef tile_size_y
#define tile_size_y 4
#endif
#ifndef reduce_block_size
#define reduce_block_size 256
#endif

template <typename FP, int dim>
FP cost(FP *A, FP *B, FP *scale_A, FP *scale_B, int m, int n) {
  double sum = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      FP dist = 0;
      for (int d = 0; d < dim; d++) {
        dist += (A[i + d * m] - B[j + d * n]) *
                (A[i + d * m] - B[j + d * n]);
      }
      sum += std::exp(-dist/(scale_A[i] + scale_B[j]));
    }
  }
  return sum;
}

template <typename FP>
void test(const int size, const int repeat, const char* dist_name, const char* reduce_name) {
  const int grid_x = size / (block_size_x * tile_size_x);
  const int grid_y = size / (block_size_y * tile_size_y);
  const int nblocks = grid_x * grid_y;

  size_t point_size_bytes = sizeof(FP) * size * 2;
  size_t scale_size_bytes = sizeof(FP) * size;
  size_t cost_size_bytes = sizeof(FP) * nblocks;

  FP *A = static_cast<FP*>(std::malloc(point_size_bytes));
  FP *B = static_cast<FP*>(std::malloc(point_size_bytes));
  FP *scaleA = static_cast<FP*>(std::malloc(scale_size_bytes));
  FP *scaleB = static_cast<FP*>(std::malloc(scale_size_bytes));
  FP *cost_terms = static_cast<FP*>(std::malloc(cost_size_bytes));
  FP output = 0;
  if (!A || !B || !scaleA || !scaleB || !cost_terms) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(A); std::free(B); std::free(scaleA); std::free(scaleB); std::free(cost_terms);
    std::exit(1);
  }

  for (int i = 0; i < size * 2; i++) {
    A[i] = 1;
    B[i] = 0;
  }
  for (int i = 0; i < size; i++) {
    scaleA[i] = 1;
    scaleB[i] = 1;
  }
  for (int i = 0; i < nblocks; i++) cost_terms[i] = 0;

  iris::DMem mem_A(A, point_size_bytes);
  iris::DMem mem_B(B, point_size_bytes);
  iris::DMem mem_scaleA(scaleA, scale_size_bytes);
  iris::DMem mem_scaleB(scaleB, scale_size_bytes);
  iris::DMem mem_cost(cost_terms, cost_size_bytes);
  iris::DMem mem_output(&output, sizeof(FP));

  size_t dist_gws[2] = {
      static_cast<size_t>(grid_x * block_size_x),
      static_cast<size_t>(grid_y * block_size_y)};
  size_t dist_lws[2] = {block_size_x, block_size_y};
  size_t reduce_gws[1] = {reduce_block_size};
  size_t reduce_lws[1] = {reduce_block_size};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task_dist;
    void* dist_params[7] = {&mem_A, &mem_B, const_cast<int*>(&size), const_cast<int*>(&size),
                            &mem_scaleA, &mem_scaleB, &mem_cost};
    int dist_pinfo[7] = {iris_r, iris_r, sizeof(size), sizeof(size), iris_r, iris_r, iris_w};
    task_dist.kernel(dist_name, 2, nullptr, dist_gws, dist_lws, 7, dist_params, dist_pinfo);
    task_dist.submit(iris_gpu, nullptr, 1);

    iris::Task task_reduce;
    void* reduce_params[5] = {&mem_output, &mem_cost, const_cast<int*>(&size),
                              const_cast<int*>(&size), const_cast<int*>(&nblocks)};
    int reduce_pinfo[5] = {iris_w, iris_r, sizeof(size), sizeof(size), sizeof(nblocks)};
    task_reduce.kernel(reduce_name, 1, nullptr, reduce_gws, reduce_lws, 5, reduce_params, reduce_pinfo);
    if (i == repeat - 1) task_reduce.flush_out(mem_output);
    task_reduce.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  mem_output.fetch(&output);

  std::printf("    device result: %lf\n", (double)output);
  output = cost<FP, 2>(A, B, scaleA, scaleB, size, size);
  std::printf("      host result: %lf\n", (double)output);
  std::printf("analytical result: %lf\n\n", size * size * std::exp(-1.0));

  std::free(A); std::free(B); std::free(scaleA); std::free(scaleB); std::free(cost_terms);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("Usage ./%s <size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int size = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  std::printf("Test single precision\n");
  test<float>(size, repeat, "distance_float", "reduce_cross_term_float");

  std::printf("Test double precision\n");
  test<double>(size, repeat, "distance_double", "reduce_cross_term_double");

  platform.finalize();
  return 0;
}
