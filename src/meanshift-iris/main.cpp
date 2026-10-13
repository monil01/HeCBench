#include <stdio.h>
#include <chrono>
#include <iostream>
#include "iris_support.h"
#include "utils.h"
#include "constants.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to data> <path to centroids>" << std::endl;
    return 1;
  }
  const auto path_to_data = argv[1];
  const auto path_to_centroids = argv[2];
  size_t iterations=mean_shift::gpu::NUM_ITER;
  if(const char* value=std::getenv("HECBENCH_MEANSHIFT_ITERATIONS"))
    iterations=std::strtoul(value,nullptr,10);
  if(iterations<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();
  bool ok=true;

  constexpr auto N = mean_shift::gpu::N;
  constexpr auto D = mean_shift::gpu::D;
  constexpr auto M = mean_shift::gpu::M;
  constexpr auto THREADS = mean_shift::gpu::THREADS;
  constexpr auto BLOCKS = mean_shift::gpu::BLOCKS;
  constexpr auto TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::gpu::DIST_TO_REAL;

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  // Load data
  const std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result;

  std::array<float,N*D> backing_a=data, backing_b={};
  size_t data_bytes=N*D*sizeof(float);
  iris::DMem mem_a(backing_a.data(),data_bytes), mem_b(backing_b.data(),data_bytes);
  iris::DMem* d_data=&mem_a;
  iris::DMem* d_data_next=&mem_b;
  size_t lws=THREADS, gws=BLOCKS*THREADS;
  auto launch = [&](const char* name) {
    iris::Task task;
    void* args[]={d_data,d_data_next}; int info[]={iris_r,iris_w};
    iris_check(task.kernel(name,1,nullptr,&gws,&lws,2,args,info),"kernel");
    iris_check(task.submit(device,nullptr,true),"HIP task");
    std::swap(d_data,d_data_next);
  };
  auto flush_result = [&](const char* filename) {
    iris::Task task;
    iris_check(task.flush_out(*d_data),"flush result");
    iris_check(task.submit(device,nullptr,true),"HIP flush");
    result=(d_data==&mem_a)?backing_a:backing_b;
    if(std::getenv("HECBENCH_DUMP"))
      mean_shift::gpu::utils::write_csv<N,D>(result,filename,',');
  };

  // Run mean shift clustering and time the execution
  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iterations; ++i) {
    launch("mean_shift");
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / iterations << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  flush_result("meanshift_base.csv");
  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = centroids.size()==M && mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  ok &= centroids.size()==M && are_close;
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  // Reset device data
  auto& reset_host=(d_data==&mem_a)?backing_a:backing_b;
  reset_host=data;
  iris_check(d_data->update(reset_host.data()),"refresh original dataset for tiled phase");

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    launch("mean_shift_tiling");
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / iterations << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  flush_result("meanshift_opt.csv");

  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = centroids.size()==M && mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  ok &= centroids.size()==M && are_close;
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  iris_check(platform.finalize(),"finalize");
  return ok ? 0 : 1;
}
