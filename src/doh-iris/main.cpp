#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int h = std::atoi(argv[1]);
  const int w = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (h <= 0 || w <= 0 || repeat <= 0) {
    std::fprintf(stderr, "height, width, and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  const int img_size = h * w;
  std::vector<float> input_img(img_size);
  std::vector<float> integral_img(img_size);
  std::vector<float> output_img(img_size, 0.0f);

  std::default_random_engine rng(123);
  std::normal_distribution<float> norm_dist(0.0f, 1.0f);
  for (int i = 0; i < img_size; ++i) input_img[i] = norm_dist(rng);

  std::printf("Integrating the input image may take a while...\n");
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      float s = 0.0f;
      for (int y = 0; y <= i; ++y)
        for (int x = 0; x <= j; ++x)
          s += input_img[y * w + x];
      integral_img[i * w + j] = s;
    }
  }

  constexpr float sigma = 4.0f;
  constexpr size_t block_size = 256;
  const size_t global_size = ((static_cast<size_t>(img_size) + block_size - 1) / block_size) * block_size;

  iris::DMem input_mem(integral_img.data(), integral_img.size() * sizeof(float));
  iris::DMem output_mem(output_img.data(), output_img.size() * sizeof(float));

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    void* params[] = {&input_mem, const_cast<int*>(&h), const_cast<int*>(&w),
                      const_cast<float*>(&sigma), &output_mem};
    int pinfo[] = {iris_r, sizeof(h), sizeof(w), sizeof(sigma), iris_w};
    size_t gws[] = {global_size};
    size_t lws[] = {block_size};
    iris::Task task;
    task.kernel("hessian_matrix_det", 1, nullptr, gws, lws, 5, params, pinfo);
    if (i == repeat - 1) task.flush_out(output_mem);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  double checksum = 0.0;
  for (float value : output_img) checksum += value;

  std::printf("Average kernel execution time : %f (us)\n", time * 1e-3 / repeat);
  std::printf("Kernel checksum: %lf\n", checksum);

  platform.finalize();
  return 0;
}
