#include <iris/iris.hpp>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "utils.hpp"

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <image width> <image height> <block size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const int block_size = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  auto gamma_f = [](ImgPixel& pixel) {
    const float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0f;
    std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.0f * v * v);
    if (gamma_pixel > 255) gamma_pixel = 255;
    pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
  };

  int index = 0;
  image.fill([&index, width, &fractal](ImgPixel& pixel) {
    const int x = index % width;
    const int y = index / width;

    auto fractal_pixel = fractal(x, y);
    if (fractal_pixel < 0) fractal_pixel = 0;
    if (fractal_pixel > 255) fractal_pixel = 255;
    pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

    ++index;
  });

  Img<ImgFormat::BMP> image2 = image;
  image.fill(gamma_f);

  iris::DMem mem_pixel(image2.data(), sizeof(ImgPixel) * image2.width() * image2.height());

  size_t gws[1] = { static_cast<size_t>(width * height) };
  size_t lws[1] = { static_cast<size_t>(block_size) };
  float total_time = 0.0f;

  for (int i = 0; i < repeat; i++) {
    mem_pixel.update(image2.data());

    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    void* params[1] = { &mem_pixel };
    int pinfo[1] = { iris_rw };
    task.kernel("gamma_correction", 1, nullptr, gws, lws, 1, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_pixel);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += static_cast<float>(time);
  }

  std::printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);

  mem_pixel.fetch(image2.data());

  if (check(image.begin(), image.end(), image2.begin())) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }

  platform.finalize();
  return 0;
}
