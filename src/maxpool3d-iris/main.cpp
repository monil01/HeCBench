#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using DTYPE = float;

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::printf("Usage: %s <image width> <image height> <image count> <repeat>\n", argv[0]);
    return 1;
  }
  int i_img_width = std::atoi(argv[1]);
  int i_img_height = std::atoi(argv[2]);
  int i_img_count = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);
  int Hstride = 2, Vstride = 2;
  int o_img_width = i_img_width / Hstride;
  int o_img_height = i_img_height / Vstride;

  std::printf("input image width %d Hstride %d\n", i_img_width, Hstride);
  std::printf("input image height %d Vstride %d\n", i_img_height, Vstride);
  std::printf("output image width %d\n", o_img_width);
  std::printf("output image height %d\n", o_img_height);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int size_image = i_img_width * i_img_height;
  size_t mem_size_image = sizeof(DTYPE) * static_cast<size_t>(size_image);
  DTYPE* h_image = static_cast<DTYPE*>(std::malloc(mem_size_image * i_img_count));
  std::srand(2);
  for (int j = 0; j < i_img_count; j++) {
    for (int i = 0; i < size_image; i++) {
      h_image[(j * size_image) + i] = std::rand() % 256 / static_cast<DTYPE>(255);
    }
  }

  int size_output = o_img_width * o_img_height;
  size_t mem_size_output = sizeof(DTYPE) * static_cast<size_t>(size_output);
  DTYPE* h_output = static_cast<DTYPE*>(std::malloc(mem_size_output * i_img_count));
  DTYPE* d_output = static_cast<DTYPE*>(std::malloc(mem_size_output * i_img_count));
  if (!h_image || !h_output || !d_output) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  iris::DMem mem_image(h_image, mem_size_image * i_img_count);
  iris::DMem mem_result(d_output, mem_size_output * i_img_count);

  int pool_width = Hstride;
  int pool_height = Vstride;
  size_t gws[3] = {
    static_cast<size_t>(((o_img_width + 7) / 8) * 8),
    static_cast<size_t>(((o_img_height + 7) / 8) * 8),
    static_cast<size_t>(((i_img_count + 3) / 4) * 4)
  };
  size_t lws[3] = {8, 8, 4};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[10] = {&mem_image, &mem_result, &Hstride, &Vstride, &pool_width, &pool_height,
                        &i_img_count, &i_img_width, &i_img_height, &o_img_width};
    int pinfo[10] = {iris_r, iris_w, sizeof(Hstride), sizeof(Vstride), sizeof(pool_width),
                     sizeof(pool_height), sizeof(i_img_count), sizeof(i_img_width),
                     sizeof(i_img_height), sizeof(o_img_width)};
    task.kernel("maxpool3d", 3, nullptr, gws, lws, 10, params, pinfo);
    if (n == repeat - 1) task.flush_out(mem_result);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  mem_result.fetch(d_output);

  for (int z = 0; z < i_img_count; z++) {
    for (int y = 0; y < o_img_height; y++) {
      for (int x = 0; x < o_img_width; x++) {
        const int xidx = Hstride * x;
        const int yidx = Vstride * y;
        DTYPE maxval = static_cast<DTYPE>(0);
        for (int r = 0; r < pool_height; r++) {
          const int idxIntmp = ((z * i_img_height + yidx + r) * i_img_width) + xidx;
          for (int c = 0; c < pool_width; c++) {
            maxval = fmaxf(maxval, h_image[idxIntmp + c]);
          }
        }
        h_output[(((z * o_img_height) + y) * o_img_width) + x] = maxval;
      }
    }
  }

  int status = std::memcmp(h_output, d_output, sizeof(DTYPE) * static_cast<size_t>(i_img_count) * o_img_width * o_img_height);
  std::printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  std::free(h_image);
  std::free(h_output);
  std::free(d_output);
  platform.finalize();
  return status;
}
