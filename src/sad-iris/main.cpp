#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y)
#define THRESHOLD 20

struct Image {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> rgb;
};

static uint16_t read_u16(const unsigned char* p)
{
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t read_u32(const unsigned char* p)
{
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

static bool load_bmp_rgb(const char* path, Image& image)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open BMP: " << path << "\n";
    return false;
  }
  std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  if (data.size() < 54 || data[0] != 'B' || data[1] != 'M') {
    std::cerr << "Unsupported BMP header: " << path << "\n";
    return false;
  }

  const uint32_t pixel_offset = read_u32(&data[10]);
  const uint32_t dib_size = read_u32(&data[14]);
  if (dib_size < 40 || data.size() < pixel_offset) {
    std::cerr << "Unsupported BMP DIB header: " << path << "\n";
    return false;
  }

  const int32_t width = static_cast<int32_t>(read_u32(&data[18]));
  const int32_t signed_height = static_cast<int32_t>(read_u32(&data[22]));
  const uint16_t planes = read_u16(&data[26]);
  const uint16_t bits_per_pixel = read_u16(&data[28]);
  const uint32_t compression = read_u32(&data[30]);
  if (width <= 0 || signed_height == 0 || planes != 1 || bits_per_pixel != 24 || compression != 0) {
    std::cerr << "Only uncompressed 24-bit BMP files are supported: " << path << "\n";
    return false;
  }

  const int height = signed_height < 0 ? -signed_height : signed_height;
  const bool top_down = signed_height < 0;
  const size_t row_stride = ((static_cast<size_t>(width) * 3 + 3) / 4) * 4;
  if (pixel_offset + row_stride * static_cast<size_t>(height) > data.size()) {
    std::cerr << "Truncated BMP pixel data: " << path << "\n";
    return false;
  }

  image.width = width;
  image.height = height;
  image.rgb.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
  for (int row = 0; row < height; row++) {
    const int src_row = top_down ? row : (height - 1 - row);
    const unsigned char* src = &data[pixel_offset + row_stride * static_cast<size_t>(src_row)];
    unsigned char* dst = &image.rgb[static_cast<size_t>(row) * static_cast<size_t>(width) * 3];
    for (int col = 0; col < width; col++) {
      dst[col * 3 + 0] = src[col * 3 + 2];
      dst[col * 3 + 1] = src[col * 3 + 1];
      dst[col * 3 + 2] = src[col * 3 + 0];
    }
  }
  return true;
}

static void launch_compute(
    iris::DMem& mem_sad,
    iris::DMem& mem_image,
    iris::DMem& mem_template,
    int sad_array_size,
    int image_width,
    int image_height,
    int template_width,
    int template_height,
    int template_size)
{
  const size_t gws[2] = {
    ((static_cast<size_t>(image_width) + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X) * BLOCK_SIZE_X,
    ((static_cast<size_t>(image_height) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y) * BLOCK_SIZE_Y
  };
  const size_t lws[2] = { BLOCK_SIZE_X, BLOCK_SIZE_Y };
  iris::Task task;
  void* params[9] = {
    &mem_sad,
    &mem_image,
    &mem_template,
    &sad_array_size,
    &image_width,
    &image_height,
    &template_width,
    &template_height,
    &template_size
  };
  int pinfo[9] = {
    iris_w,
    iris_r,
    iris_r,
    sizeof(sad_array_size),
    sizeof(image_width),
    sizeof(image_height),
    sizeof(template_width),
    sizeof(template_height),
    sizeof(template_size)
  };
  task.kernel("compute_sad_array", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 9, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_find_min(int sad_array_size, iris::DMem& mem_sad, iris::DMem& mem_min)
{
  const size_t blocks = (static_cast<size_t>(sad_array_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };
  iris::Task task;
  void* params[3] = { &sad_array_size, &mem_sad, &mem_min };
  int pinfo[3] = { sizeof(sad_array_size), iris_r, iris_rw };
  task.kernel("find_min_in_sad_array", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_count(int sad_array_size, iris::DMem& mem_sad, iris::DMem& mem_min, iris::DMem& mem_occurrences)
{
  const size_t blocks = (static_cast<size_t>(sad_array_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };
  iris::Task task;
  void* params[4] = { &sad_array_size, &mem_sad, &mem_min, &mem_occurrences };
  int pinfo[4] = { sizeof(sad_array_size), iris_r, iris_r, iris_rw };
  task.kernel("get_num_of_occurrences", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
  task.flush_out(mem_min);
  task.flush_out(mem_occurrences);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "Usage: ./main <image> <template image> <repeat>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  Image main_image;
  Image template_image;
  if (!load_bmp_rgb(argv[1], main_image) || !load_bmp_rgb(argv[2], template_image)) {
    platform.finalize();
    return 1;
  }

  const int repeat = std::atoi(argv[3]);
  const int main_width = main_image.width;
  const int main_height = main_image.height;
  const int main_size = main_width * main_height;
  const int template_width = template_image.width;
  const int template_height = template_image.height;
  const int template_size = template_width * template_height;
  const int sad_array_size = (main_height - template_height + 1) * (main_width - template_width + 1);
  if (repeat < 1 || sad_array_size <= 0) {
    std::cerr << "Invalid repeat count or template dimensions\n";
    platform.finalize();
    return 1;
  }

  std::vector<int> sad_array(static_cast<size_t>(sad_array_size), 0);
  int min_mse = THRESHOLD;
  int num_occurrences = 0;

  iris::DMem mem_image(main_image.rgb.data(), main_image.rgb.size() * sizeof(unsigned char));
  iris::DMem mem_template(template_image.rgb.data(), template_image.rgb.size() * sizeof(unsigned char));
  iris::DMem mem_sad(sad_array.data(), sad_array.size() * sizeof(int));
  iris::DMem mem_min(&min_mse, sizeof(int));
  iris::DMem mem_occurrences(&num_occurrences, sizeof(int));

  double kernel_time = 0.0;
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    min_mse = THRESHOLD;
    num_occurrences = 0;
    mem_min.update(&min_mse);
    mem_occurrences.update(&num_occurrences);

    auto kbegin = std::chrono::steady_clock::now();
    launch_compute(mem_sad, mem_image, mem_template, sad_array_size,
                   main_width, main_height, template_width, template_height, template_size);
    launch_find_min(sad_array_size, mem_sad, mem_min);
    launch_count(sad_array_size, mem_sad, mem_min, mem_occurrences);
    auto kend = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::milliseconds>(kend - kbegin).count();

    mem_min.fetch(&min_mse);
    mem_occurrences.fetch(&num_occurrences);
  }

  auto end = std::chrono::steady_clock::now();
  float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

  std::cout << "Parallel Computation Results: " << std::endl;
  std::cout << "Kernel time in msec: " << kernel_time << std::endl;
  std::cout << "Elapsed time in msec: " << elapsed_time << std::endl;
  std::cout << "Main Image Dimensions: " << main_width << "*" << main_height << std::endl;
  std::cout << "Template Image Dimensions: " << template_width << "*" << template_height << std::endl;
  std::cout << "Found Minimum: " << min_mse << std::endl;
  std::cout << "Number of Occurances: " << num_occurrences << std::endl;

  platform.finalize();
  return 0;
}
