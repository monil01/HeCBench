#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int round_up(int group_size, int global_size)
{
  int r = global_size % group_size;
  return r == 0 ? global_size : global_size + group_size - r;
}

static bool read_token(FILE* f, std::string& token)
{
  token.clear();
  int c = 0;
  do {
    c = std::fgetc(f);
    if (c == '#') {
      while (c != '\n' && c != EOF) c = std::fgetc(f);
    }
  } while (c != EOF && std::isspace(c));
  if (c == EOF) return false;
  while (c != EOF && !std::isspace(c)) {
    token.push_back(static_cast<char>(c));
    c = std::fgetc(f);
  }
  return !token.empty();
}

static bool load_ppm4ub(const char* path, std::vector<unsigned int>& pixels,
                        unsigned int& width, unsigned int& height)
{
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::string tok;
  if (!read_token(f, tok) || tok != "P6" ||
      !read_token(f, tok)) {
    std::fclose(f);
    return false;
  }
  width = static_cast<unsigned int>(std::strtoul(tok.c_str(), nullptr, 10));
  if (!read_token(f, tok)) {
    std::fclose(f);
    return false;
  }
  height = static_cast<unsigned int>(std::strtoul(tok.c_str(), nullptr, 10));
  if (!read_token(f, tok) || std::atoi(tok.c_str()) != 255) {
    std::fclose(f);
    return false;
  }
  pixels.assign(static_cast<size_t>(width) * height, 0);
  for (size_t i = 0; i < pixels.size(); i++) {
    unsigned char rgb[3];
    if (std::fread(rgb, 1, 3, f) != 3) {
      std::fclose(f);
      return false;
    }
    pixels[i] = static_cast<unsigned int>(rgb[0]) |
                (static_cast<unsigned int>(rgb[1]) << 8) |
                (static_cast<unsigned int>(rgb[2]) << 16);
  }
  std::fclose(f);
  return true;
}

static void MedianFilterHost(const unsigned int* input, unsigned int* output,
                             int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const unsigned int zero = 0U;
      float median[3] = {128.0f, 128.0f, 128.0f};
      float minb[3] = {0.0f, 0.0f, 0.0f};
      float maxb[3] = {255.0f, 255.0f, 255.0f};
      for (int search = 0; search < 8; search++) {
        unsigned int high[3] = {0, 0, 0};
        for (int row = -1; row <= 1; row++) {
          for (int col = -1; col <= 1; col++) {
            const int yy = y + row;
            const int xx = x + col;
            const unsigned int pix = (xx >= 0 && xx < width && yy >= 0 && yy < height)
                                         ? input[yy * width + xx]
                                         : zero;
            const unsigned char* rgba = reinterpret_cast<const unsigned char*>(&pix);
            high[0] += (median[0] < rgba[0]);
            high[1] += (median[1] < rgba[1]);
            high[2] += (median[2] < rgba[2]);
          }
        }
        for (int c = 0; c < 3; c++) {
          if (high[c] > 4)
            minb[c] = median[c];
          else
            maxb[c] = median[c];
          median[c] = 0.5f * (maxb[c] + minb[c]);
        }
      }
      unsigned int packed = 0x000000FF & static_cast<unsigned int>(median[0] + 0.5f);
      packed |= 0x0000FF00 & (static_cast<unsigned int>(median[1] + 0.5f) << 8);
      packed |= 0x00FF0000 & (static_cast<unsigned int>(median[2] + 0.5f) << 16);
      output[y * width + x] = packed;
    }
  }
}

static bool compare_pixels(const unsigned int* ref, const unsigned int* got, size_t n)
{
  size_t errors = 0;
  const size_t allowed = static_cast<size_t>(n * 0.0001f);
  for (size_t i = 0; i < n; i++) {
    const unsigned char* a = reinterpret_cast<const unsigned char*>(&ref[i]);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&got[i]);
    for (int c = 0; c < 3; c++) {
      if (std::abs(static_cast<int>(a[c]) - static_cast<int>(b[c])) > 1) {
        errors++;
        break;
      }
    }
  }
  return errors <= allowed;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <image file> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const char* path = argv[1];
  const int cycles = std::atoi(argv[2]);
  unsigned int width = 0, height = 0;
  std::vector<unsigned int> input;
  if (!load_ppm4ub(path, input, width, height)) {
    std::printf("Error: failed to load %s\n", path);
    platform.finalize();
    return 1;
  }

  const unsigned int max_width = 1920;
  const unsigned int max_height = 1080;
  const size_t max_words = static_cast<size_t>(max_width) * max_height;
  const size_t active_words = static_cast<size_t>(width) * height;
  if (width > max_width || height > max_height) {
    std::printf("Error: Image Dimensions exceed the maximum values");
    platform.finalize();
    return 1;
  }

  std::vector<unsigned int> input_padded(max_words, 0);
  std::vector<unsigned int> output(max_words, 0);
  std::vector<unsigned int> golden(max_words, 0);
  std::copy(input.begin(), input.end(), input_padded.begin());

  std::printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %lu bpp\n\n",
              path, width, height, sizeof(unsigned int) << 3);

  iris::DMem mem_in(input_padded.data(), max_words * sizeof(unsigned int));
  iris::DMem mem_out(output.data(), max_words * sizeof(unsigned int));

  const int block_x = 16;
  const int block_y = 4;
  const int local_pitch = block_x + 2;
  const size_t gws[2] = {static_cast<size_t>(round_up(block_x, width)),
                         static_cast<size_t>(round_up(block_y, height))};
  const size_t lws[2] = {static_cast<size_t>(block_x), static_cast<size_t>(block_y)};

  double time = 0.0;
  std::printf("\nRunning MedianFilterGPU for %d cycles...\n\n", cycles);
  for (int i = 0; i < cycles + 1; i++) {
    iris::Task task;
    const int iwidth = static_cast<int>(width);
    const int iheight = static_cast<int>(height);
    void* params[5] = {&mem_in, &mem_out, const_cast<int*>(&local_pitch),
                       const_cast<int*>(&iwidth), const_cast<int*>(&iheight)};
    int pinfo[5] = {iris_r, iris_w, sizeof(local_pitch), sizeof(iwidth), sizeof(iheight)};
    auto start = std::chrono::steady_clock::now();
    task.kernel("ckMedian", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
                5, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    if (i > 0) time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  mem_out.fetch(output.data());
  std::printf("Average kernel execution time: %f (s)\n\n", (time * 1e-9f) / cycles);

  MedianFilterHost(input_padded.data(), golden.data(), width, height);
  std::printf("Comparing GPU Result to CPU Result...\n");
  const bool match = compare_pixels(golden.data(), output.data(), active_words);
  std::printf("\nGPU Result %s CPU Result within tolerance...\n",
              match ? "matches" : "DOESN'T match");
  std::printf("%s\n", match ? "PASS" : "FAIL");

  platform.finalize();
  return 0;
}
