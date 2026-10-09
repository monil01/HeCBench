#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

typedef struct {
  int output_dim_b;
  int output_dim_c;
  int output_dim_h;
  int output_dim_w;
  int kernel_size;
  int image_w;
  int image_h;
} params;

static double LCG_random_double(unsigned long long* seed)
{
  const unsigned long long m = 9223372036854775808ULL;
  const unsigned long long a = 2806196910506780709ULL;
  const unsigned long long c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return static_cast<double>(*seed) / static_cast<double>(m);
}

static void reference(const float* img, const float* kernels, const float* offsets_h,
                      const float* offsets_v, float* output, const params p,
                      int offset_unit, int padding)
{
  const int dim_b = p.output_dim_b;
  const int dim_c = p.output_dim_c;
  const int dim_h = p.output_dim_h;
  const int dim_w = p.output_dim_w;
  const int kernels_size = p.kernel_size;
  const int img_w = p.image_w;
  const int img_h = p.image_h;
  const int k_size = static_cast<int>(std::sqrt(static_cast<float>(kernels_size)));
  const int w = img_w - 2 * padding;
  const int h = img_h - 2 * padding;

  for (int idb = 0; idb < dim_b; idb++)
  for (int idc = 0; idc < dim_c; idc++)
  for (int idy = 0; idy < dim_h; idy++)
  for (int idx = 0; idx < dim_w; idx++) {
    float result = 0;
    for (int k_y = 0; k_y < k_size; ++k_y) {
      for (int k_x = 0; k_x < k_size; ++k_x) {
        const size_t off_idx = static_cast<size_t>(idb) * k_size * k_size * dim_w * dim_h
                             + static_cast<size_t>(k_size * k_y + k_x) * dim_w * dim_h
                             + static_cast<size_t>(idy) * dim_w + idx;
        const float offset_h = offsets_h[off_idx] * offset_unit;
        const float offset_v = offsets_v[off_idx] * offset_unit;
        const float p_x = static_cast<float>(idx + 0.5f) / dim_w * w + k_x + offset_h - 0.5f;
        const float p_y = static_cast<float>(idy + 0.5f) / dim_h * h + k_y + offset_v - 0.5f;
        const float alpha = p_x - std::floor(p_x);
        const float beta = p_y - std::floor(p_y);
        const int xL = std::max(std::min(static_cast<int>(std::floor(p_x)), w + 2 * padding - 1), 0);
        const int xR = std::max(std::min(xL + 1, w + 2 * padding - 1), 0);
        const int yT = std::max(std::min(static_cast<int>(std::floor(p_y)), h + 2 * padding - 1), 0);
        const int yB = std::max(std::min(yT + 1, h + 2 * padding - 1), 0);
        const size_t img_base = static_cast<size_t>(idb) * dim_c * img_w * img_h
                              + static_cast<size_t>(idc) * img_w * img_h;
        float val = (1.f - alpha) * (1.f - beta) * img[img_base + static_cast<size_t>(yT) * img_w + xL];
        val += alpha * (1.f - beta) * img[img_base + static_cast<size_t>(yT) * img_w + xR];
        val += (1.f - alpha) * beta * img[img_base + static_cast<size_t>(yB) * img_w + xL];
        val += alpha * beta * img[img_base + static_cast<size_t>(yB) * img_w + xR];
        result += val * kernels[off_idx];
      }
    }
    output[static_cast<size_t>(idb) * dim_c * dim_w * dim_h
         + static_cast<size_t>(idc) * dim_w * dim_h
         + static_cast<size_t>(idy) * dim_w + idx] = result;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const params p = {128, 3, 480, 640, 9, 1024, 1024};
  const int padding = 1;
  const size_t image_size = static_cast<size_t>(p.output_dim_b) * p.output_dim_c * (p.image_w + padding) * (p.image_h + padding);
  const size_t offset_size = static_cast<size_t>(p.output_dim_b) * p.kernel_size * p.output_dim_w * p.output_dim_h;
  const size_t output_size = static_cast<size_t>(p.output_dim_b) * p.output_dim_c * p.output_dim_w * p.output_dim_h;

  float* img = static_cast<float*>(std::malloc(sizeof(float) * image_size));
  float* offsets_h = static_cast<float*>(std::malloc(sizeof(float) * offset_size));
  float* offsets_v = static_cast<float*>(std::malloc(sizeof(float) * offset_size));
  float* kernel = static_cast<float*>(std::malloc(sizeof(float) * offset_size));
  float* output = static_cast<float*>(std::malloc(sizeof(float) * output_size));
  float* output_ref = static_cast<float*>(std::malloc(sizeof(float) * output_size));
  if (!img || !offsets_h || !offsets_v || !kernel || !output || !output_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  unsigned long long seed = 123;
  for (size_t i = 0; i < image_size; i++) img[i] = static_cast<unsigned char>(256 * LCG_random_double(&seed));
  for (size_t i = 0; i < offset_size; i++) kernel[i] = static_cast<unsigned char>(256 * LCG_random_double(&seed));
  for (size_t i = 0; i < offset_size; i++) {
    offsets_h[i] = static_cast<float>(LCG_random_double(&seed));
    offsets_v[i] = static_cast<float>(LCG_random_double(&seed));
  }

  iris::DMem mem_img(img, sizeof(float) * image_size);
  iris::DMem mem_offsets_h(offsets_h, sizeof(float) * offset_size);
  iris::DMem mem_offsets_v(offsets_v, sizeof(float) * offset_size);
  iris::DMem mem_kernel(kernel, sizeof(float) * offset_size);
  iris::DMem mem_output(output, sizeof(float) * output_size);

  size_t gws[1] = { ((output_size + 255) / 256) * 256 };
  size_t lws[1] = { 256 };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    const int offset_unit = 1;
    void* args[9] = { &mem_img, &mem_kernel, &mem_offsets_h, &mem_offsets_v, &mem_output,
                      const_cast<params*>(&p), const_cast<int*>(&offset_unit),
                      const_cast<int*>(&padding), const_cast<size_t*>(&output_size) };
    int pinfo[9] = { iris_r, iris_r, iris_r, iris_r, iris_w,
                     static_cast<int>(sizeof(params)), sizeof(offset_unit),
                     sizeof(padding), static_cast<int>(sizeof(output_size)) };
    task.kernel("car", 1, nullptr, gws, lws, 9, args, pinfo);
    if (i == repeat - 1) task.flush_out(mem_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  reference(img, kernel, offsets_h, offsets_v, output_ref, p, 1, padding);

  float rmse = 0;
  for (size_t i = 0; i < output_size; i++) {
    rmse += (output_ref[i] - output[i]) * (output_ref[i] - output[i]);
  }
  std::printf("RMSE: %f\n", std::sqrt(rmse / output_size));

  std::free(img);
  std::free(offsets_h);
  std::free(offsets_v);
  std::free(kernel);
  std::free(output);
  std::free(output_ref);
  platform.finalize();
  return 0;
}
