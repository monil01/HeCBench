#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>

struct Vec3i {
  int x;
  int y;
  int z;

  Vec3i() : x(0), y(0), z(0) {}
  Vec3i(int vx, int vy, int vz) : x(vx), y(vy), z(vz) {}
  explicit Vec3i(int v) : x(v), y(v), z(v) {}

  long long_product() const {
    return static_cast<long>(x) * static_cast<long>(y) * static_cast<long>(z);
  }
};

static Vec3i operator+(const Vec3i& a, const Vec3i& b) {
  return Vec3i(a.x + b.x, a.y + b.y, a.z + b.z);
}

static Vec3i operator+(const Vec3i& a, int b) {
  return Vec3i(a.x + b, a.y + b, a.z + b);
}

static Vec3i operator-(const Vec3i& a, const Vec3i& b) {
  return Vec3i(a.x - b.x, a.y - b.y, a.z - b.z);
}

static Vec3i operator-(const Vec3i& a, int b) {
  return Vec3i(a.x - b, a.y - b, a.z - b);
}

static Vec3i min_vec(const Vec3i& a, const Vec3i& b) {
  return Vec3i(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

template <typename T>
static T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename T>
static T next_multiple(T val, T divisor) {
  return div_round_up(val, divisor) * divisor;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const Vec3i dims(4096, 1024, 1024);
  constexpr float K1 = 0.01f;
  constexpr float K2 = 0.03f;
  const float data_range = 1.f;

  constexpr bool use_sample_covariance = true;
  constexpr int win_size = 7;
  constexpr int crop = win_size >> 1;
  constexpr int NP = win_size * win_size * win_size;
  constexpr float cov_norm = use_sample_covariance ? static_cast<float>(NP) / (NP - 1) : 1.f;

  const Vec3i batch = min_vec(Vec3i(4096, 16, 16), dims);
  const Vec3i batch_grid = batch + win_size - 1;
  const auto batch_grid_count = next_multiple<size_t>(
      static_cast<size_t>(batch_grid.long_product()), 256);
  const size_t input_size = sizeof(float) * batch_grid_count;
  const size_t output_size = input_size * 3;
  const size_t output_count = output_size / sizeof(float);
  const size_t max_partial_count = div_round_up(output_count, static_cast<size_t>(256));

  float* h_grid_inference = static_cast<float*>(std::malloc(input_size));
  float* h_grid_reference = static_cast<float*>(std::malloc(input_size));
  float* h_grid_output = static_cast<float*>(std::malloc(output_size));
  float* h_partials = static_cast<float*>(std::malloc(max_partial_count * sizeof(float)));
  if (h_grid_inference == nullptr || h_grid_reference == nullptr ||
      h_grid_output == nullptr || h_partials == nullptr) {
    std::cerr << "Host allocation failed\n";
    std::free(h_grid_inference);
    std::free(h_grid_reference);
    std::free(h_grid_output);
    std::free(h_partials);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (size_t i = 0; i < batch_grid_count; i++) {
    h_grid_reference[i] = std::rand() / static_cast<float>(RAND_MAX);
    h_grid_inference[i] = h_grid_reference[i] * 0.75f;
  }
  std::fill(h_grid_output, h_grid_output + output_count, 0.f);
  std::fill(h_partials, h_partials + max_partial_count, 0.f);

  iris::DMem mem_reference(h_grid_reference, input_size);
  iris::DMem mem_inference(h_grid_inference, input_size);
  iris::DMem mem_output(h_grid_output, output_size);
  iris::DMem mem_partials(h_partials, max_partial_count * sizeof(float));
  mem_reference.update(h_grid_reference);
  mem_inference.update(h_grid_inference);
  mem_output.update(h_grid_output);
  mem_partials.update(h_partials);

  float ssim_sum = 0.f;
  long etime = 0;
  for (int z = crop; z < dims.z - crop; z += batch.z) {
    for (int y = crop; y < dims.y - crop; y += batch.y) {
      for (int x = crop; x < dims.x - crop; x += batch.x) {
        const Vec3i block_offset(x, y, z);
        const Vec3i block = min_vec(batch, dims - crop - block_offset);
        const auto block_count = static_cast<size_t>(block.long_product());
        if (block_count == 0) continue;

        const Vec3i block_grid = block + win_size - 1;
        const size_t gws[1] = {next_multiple<size_t>(block_count, 256)};
        const size_t lws[1] = {256};

        auto start = std::chrono::steady_clock::now();

        uint32_t dimx = static_cast<uint32_t>(block.x);
        uint32_t dimy = static_cast<uint32_t>(block.y);
        uint32_t dimz = static_cast<uint32_t>(block.z);
        uint32_t gdimx = static_cast<uint32_t>(block_grid.x);
        uint32_t gdimy = static_cast<uint32_t>(block_grid.y);
        uint32_t gdimz = static_cast<uint32_t>(block_grid.z);
        float range = data_range;
        float covariance_norm = cov_norm;
        float k1 = K1;
        float k2 = K2;

        iris::Task task;
        void* params[] = {&dimx,
                          &dimy,
                          &dimz,
                          &mem_reference,
                          &mem_inference,
                          &gdimx,
                          &gdimy,
                          &gdimz,
                          &range,
                          &covariance_norm,
                          &k1,
                          &k2,
                          &mem_output};
        int pinfo[] = {sizeof(dimx),
                       sizeof(dimy),
                       sizeof(dimz),
                       iris_r,
                       iris_r,
                       sizeof(gdimx),
                       sizeof(gdimy),
                       sizeof(gdimz),
                       sizeof(range),
                       sizeof(covariance_norm),
                       sizeof(k1),
                       sizeof(k2),
                       iris_w};
        task.kernel("compute_ssim", 1, nullptr, const_cast<size_t*>(gws),
                    const_cast<size_t*>(lws), 13, params, pinfo);
        task.submit(iris_gpu, nullptr, 1);

        uint32_t reduce_count = static_cast<uint32_t>(block_count);
        const uint32_t partial_count =
            static_cast<uint32_t>(div_round_up(block_count, static_cast<size_t>(256)));
        const size_t reduce_gws[1] = {static_cast<size_t>(partial_count) * 256};
        const size_t reduce_lws[1] = {256};
        iris::Task reduce_task;
        void* reduce_params[] = {&mem_output, &reduce_count, &mem_partials};
        int reduce_pinfo[] = {iris_r, sizeof(reduce_count), iris_w};
        reduce_task.kernel("reduce_ssim", 1, nullptr, const_cast<size_t*>(reduce_gws),
                           const_cast<size_t*>(reduce_lws), 3, reduce_params, reduce_pinfo);
        reduce_task.flush_out(mem_partials);
        reduce_task.submit(iris_gpu, nullptr, 1);

        ssim_sum += std::accumulate(h_partials, h_partials + partial_count, 0.f);

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        etime += time;
      }
    }
  }

  std::free(h_grid_reference);
  std::free(h_grid_inference);
  std::free(h_grid_output);
  std::free(h_partials);

  std::cout << "Total kernel execution time (s): " << etime * 1e-9 << "\n";
  std::cout << "Structural Similarity Index Measure: "
            << ssim_sum / static_cast<float>((dims - win_size + 1).long_product()) << "\n";

  platform.finalize();
  return 0;
}
