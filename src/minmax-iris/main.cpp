#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "utils.hpp"

struct extrema_pair {
  float min_value;
  float max_value;
  int min_index;
  int max_index;
};

struct compare_distance {
  bool operator()(const vec_2d<float>& a, const vec_2d<float>& b) const
  {
    return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
  }
};

static void reduce_partials(const std::vector<extrema_pair>& partials,
                            int& min_index, int& max_index)
{
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  min_index = 0;
  max_index = 0;

  for (const auto& p : partials) {
    if (p.min_index >= 0 && p.min_value < min_value) {
      min_value = p.min_value;
      min_index = p.min_index;
    }
    if (p.max_index >= 0 && p.max_value > max_value) {
      max_value = p.max_value;
      max_index = p.max_index;
    }
  }
}

static void eval(const float bounding_box_size, const int repeat)
{
  auto const [total_points, points] = generate_points(bounding_box_size);
  std::printf("Total number of points: %zu\n", total_points);

  const int block_size = 256;
  const int num_blocks = static_cast<int>((total_points + block_size - 1) / block_size);
  std::vector<extrema_pair> partials(num_blocks);
  vec_2d<float> min_point[2] = {{}, {}};
  vec_2d<float> max_point[2] = {{}, {}};

  iris::DMem mem_points(const_cast<vec_2d<float>*>(points.data()), sizeof(vec_2d<float>) * total_points);
  iris::DMem mem_partials(partials.data(), sizeof(extrema_pair) * partials.size());

  float minmax_time_us = 0.0f;
  const size_t gws[1] = { static_cast<size_t>(num_blocks * block_size) };
  const size_t lws[1] = { static_cast<size_t>(block_size) };

  for (int i = 0; i < repeat; i++) {
    const int n = static_cast<int>(total_points);
    void* params[3] = { &mem_points, &mem_partials, const_cast<int*>(&n) };
    int pinfo[3] = { iris_r, iris_w, sizeof(n) };

    const auto start = std::chrono::steady_clock::now();
    iris::Task task;
    task.kernel("minmax_kernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    task.flush_out(mem_partials);
    task.submit(iris_gpu, nullptr, 1);
    const auto end = std::chrono::steady_clock::now();
    minmax_time_us += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3f;

    int min_index = 0;
    int max_index = 0;
    reduce_partials(partials, min_index, max_index);
    min_point[0] = points[min_index];
    max_point[0] = points[max_index];
    min_point[1] = points[min_index];
    max_point[1] = points[max_index];
  }

  std::printf("Average execution time of thrust:min() and thrust:max(): %f (us)\n",
              minmax_time_us / repeat);
  std::printf("Average execution time of thrust:min_max(): %f (us)\n",
              minmax_time_us / repeat);

  auto const min_itr = std::min_element(points.begin(), points.end(), compare_distance());
  auto const r_min_point = points[std::distance(points.begin(), min_itr)];
  auto const max_itr = std::max_element(points.begin(), points.end(), compare_distance());
  auto const r_max_point = points[std::distance(points.begin(), max_itr)];

  bool ok = (min_point[0] == r_min_point) && (max_point[0] == r_max_point);
  ok &= (min_point[1] == r_min_point) && (max_point[1] == r_max_point);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <bounding-box size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int size = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  eval(static_cast<float>(size), repeat);

  platform.finalize();
  return 0;
}
