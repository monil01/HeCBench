#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define DEGREE_TO_RADIAN M_PI / 180.0
#define EARTH_RADIUS_KM 6371.0

struct alignas(16) HaversinePoint {
  double x, y, z, w;
};

static double verify(
    int size,
    const std::vector<double>& output,
    const std::vector<double>& expected_output)
{
  double error_rate = 0.0;
  for (int i = 0; i < size; i++) {
    double error = std::fabs(output[i] - expected_output[i]);
    if (error > error_rate) error_rate = error;
  }
  std::printf("The maximum error in distance is %f\n", error_rate);
  return error_rate;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <file> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const char* filename = argv[1];
  int repeat = std::atoi(argv[2]);

  std::printf("Reading city locations from file %s...\n", filename);
  FILE* fp = std::fopen(filename, "r");
  if (fp == nullptr) {
    std::perror("Error opening the file");
    platform.finalize();
    return 1;
  }

  int num_cities = 2097152;
  int num_ref_cities = 6;
  int index_map[] = {436483, 1952407, 627919, 377884, 442703, 1863423};
  int N = num_cities * num_ref_cities;
  int city = 0;
  double lat, lon;

  std::vector<HaversinePoint> input(static_cast<size_t>(N));
  std::vector<double> output(static_cast<size_t>(N));
  std::vector<double> expected_output(static_cast<size_t>(N));

  while (std::fscanf(fp, "%lf %lf\n", &lat, &lon) != EOF) {
    input[city].x = lat;
    input[city].y = lon;
    city++;
    if (city == num_cities) break;
  }
  std::fclose(fp);

  for (int c = 1; c < num_ref_cities; c++) {
    std::copy(input.begin(), input.begin() + num_cities,
              input.begin() + static_cast<size_t>(c) * num_cities);
  }
  for (int c = 0; c < num_ref_cities; c++) {
    int index = index_map[c] - 1;
    for (int j = c * num_cities; j < (c + 1) * num_cities; ++j) {
      input[j].z = input[index].x;
      input[j].w = input[index].y;
    }
  }

  for (int i = 0; i < N; i++) {
    double a_lat = input[i].x;
    double a_lon = input[i].y;
    double b_lat = input[i].z;
    double b_lon = input[i].w;

    auto ax = a_lon * DEGREE_TO_RADIAN;
    auto ay = a_lat * DEGREE_TO_RADIAN;
    auto bx = b_lon * DEGREE_TO_RADIAN;
    auto by = b_lat * DEGREE_TO_RADIAN;

    auto x = (bx - ax) / 2.0;
    auto y = (by - ay) / 2.0;
    auto sinysqrd = std::sin(y) * std::sin(y);
    auto sinxsqrd = std::sin(x) * std::sin(x);
    auto scale = std::cos(ay) * std::cos(by);
    expected_output[i] =
        2.0 * EARTH_RADIUS_KM * std::asin(std::sqrt(sinysqrd + sinxsqrd * scale));
  }

  iris::DMem d_loc(input.data(), sizeof(HaversinePoint) * static_cast<size_t>(N));
  iris::DMem d_dist(output.data(), sizeof(double) * static_cast<size_t>(N));

  const int threads = 256;
  const int grids = (N + threads - 1) / threads;
  const size_t gws[1] = {static_cast<size_t>(grids * threads)};
  const size_t lws[1] = {static_cast<size_t>(threads)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = {&d_loc, &d_dist, &N};
    int pinfo[3] = {iris_r, iris_w, sizeof(N)};
    task.kernel("compute_haversine_distance", 1, nullptr,
                const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params,
                pinfo);
    task.flush_out(d_dist);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time %f (s)\n",
              (time * 1e-9f) / repeat);

  d_dist.fetch(output.data());
  double max_error = verify(N, output, expected_output);

  platform.finalize();
  return max_error <= 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
