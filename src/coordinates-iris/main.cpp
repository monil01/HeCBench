#include <iris/iris.hpp>

#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <typename T>
const char* kernel_name();

template <>
const char* kernel_name<double>() { return "coordinates_transform_double"; }

template <>
const char* kernel_name<float>() { return "coordinates_transform_float"; }

template <typename T>
bool coordinates_transform(const int num_coords, const int repeat)
{
  std::vector<lonlat_2d<T>> h_input(num_coords);
  std::vector<cartesian_2d<T>> h_output(num_coords);
  std::vector<cartesian_2d<T>> h_ref_output(num_coords);

  lonlat_2d<T> h_origin;
  h_origin.x = static_cast<T>(90);
  h_origin.y = static_cast<T>(45);

  for (int i = 0; i < num_coords; i++) {
    h_input[i].x = static_cast<T>(std::rand() % 360 - 180);
    h_input[i].y = static_cast<T>(std::rand() % 180 - 90);
  }

  iris::DMem mem_input(h_input.data(), h_input.size() * sizeof(lonlat_2d<T>));
  iris::DMem mem_output(h_output.data(), h_output.size() * sizeof(cartesian_2d<T>));

  constexpr int block_size = 256;
  size_t gws[1] = {
      static_cast<size_t>((num_coords + block_size - 1) / block_size * block_size)};
  size_t lws[1] = {block_size};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    int coords_arg = num_coords;
    T origin_x = h_origin.x;
    T origin_y = h_origin.y;
    void* params[5] = {&mem_input, &mem_output, &coords_arg, &origin_x, &origin_y};
    int pinfo[5] = {
        iris_r,
        iris_w,
        sizeof(num_coords),
        sizeof(origin_x),
        sizeof(origin_y)};
    task.kernel(kernel_name<T>(), 1, nullptr, gws, lws, 5, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of device transform: %f (us)\n",
              (time * 1e-3f) / repeat);

  mem_output.fetch(h_output.data());

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; i++) {
    std::transform(h_input.cbegin(),
                   h_input.cend(),
                   h_ref_output.begin(),
                   to_cartesian_functor<T>(h_origin));
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of host transform: %f (us)\n",
              (time * 1e-3f) / 10);

  bool ok = true;
  for (int i = 0; i < num_coords; i++) {
    if (!(h_output[i] == h_ref_output[i])) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num_coords = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  std::srand(123);

  std::printf("\nDouble-precision coordinates transform\n");
  bool ok_double = coordinates_transform<double>(num_coords, repeat);

  std::printf("\nSingle-precision coordinates transform\n");
  bool ok_float = coordinates_transform<float>(num_coords, repeat);

  platform.finalize();
  return ok_double && ok_float ? 0 : 2;
}
