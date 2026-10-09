#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

struct Float3 {
  float x;
  float y;
  float z;
};

struct Float4 {
  float x;
  float y;
  float z;
  float w;
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int width = std::atoi(argv[1]);
  const int height = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  const int numPts = width * height;
  const size_t size = static_cast<size_t>(numPts) * sizeof(Float3);
  const size_t normal_size = static_cast<size_t>(numPts) * sizeof(Float4);

  Float3* points = static_cast<Float3*>(std::malloc(size));
  Float4* normal_points = static_cast<Float4*>(std::malloc(normal_size));
  if (!points || !normal_points) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < numPts; i++) {
    points[i].x = std::rand() % width;
    points[i].y = std::rand() % height;
    points[i].z = std::rand() % 256;
  }

  iris::DMem mem_points(points, size);
  iris::DMem mem_normal_points(normal_points, normal_size);

  size_t lws = 256;
  size_t gws = ((static_cast<size_t>(numPts) + lws - 1) / lws) * lws;
  void* params[] = {&mem_points, &mem_normal_points, (void*)&width,
                    (void*)&height, (void*)&numPts};
  int pinfo[] = {iris_r, iris_w, sizeof(width), sizeof(height), sizeof(numPts)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel("ne", 1, nullptr, &gws, &lws, 5, params, pinfo);
    task.flush_out(mem_normal_points);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (time * 1e-9f) / repeat);

  float sx = 0.0f;
  float sy = 0.0f;
  float sz = 0.0f;
  float sw = 0.0f;
  for (int i = 0; i < numPts; i++) {
    sx += normal_points[i].x;
    sy += normal_points[i].y;
    sz += normal_points[i].z;
    sw += normal_points[i].w;
  }
  std::printf("Checksum: x=%f y=%f z=%f w=%f\n", sx, sy, sz, sw);

  std::free(normal_points);
  std::free(points);
  platform.finalize();
  return 0;
}
