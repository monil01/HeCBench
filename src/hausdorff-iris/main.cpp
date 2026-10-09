#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

struct alignas(8) Point2 {
  float x, y;
};

static inline float hd(const Point2 ap, const Point2 bp)
{
  return (ap.x - bp.x) * (ap.x - bp.x) + (ap.y - bp.y) * (ap.y - bp.y);
}

static float computeDirDistance(
    const std::vector<Point2>& Apoints,
    const std::vector<Point2>& Bpoints,
    int numA,
    int numB)
{
  std::vector<float> disA(numA);
  for (int i = 0; i < numA; i++) {
    float d = std::numeric_limits<float>::max();
    for (int j = 0; j < numB; j++) {
      float t = hd(Apoints[i], Bpoints[j]);
      d = std::min(t, d);
    }
    disA[i] = d;
  }
  std::sort(disA.begin(), disA.end());
  return disA[numA - 1];
}

static float hausdorff_distance(
    const std::vector<Point2>& Apoints,
    const std::vector<Point2>& Bpoints,
    int numA,
    int numB)
{
  float hAB = computeDirDistance(Apoints, Bpoints, numA, numB);
  float hBA = computeDirDistance(Bpoints, Apoints, numB, numA);
  return std::max(hAB, hBA);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of points in space A>", argv[0]);
    std::printf(" <number of points in space B> <repeat>\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int num_Apoints = std::atoi(argv[1]);
  int num_Bpoints = std::atoi(argv[2]);
  int repeat = std::atoi(argv[3]);

  std::vector<Point2> h_Apoints(num_Apoints);
  std::vector<Point2> h_Bpoints(num_Bpoints);

  std::srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    h_Apoints[i].y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    h_Bpoints[i].y = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }

  float h_distance[2] = {-1.0f, -1.0f};
  iris::DMem d_Apoints(h_Apoints.data(), sizeof(Point2) * static_cast<size_t>(num_Apoints));
  iris::DMem d_Bpoints(h_Bpoints.data(), sizeof(Point2) * static_cast<size_t>(num_Bpoints));
  iris::DMem d_distance(h_distance, sizeof(h_distance));

  const int blocks = 256;
  const size_t lws[1] = {static_cast<size_t>(blocks)};
  const size_t gwsA[1] = {static_cast<size_t>(((num_Apoints + blocks - 1) / blocks) * blocks)};
  const size_t gwsB[1] = {static_cast<size_t>(((num_Bpoints + blocks - 1) / blocks) * blocks)};

  double time = 0.0;
  for (int i = 0; i < repeat; i++) {
    h_distance[0] = -1.0f;
    h_distance[1] = -1.0f;
    d_distance.update(h_distance);

    auto start = std::chrono::steady_clock::now();

    int distance_index = 0;
    iris::Task taskA;
    void* paramsA[6] = {&d_Apoints, &d_Bpoints, &d_distance, &num_Apoints, &num_Bpoints, &distance_index};
    int pinfoA[6] = {iris_r, iris_r, iris_rw, sizeof(num_Apoints), sizeof(num_Bpoints), sizeof(distance_index)};
    taskA.kernel("computeDistance", 1, nullptr, const_cast<size_t*>(gwsA),
                 const_cast<size_t*>(lws), 6, paramsA, pinfoA);
    taskA.submit(iris_gpu, nullptr, 1);

    distance_index = 1;
    iris::Task taskB;
    void* paramsB[6] = {&d_Bpoints, &d_Apoints, &d_distance, &num_Bpoints, &num_Apoints, &distance_index};
    int pinfoB[6] = {iris_r, iris_r, iris_rw, sizeof(num_Bpoints), sizeof(num_Apoints), sizeof(distance_index)};
    taskB.kernel("computeDistance", 1, nullptr, const_cast<size_t*>(gwsB),
                 const_cast<size_t*>(lws), 6, paramsB, pinfoB);
    taskB.flush_out(d_distance);
    taskB.submit(iris_gpu, nullptr, 1);

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  std::printf("Average execution time of kernels: %f (ms)\n",
              (time * 1e-6f) / repeat);

  d_distance.fetch(h_distance);

  std::printf("Verifying the result may take a while..\n");
  float r_distance = hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
  float t_distance = std::max(h_distance[0], h_distance[1]);

  bool error = (std::fabs(t_distance - r_distance)) > 1e-3f;
  std::printf("%s\n", error ? "FAIL" : "PASS");

  platform.finalize();
  return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
