#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_DETECTIONS 4096
#define N_PARTITIONS 32

struct Float4 {
  float x;
  float y;
  float z;
  float w;
};

static void print_help()
{
  std::printf("\nUsage: nmstest <detections.txt> <output.txt> <repeat>\n\n");
  std::printf("               detections.txt -> Input file containing detected objects\n");
  std::printf("               output.txt     -> Output file after performing NMS\n");
  std::printf("               repeat         -> Kernel execution count\n\n");
}

static int get_optimal_dim(int val)
{
  int div = 16;
  int cntneg = div;
  int cntpos = div;
  int neg = 1;
  for (int i = 0; i < 5; i++) {
    if (val % div == 0) return div;
    if (neg) {
      cntneg--;
      div = cntneg;
      neg = 0;
    } else {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }
  return 16;
}

static int get_upper_limit(int val, int mul)
{
  int cnt = mul;
  while (cnt < val) cnt += mul;
  return std::min(cnt, MAX_DETECTIONS);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    print_help();
    return 0;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  FILE* fp = std::fopen(argv[1], "r");
  if (!fp) {
    std::printf("Error: Unable to open file %s for input detection coordinates.\n", argv[1]);
    platform.finalize();
    return 1;
  }

  Float4* cpu_points = static_cast<Float4*>(std::malloc(sizeof(Float4) * MAX_DETECTIONS));
  unsigned char* cpu_pointsbitmap =
      static_cast<unsigned char*>(std::malloc(sizeof(unsigned char) * MAX_DETECTIONS));
  unsigned char* cpu_nmsbitmap = static_cast<unsigned char*>(
      std::malloc(sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS));
  if (!cpu_points || !cpu_pointsbitmap || !cpu_nmsbitmap) {
    std::printf("Error: Unable to allocate CPU memory.\n");
    std::fclose(fp);
    std::free(cpu_nmsbitmap);
    std::free(cpu_pointsbitmap);
    std::free(cpu_points);
    platform.finalize();
    return 1;
  }

  std::memset(cpu_points, 0, sizeof(Float4) * MAX_DETECTIONS);
  std::memset(cpu_pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);
  std::memset(cpu_nmsbitmap, 1, sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);

  int x, y, w;
  float score;
  int ndetections = 0;
  while (!std::feof(fp)) {
    int cnt = std::fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);
    if (cnt != 4) {
      std::printf("Error: Invalid file format in line %d when reading %s\n", ndetections,
                  argv[1]);
      std::fclose(fp);
      platform.finalize();
      return 1;
    }
    if (ndetections >= MAX_DETECTIONS) break;
    cpu_points[ndetections].x = static_cast<float>(x);
    cpu_points[ndetections].y = static_cast<float>(y);
    cpu_points[ndetections].z = static_cast<float>(w);
    cpu_points[ndetections].w = score;
    ndetections++;
  }
  std::fclose(fp);
  std::printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  iris::DMem points(cpu_points, sizeof(Float4) * MAX_DETECTIONS);
  iris::DMem nmsbitmap(cpu_nmsbitmap,
                       sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);
  iris::DMem pointsbitmap(cpu_pointsbitmap, sizeof(unsigned char) * MAX_DETECTIONS);

  const int repeat = std::atoi(argv[3]);
  const int limit = get_upper_limit(ndetections, 16);
  const int pkthreads_x = get_optimal_dim(limit);
  const int pkthreads_y = get_optimal_dim(limit);
  const size_t map_gws[2] = {static_cast<size_t>(limit), static_cast<size_t>(limit)};
  const size_t map_lws[2] = {static_cast<size_t>(pkthreads_x),
                             static_cast<size_t>(pkthreads_y)};
  float threshold = 0.3f;

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[3] = {&points, &nmsbitmap, &threshold};
    int pinfo[3] = {iris_r, iris_rw, sizeof(threshold)};
    task.kernel("generate_nms_bitmap", 2, nullptr, const_cast<size_t*>(map_gws),
                const_cast<size_t*>(map_lws), 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n",
              (elapsed * 1e-9f) / repeat);

  const size_t reduce_gws[1] = {static_cast<size_t>(ndetections * (MAX_DETECTIONS / N_PARTITIONS))};
  const size_t reduce_lws[1] = {MAX_DETECTIONS / N_PARTITIONS};
  start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[3] = {&nmsbitmap, &pointsbitmap, &ndetections};
    int pinfo[3] = {iris_r, iris_w, sizeof(ndetections)};
    task.kernel("reduce_nms_bitmap", 1, nullptr, const_cast<size_t*>(reduce_gws),
                const_cast<size_t*>(reduce_lws), 3, params, pinfo);
    if (n == repeat - 1) task.flush_out(pointsbitmap);
    task.submit(iris_gpu, nullptr, 1);
  }
  end = std::chrono::steady_clock::now();
  elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n",
              (elapsed * 1e-9f) / repeat);

  fp = std::fopen(argv[2], "w");
  if (!fp) {
    std::printf("Error: Unable to open file %s for detection outcome.\n", argv[2]);
    platform.finalize();
    return 1;
  }
  int totaldets = 0;
  for (int i = 0; i < ndetections; i++) {
    if (cpu_pointsbitmap[i]) {
      std::fprintf(fp, "%d,%d,%d,%f\n", static_cast<int>(cpu_points[i].x),
                   static_cast<int>(cpu_points[i].y), static_cast<int>(cpu_points[i].z),
                   cpu_points[i].w);
      totaldets++;
    }
  }
  std::fclose(fp);
  std::printf("Detections after NMS: %d\n", totaldets);

  std::free(cpu_nmsbitmap);
  std::free(cpu_pointsbitmap);
  std::free(cpu_points);
  platform.finalize();
  return 0;
}
