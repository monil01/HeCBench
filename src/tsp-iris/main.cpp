#include <iris/iris.hpp>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int best_thread_count(int cities)
{
  int max = cities - 2;
  if (max > 256) max = 256;
  int best = 0;
  int bthr = 4;
  for (int threads = 1; threads <= max; threads++) {
    int smem = static_cast<int>(sizeof(int)) * threads +
               2 * static_cast<int>(sizeof(float)) * 128 +
               static_cast<int>(sizeof(int)) * 128;
    int blocks = (16384 * 2) / smem;
    if (blocks > 16) blocks = 16;
    int thr = (threads + 31) / 32 * 32;
    while (blocks * thr > 2048) blocks--;
    int perf = threads * blocks;
    if (perf > best) {
      best = perf;
      bthr = threads;
    }
  }
  return bthr;
}

int main(int argc, char *argv[])
{
  std::printf("2-opt TSP IRIS HIP GPU code v2.3\n");
  std::printf("Copyright (c) 2014-2020, Texas State University. All rights reserved.\n");

  if (argc != 4) {
    std::fprintf(stderr, "\narguments: <input_file> <restart_count> <repeat>\n");
    return 1;
  }

  FILE *f = std::fopen(argv[1], "rt");
  if (f == nullptr) {
    std::fprintf(stderr, "could not open file %s\n", argv[1]);
    return 1;
  }

  int restarts = std::atoi(argv[2]);
  if (restarts < 1) {
    std::fprintf(stderr, "restart_count is too small: %d\n", restarts);
    std::fclose(f);
    return 1;
  }

  int repeat = std::atoi(argv[3]);

  int ch = std::getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = std::getc(f);
  ch = std::getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = std::getc(f);
  ch = std::getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = std::getc(f);
  ch = std::getc(f);
  while ((ch != EOF) && (ch != ':')) ch = std::getc(f);

  char str[256];
  std::fscanf(f, "%255s\n", str);

  int cities = std::atoi(str);
  if (cities < 100) {
    std::fprintf(stderr, "the problem size must be at least 100 for this version of the code\n");
    std::fclose(f);
    return 1;
  }

  ch = std::getc(f);
  while ((ch != EOF) && (ch != '\n')) ch = std::getc(f);
  std::fscanf(f, "%255s\n", str);
  if (std::strcmp(str, "NODE_COORD_SECTION") != 0) {
    std::fprintf(stderr, "wrong file format\n");
    std::fclose(f);
    return 1;
  }

  float *posx = static_cast<float *>(std::malloc(sizeof(float) * cities));
  float *posy = static_cast<float *>(std::malloc(sizeof(float) * cities));
  if (posx == nullptr || posy == nullptr) {
    std::fprintf(stderr, "cannot allocate city coordinate buffers\n");
    std::free(posx);
    std::free(posy);
    std::fclose(f);
    return 1;
  }

  int cnt = 0;
  int in1 = 0;
  float in2 = 0.0f;
  float in3 = 0.0f;
  while (std::fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    posx[cnt] = in2;
    posy[cnt] = in3;
    cnt++;
    if (cnt > cities) std::fprintf(stderr, "input too long\n");
    if (cnt != in1) std::fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);
  }
  if (cnt != cities) std::fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);

  std::fscanf(f, "%255s", str);
  if (std::strcmp(str, "EOF") != 0) std::fprintf(stderr, "didn't see 'EOF' at end of file\n");
  std::fclose(f);

  std::printf("configuration: %d cities, %d restarts, %s input\n", cities, restarts, argv[1]);

  int climbs = 0;
  int best = INT_MAX;
  const int glob_words = restarts * ((3 * cities + 2 + 31) / 32 * 32);
  int *glob = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(glob_words)));
  if (glob == nullptr) {
    std::fprintf(stderr, "cannot allocate global scratch buffer\n");
    std::free(posx);
    std::free(posy);
    return 1;
  }

  int threads = best_thread_count(cities);
  std::printf("thread block size: %d\n", threads);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_posx(posx, sizeof(float) * static_cast<size_t>(cities));
  iris::DMem mem_posy(posy, sizeof(float) * static_cast<size_t>(cities));
  iris::DMem mem_glob(glob, sizeof(int) * static_cast<size_t>(glob_words));
  iris::DMem mem_climbs(&climbs, sizeof(int));
  iris::DMem mem_best(&best, sizeof(int));

  double ktime = 0.0;
  for (int i = 0; i <= repeat; i++) {
    climbs = 0;
    best = INT_MAX;
    mem_climbs.update(&climbs);
    mem_best.update(&best);

    size_t gws[1] = {static_cast<size_t>(restarts) * static_cast<size_t>(threads)};
    size_t lws[1] = {static_cast<size_t>(threads)};
    void *params[6] = {&cities, &mem_posx, &mem_posy, &mem_glob, &mem_climbs, &mem_best};
    int pinfo[6] = {sizeof(cities), iris_r, iris_r, iris_rw, iris_rw, iris_rw};

    auto kstart = std::chrono::steady_clock::now();
    iris::Task task;
    task.kernel("TwoOpt", 1, nullptr, gws, lws, 6, params, pinfo);
    task.flush_out(mem_best);
    task.flush_out(mem_climbs);
    task.submit(iris_gpu, nullptr, 1);
    auto kend = std::chrono::steady_clock::now();
    if (i > 0) {
      ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    }
  }

  mem_best.fetch(&best);
  mem_climbs.fetch(&climbs);
  platform.finalize();

  long long moves = 1LL * climbs * (cities - 2) * (cities - 1) / 2;

  std::printf("Average kernel time: %.4f s\n", ktime * 1e-9f / repeat);
  std::printf("%.3f Gmoves/s\n", moves * repeat / ktime);
  std::printf("Best found tour length is %d with %d climbers\n", best, climbs);

  if (best < 38000 && best >= 35002)
    std::printf("PASS\n");
  else
    std::printf("FAIL\n");

  std::free(glob);
  std::free(posx);
  std::free(posy);
  return (best < 38000 && best >= 35002) ? 0 : 1;
}
