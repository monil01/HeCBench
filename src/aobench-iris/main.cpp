#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define WIDTH 256
#define HEIGHT 256
#define NSUBSAMPLES 2
#define BLOCK_SIZE 16

struct vec {
  float x;
  float y;
  float z;
};

struct Sphere {
  vec center;
  float radius;
};

struct Plane {
  vec p;
  vec n;
};

static void init_scene(Sphere* spheres, Plane& plane)
{
  spheres[0].center.x = -2.0f;
  spheres[0].center.y = 0.0f;
  spheres[0].center.z = -3.5f;
  spheres[0].radius = 0.5f;

  spheres[1].center.x = -0.5f;
  spheres[1].center.y = 0.0f;
  spheres[1].center.z = -3.0f;
  spheres[1].radius = 0.5f;

  spheres[2].center.x = 1.0f;
  spheres[2].center.y = 0.0f;
  spheres[2].center.z = -2.2f;
  spheres[2].radius = 0.5f;

  plane.p.x = 0.0f;
  plane.p.y = -0.5f;
  plane.p.z = 0.0f;

  plane.n.x = 0.0f;
  plane.n.y = 1.0f;
  plane.n.z = 0.0f;
}

static void saveppm(const char* fname, int w, int h, unsigned char* img)
{
  FILE* fp = std::fopen(fname, "wb");
  if (!fp) {
    std::printf("Failed to open the file %s\n", fname);
    return;
  }
  std::fprintf(fp, "P6\n");
  std::fprintf(fp, "%d %d\n", w, h);
  std::fprintf(fp, "255\n");
  std::fwrite(img, w * h * 3, 1, fp);
  std::fclose(fp);
}

static long render(iris::DMem& mem_img, iris::DMem& mem_spheres, Plane& plane,
                   int w, int h, int nsubsamples, bool flush)
{
  const size_t gws[2] = {
      static_cast<size_t>(((w + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE),
      static_cast<size_t>(((h + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  const size_t lws[2] = {BLOCK_SIZE, BLOCK_SIZE};
  void* params[6] = {&mem_img, &mem_spheres, &plane, &h, &w, &nsubsamples};
  int pinfo[6] = {iris_w, iris_r, sizeof(plane), sizeof(h), sizeof(w), sizeof(nsubsamples)};

  iris::Task task;
  task.kernel("render_kernel", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
              6, params, pinfo);
  if (flush) task.flush_out(mem_img);

  auto start = std::chrono::steady_clock::now();
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int loopmax = std::atoi(argv[1]);
  Sphere spheres[3];
  Plane plane;
  init_scene(spheres, plane);

  const size_t image_size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
  unsigned char* img = static_cast<unsigned char*>(std::malloc(image_size));
  if (!img) return 1;

  iris::DMem mem_img(img, image_size);
  iris::DMem mem_spheres(spheres, sizeof(Sphere) * 3);

  long time = 0;
  for (int i = 0; i < loopmax; ++i) {
    time += render(mem_img, mem_spheres, plane, WIDTH, HEIGHT, NSUBSAMPLES, i == loopmax - 1);
  }
  std::printf("Average kernel time: %lf usec.\n", static_cast<double>(time) / (1e3 * loopmax));

  saveppm("ao.ppm", WIDTH, HEIGHT, img);
  std::free(img);
  platform.finalize();
  return 0;
}
