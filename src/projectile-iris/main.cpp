#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Projectile.hpp"
#include "reference.hpp"

static const int num_elements = 10000000;
const int BLOCK_SIZE = 256;

void GpuParallel(std::vector<Projectile>& in_vect,
                 std::vector<Projectile>& out_vect, const int repeat) {
  iris::DMem mem_in(in_vect.data(), sizeof(Projectile) * num_elements);
  iris::DMem mem_out(out_vect.data(), sizeof(Projectile) * num_elements);
  int n = num_elements;
  size_t gws =
      static_cast<size_t>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE) *
      BLOCK_SIZE;
  size_t lws = BLOCK_SIZE;

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    iris::Task task;
    void* params[3] = {&mem_in, &mem_out, &n};
    int pinfo[3] = {iris_r, iris_w, sizeof(n)};
    task.kernel("CalculateRange", 1, nullptr, &gws, &lws, 3, params, pinfo);
    if (iter == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (time * 1e-9f) / repeat);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  float init_angle = 0.0f;
  float init_vel = 0.0f;
  std::vector<Projectile> input_vect;
  std::vector<Projectile> out_parallel_vect;
  std::vector<Projectile> out_scalar_vect;
  input_vect.reserve(num_elements);
  out_parallel_vect.reserve(num_elements);
  out_scalar_vect.reserve(num_elements);

  std::srand(2);
  for (int i = 0; i < num_elements; i++) {
    init_angle = std::rand() % 90 + 10;
    init_vel = std::rand() % 400 + 10;
    input_vect.push_back(Projectile(init_angle, init_vel, 1.0f, 1.0f, 1.0f));
    out_parallel_vect.push_back(Projectile());
    out_scalar_vect.push_back(Projectile());
  }

  GpuParallel(input_vect, out_parallel_vect, repeat);

  reference(input_vect.data(), out_scalar_vect.data(), num_elements);

  bool ok = true;
  for (int i = 0; i < num_elements; i++) {
    if (out_parallel_vect[i] != out_scalar_vect[i]) {
      ok = false;
      std::cout << out_parallel_vect[i] << std::endl;
      std::cout << out_scalar_vect[i] << std::endl;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return 0;
}
