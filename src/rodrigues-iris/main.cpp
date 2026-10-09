#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
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

static constexpr int kBlockSize = 256;

static Float3 rotate_point(Float3 p, float angle, Float3 w) {
  float s;
  float c;
  sincosf(angle, &s, &c);

  const float mc = 1.f - c;
  const float m1 = c + (w.x) * (w.x) * (mc);
  const float m2 = (w.z) * s + (w.x) * (w.y) * (mc);
  const float m3 = -(w.y) * s + (w.x) * (w.z) * (mc);

  const float m4 = -(w.z) * s + (w.x) * (w.y) * (mc);
  const float m5 = c + (w.y) * (w.y) * (mc);
  const float m6 = (w.x) * s + (w.y) * (w.z) * (mc);

  const float m7 = (w.y) * s + (w.x) * (w.z) * (mc);
  const float m8 = -(w.x) * s + (w.y) * (w.z) * (mc);
  const float m9 = c + (w.z) * (w.z) * (mc);

  return {
    p.x * m1 + p.y * m2 + p.z * m3,
    p.x * m4 + p.y * m5 + p.z * m6,
    p.x * m7 + p.y * m8 + p.z * m9
  };
}

static int verify_float3(const Float3* out, const Float3* input, int n, int repeat, float angle, Float3 w) {
  double checksum = 0.0;
  double reference_checksum = 0.0;
  float max_abs_err = 0.0f;

  for (int i = 0; i < n; i++) {
    Float3 ref = input[i];
    for (int r = 0; r < repeat; r++) {
      ref = rotate_point(ref, angle, w);
    }
    const float ex = fabsf(out[i].x - ref.x);
    const float ey = fabsf(out[i].y - ref.y);
    const float ez = fabsf(out[i].z - ref.z);
    max_abs_err = fmaxf(max_abs_err, fmaxf(ex, fmaxf(ey, ez)));
    checksum += out[i].x + out[i].y + out[i].z;
    reference_checksum += ref.x + ref.y + ref.z;
  }

  std::printf("Checksum float3: %.9e reference %.9e max_abs_err %.9e\n",
              checksum, reference_checksum, max_abs_err);
  const int pass = max_abs_err <= 2.0e-4f;
  std::printf("float3: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

static int verify_float4(const Float4* out, const Float4* input, int n, int repeat, float angle, Float3 w) {
  double checksum = 0.0;
  double reference_checksum = 0.0;
  float max_abs_err = 0.0f;

  for (int i = 0; i < n; i++) {
    Float3 ref = { input[i].x, input[i].y, input[i].z };
    for (int r = 0; r < repeat; r++) {
      ref = rotate_point(ref, angle, w);
    }
    const float ex = fabsf(out[i].x - ref.x);
    const float ey = fabsf(out[i].y - ref.y);
    const float ez = fabsf(out[i].z - ref.z);
    const float ew = fabsf(out[i].w);
    max_abs_err = fmaxf(max_abs_err, fmaxf(fmaxf(ex, ey), fmaxf(ez, ew)));
    checksum += out[i].x + out[i].y + out[i].z + out[i].w;
    reference_checksum += ref.x + ref.y + ref.z;
  }

  std::printf("Checksum float4: %.9e reference %.9e max_abs_err %.9e\n",
              checksum, reference_checksum, max_abs_err);
  const int pass = max_abs_err <= 2.0e-4f;
  std::printf("float4: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);

  const float wx = -0.3f;
  const float wy = -0.6f;
  const float wz = 0.15f;
  const float norm = 1.f / sqrtf(wx * wx + wy * wy + wz * wz);
  const Float3 w = { wx * norm, wy * norm, wz * norm };
  const float angle = 0.5f;

  auto* h = static_cast<Float3*>(std::malloc(sizeof(Float3) * static_cast<size_t>(n)));
  auto* h2 = static_cast<Float4*>(std::malloc(sizeof(Float4) * static_cast<size_t>(n)));
  auto* input = static_cast<Float3*>(std::malloc(sizeof(Float3) * static_cast<size_t>(n)));
  auto* input2 = static_cast<Float4*>(std::malloc(sizeof(Float4) * static_cast<size_t>(n)));
  if (!h || !h2 || !input || !input2) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h);
    std::free(h2);
    std::free(input);
    std::free(input2);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < n; i++) {
    float a = std::rand();
    float b = std::rand();
    float c = std::rand();
    float d = sqrtf(a * a + b * b + c * c);
    h[i] = { a / d, b / d, c / d };
    h2[i] = { a / d, b / d, c / d, 0.f };
    input[i] = h[i];
    input2[i] = h2[i];
  }

  iris::DMem mem_h(h, sizeof(Float3) * static_cast<size_t>(n));
  iris::DMem mem_h2(h2, sizeof(Float4) * static_cast<size_t>(n));

  int n_arg = n;
  float angle_arg = angle;
  Float3 w_arg = w;

  const size_t blocks = (static_cast<size_t>(n) + kBlockSize - 1) / kBlockSize;
  size_t gws[1] = { blocks * kBlockSize };
  size_t lws[1] = { kBlockSize };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = { &n_arg, &angle_arg, &w_arg, &mem_h };
    int pinfo[4] = { sizeof(n), sizeof(angle), sizeof(w), iris_rw };
    task.kernel("rotate", 1, nullptr, gws, lws, 4, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  mem_h.fetch(h);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (float3): %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = { &n_arg, &angle_arg, &w_arg, &mem_h2 };
    int pinfo[4] = { sizeof(n), sizeof(angle), sizeof(w), iris_rw };
    task.kernel("rotate2", 1, nullptr, gws, lws, 4, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  mem_h2.fetch(h2);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (float4): %f (us)\n", (time * 1e-3f) / repeat);

  int errors = 0;
  errors += verify_float3(h, input, n, repeat, angle, w);
  errors += verify_float4(h2, input2, n, repeat, angle, w);

  std::free(h);
  std::free(h2);
  std::free(input);
  std::free(input2);
  platform.finalize();
  return errors == 0 ? 0 : 1;
}
