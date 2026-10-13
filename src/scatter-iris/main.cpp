#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

enum ReductionType { SUM, MUL, DIV, MIN, MAX };
template<class T> T initial(int op) {
  if (op == MUL || op == DIV) return 1;
  if (op == MIN) return std::numeric_limits<T>::max();
  if (op == MAX) return std::numeric_limits<T>::lowest();
  return 0;
}
template<class T> void update(T& x, T y, int op) {
  if (op == SUM) x += y;
  else if (op == MUL) x *= y;
  else if (op == DIV) x /= y;
  else if (op == MIN) x = std::min(x, y);
  else x = std::max(x, y);
}
template<class T> bool scatter(int64_t count, int repeat, int op, const char* type) {
  int numel = static_cast<int>(count);
  size_t out_size = count / 2;
  std::vector<int64_t> idx(count);
  std::vector<T> src(count), out(out_size, initial<T>(op)), ref(out);
  srand(123);
  for (int64_t i = 0; i < count; ++i) {
    idx[i] = rand() % out_size;
    src[i] = rand() % 2 ? 1 : -1;
  }
  iris::DMem mi(idx.data(), idx.size() * sizeof(int64_t));
  iris::DMem ms(src.data(), src.size() * sizeof(T));
  iris::DMem mo(out.data(), out.size() * sizeof(T));
  const char* names[] = {"sum", "mul", "div", "min", "max"};
  std::string kernel = std::string("scatter_") + type + "_" + names[op];
  size_t gws[] = {(static_cast<size_t>(count) + 255) / 256 * 256};
  size_t lws[] = {256};
  int64_t ns = 0;
  for (int r = 0; r < repeat; ++r) {
    std::fill(out.begin(), out.end(), initial<T>(op));
    mo.update(out.data());
    iris::Task task;
    void* args[] = {&ms, &mi, &mo, &numel};
    int info[] = {iris_r, iris_r, iris_rw, sizeof(numel)};
    auto start = std::chrono::steady_clock::now();
    if (task.kernel(kernel.c_str(), 1, nullptr, gws, lws, 4, args, info) != IRIS_SUCCESS ||
        task.flush_out(mo) != IRIS_SUCCESS || task.submit(iris_gpu, nullptr, 1) != IRIS_SUCCESS)
      return false;
    ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  }
  printf("Average execution time of kernel: %f (us)\n", ns * 1e-3 / repeat);
  for (int64_t i = 0; i < count; ++i) update(ref[idx[i]], src[i], op);
  bool native_ok = true, full_ok = true;
  for (size_t i = 0; i < out_size; ++i) {
    if (double(out[i]) - double(ref[i]) > 1e-3) native_ok = false;
    if (out[i] != ref[i]) full_ok = false;
  }
  printf("%s\n\n", native_ok ? "PASS" : "FAIL");
  printf("FULL_CHECK %s %s elements=%zu %s\n", type, names[op], out_size, full_ok ? "PASS" : "FAIL");
  return native_ok && full_ok;
}
template<class T> bool cases(int64_t n, int r, const char* type, const char* title) {
  printf("%s scatter (mul, div, sum, min, max)\n", title);
  bool ok = true;
  for (int op : {MUL, DIV, SUM, MIN, MAX}) ok = scatter<T>(n, r, op, type) && ok;
  return ok;
}
int main(int argc, char** argv) {
  if (argc != 3) { printf("Usage: %s <number of elements> <repeat>\n", argv[0]); return 1; }
  int64_t n = atol(argv[1]); int r = atoi(argv[2]);
  if (n < 2 || n > std::numeric_limits<int>::max() || r < 1) return 1;
  iris::Platform platform;
  if (platform.init(&argc, &argv, 1) != IRIS_SUCCESS) return 1;
  bool gpu = false;
  for (int d = 0; d < platform.ndevs(); ++d) {
    int backend = 0, type = 0; char name[256] = {}; size_t bytes = 0;
    iris_device_info(d, iris_backend, &backend, &bytes);
    iris_device_info(d, iris_type, &type, &bytes);
    iris_device_info(d, iris_name, name, &bytes);
    printf("DEVICE backend=%d type=%d name=%s\n", backend, type, name);
    gpu |= backend == 3 && type == 256;
  }
  if (!gpu) return 1;
  bool ok = cases<int32_t>(n, r, "i32", "INT32");
  ok = cases<int64_t>(n, r, "i64", "INT64") && ok;
  ok = cases<float>(n, r, "f32", "FP32") && ok;
  ok = cases<double>(n, r, "f64", "FP64") && ok;
  int errors = platform.error_count(); platform.finalize();
  return ok && !errors ? 0 : 1;
}
