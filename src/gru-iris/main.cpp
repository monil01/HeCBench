#include <hip/hip_fp16.h>
#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static float sigmoidf_host(float in)
{
  return 1.0f / (1.0f + std::exp(-in));
}

static void reference(const std::vector<half>& input,
                      const std::vector<half>& hidden,
                      const std::vector<half>& bias1,
                      const std::vector<half>& bias2,
                      const std::vector<half>& hx,
                      std::vector<half>& hy,
                      std::vector<half>& storage,
                      int hsz,
                      int total_elements)
{
  for (int linear_idx = 0; linear_idx < total_elements; linear_idx++) {
    const int batch_idx = linear_idx / hsz;
    const int hidden_idx = linear_idx % hsz;
    const int input_offset = batch_idx * 3 * hsz + hidden_idx;

    const half ir = input[input_offset + 0 * hsz];
    const half ii = input[input_offset + 1 * hsz];
    const half in = input[input_offset + 2 * hsz];
    const half hr = hidden[input_offset + 0 * hsz];
    const half hi = hidden[input_offset + 1 * hsz];
    const half hn = hidden[input_offset + 2 * hsz];
    const half hxv = hx[linear_idx];

    const half b1r = bias1[hidden_idx + 0 * hsz];
    const half b1i = bias1[hidden_idx + 1 * hsz];
    const half b1n = bias1[hidden_idx + 2 * hsz];
    const half b2r = bias2[hidden_idx + 0 * hsz];
    const half b2i = bias2[hidden_idx + 1 * hsz];
    const half b2n = bias2[hidden_idx + 2 * hsz];

    const float rg = sigmoidf_host(static_cast<float>(ir) + static_cast<float>(hr) +
                                   static_cast<float>(b1r) + static_cast<float>(b2r));
    const float ig = sigmoidf_host(static_cast<float>(ii) + static_cast<float>(hi) +
                                   static_cast<float>(b1i) + static_cast<float>(b2i));
    float ng = static_cast<float>(in) + static_cast<float>(b1n) +
               rg * (static_cast<float>(hn) + static_cast<float>(b2n));
    ng = std::tanh(ng);

    hy[linear_idx] = static_cast<half>(ng + ig * (static_cast<float>(hxv) - ng));

    const int store_offset = batch_idx * 5 * hsz + hidden_idx;
    storage[store_offset + 0 * hsz] = static_cast<half>(rg);
    storage[store_offset + 1 * hsz] = static_cast<half>(ig);
    storage[store_offset + 2 * hsz] = static_cast<half>(ng);
    storage[store_offset + 3 * hsz] = hxv;
    storage[store_offset + 4 * hsz] =
        static_cast<half>(static_cast<float>(hn) + static_cast<float>(b2n));
  }
}

static void launch_gru(iris::DMem& input,
                       iris::DMem& hidden,
                       iris::DMem& bias1,
                       iris::DMem& bias2,
                       iris::DMem& hx,
                       iris::DMem& hy,
                       iris::DMem& storage,
                       int hsz,
                       int total_elements)
{
  size_t gws[1] = {((static_cast<size_t>(total_elements) + 255) / 256) * 256};
  size_t lws[1] = {256};
  void* params[] = {&input, &hidden, &bias1, &bias2, &hx, &hy, &storage, &hsz, &total_elements};
  int pinfo[] = {iris_r, iris_r, iris_r, iris_r, iris_r, iris_w, iris_w,
                 sizeof(hsz), sizeof(total_elements)};
  iris::Task task;
  task.kernel("gru_cell_forward", 1, nullptr, gws, lws, 9, params, pinfo);
  task.flush_out(hy);
  task.flush_out(storage);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of sequences> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int vsz = std::atoi(argv[1]);
  const int hsz = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  const int total_elements = vsz * hsz;
  const size_t input_size = 3 * static_cast<size_t>(total_elements);
  const size_t hidden_size = 3 * static_cast<size_t>(total_elements);
  const size_t bias_size = 3 * static_cast<size_t>(hsz);
  const size_t store_size = 5 * static_cast<size_t>(total_elements);
  const size_t state_size = static_cast<size_t>(total_elements);

  std::vector<half> h_input(input_size);
  std::vector<half> h_hidden(hidden_size);
  std::vector<half> h_input_bias(bias_size);
  std::vector<half> h_hidden_bias(bias_size);
  std::vector<half> h_hx(state_size);
  std::vector<half> h_hy(state_size);
  std::vector<half> h_store(store_size);
  std::vector<half> h_hy_ref(state_size);
  std::vector<half> h_store_ref(store_size);

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(-2.0f, 2.0f);
  for (size_t i = 0; i < input_size; i++) h_input[i] = distr(g);
  for (size_t i = 0; i < hidden_size; i++) h_hidden[i] = distr(g);
  for (size_t i = 0; i < bias_size; i++) {
    h_input_bias[i] = distr(g);
    h_hidden_bias[i] = distr(g);
  }
  for (size_t i = 0; i < state_size; i++) h_hx[i] = distr(g);

  reference(h_input, h_hidden, h_input_bias, h_hidden_bias,
            h_hx, h_hy_ref, h_store_ref, hsz, total_elements);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_input(h_input.data(), h_input.size() * sizeof(half));
  iris::DMem mem_hidden(h_hidden.data(), h_hidden.size() * sizeof(half));
  iris::DMem mem_input_bias(h_input_bias.data(), h_input_bias.size() * sizeof(half));
  iris::DMem mem_hidden_bias(h_hidden_bias.data(), h_hidden_bias.size() * sizeof(half));
  iris::DMem mem_hx(h_hx.data(), h_hx.size() * sizeof(half));
  iris::DMem mem_hy(h_hy.data(), h_hy.size() * sizeof(half));
  iris::DMem mem_store(h_store.data(), h_store.size() * sizeof(half));

  launch_gru(mem_input, mem_hidden, mem_input_bias, mem_hidden_bias,
             mem_hx, mem_hy, mem_store, hsz, total_elements);
  mem_hy.fetch(h_hy.data());
  mem_store.fetch(h_store.data());

  bool ok = true;
  for (size_t i = 0; i < state_size; i++) {
    if (std::fabs(static_cast<float>(h_hy[i]) - static_cast<float>(h_hy_ref[i])) > 1e-3f) {
      ok = false;
      break;
    }
  }
  for (size_t i = 0; i < store_size; i++) {
    if (std::fabs(static_cast<float>(h_store[i]) - static_cast<float>(h_store_ref[i])) > 1e-3f) {
      ok = false;
      break;
    }
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_gru(mem_input, mem_hidden, mem_input_bias, mem_hidden_bias,
               mem_hx, mem_hy, mem_store, hsz, total_elements);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of gru_cell_forward: %f (us)\n",
              (time * 1e-3f) / repeat);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 2;
}
