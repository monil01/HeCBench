#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

#define BLOCK_SIZE 256

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <neurons per item> <num_items> <num_steps>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int neurons_per_item = std::atoi(argv[1]);
  int num_items = std::atoi(argv[2]);
  int num_steps = std::atoi(argv[3]);
  size_t num_neurons = static_cast<size_t>(neurons_per_item) * static_cast<size_t>(num_items);

  float dt = 0.1f;
  float tau_rc = 10.f;
  float tau_ref = 2.f;

  std::vector<float> encode_result(num_items);
  std::vector<float> bias(neurons_per_item);
  std::vector<float> gain(neurons_per_item);
  std::vector<float> voltage(num_neurons);
  std::vector<float> reftime(num_neurons);
  std::vector<float> spikes(num_neurons);
  std::vector<float> voltage_host(num_neurons);
  std::vector<float> reftime_host(num_neurons);
  std::vector<float> spikes_host(num_neurons);

  std::srand(123);
  for (int i = 0; i < num_items; i++) {
    encode_result[i] = std::rand() / static_cast<float>(RAND_MAX);
  }
  for (size_t i = 0; i < num_neurons; i++) {
    voltage_host[i] = voltage[i] = 1.f + std::rand() / static_cast<float>(RAND_MAX);
    reftime_host[i] = reftime[i] = std::rand() % 5 / 10.f;
  }
  for (int i = 0; i < neurons_per_item; i++) {
    bias[i] = std::rand() / static_cast<float>(RAND_MAX);
    gain[i] = std::rand() / static_cast<float>(RAND_MAX) + 0.5f;
  }

  iris::DMem mem_encode(encode_result.data(), encode_result.size() * sizeof(float));
  iris::DMem mem_bias(bias.data(), bias.size() * sizeof(float));
  iris::DMem mem_gain(gain.data(), gain.size() * sizeof(float));
  iris::DMem mem_voltage(voltage.data(), voltage.size() * sizeof(float));
  iris::DMem mem_reftime(reftime.data(), reftime.size() * sizeof(float));
  iris::DMem mem_spikes(spikes.data(), spikes.size() * sizeof(float));

  const size_t blocks = (num_neurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[12] = {
    &num_neurons,
    &neurons_per_item,
    &num_steps,
    &dt,
    &mem_encode,
    &mem_voltage,
    &mem_reftime,
    &tau_rc,
    &tau_ref,
    &mem_bias,
    &mem_gain,
    &mem_spikes
  };
  int pinfo[12] = {
    sizeof(num_neurons),
    sizeof(neurons_per_item),
    sizeof(num_steps),
    sizeof(dt),
    iris_r,
    iris_rw,
    iris_rw,
    sizeof(tau_rc),
    sizeof(tau_ref),
    iris_r,
    iris_r,
    iris_w
  };
  task.kernel("lif_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 12, params, pinfo);
  task.flush_out(mem_spikes);
  task.flush_out(mem_voltage);
  task.flush_out(mem_reftime);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  mem_spikes.fetch(spikes.data());
  mem_voltage.fetch(voltage.data());
  mem_reftime.fetch(reftime.data());

  std::printf("Average kernel execution time: %f (us)\n", (elapsed_time * 1e-3) / num_steps);

  for (int step = 0; step < num_steps; step++) {
    reference(num_neurons,
              neurons_per_item,
              dt,
              encode_result.data(),
              voltage_host.data(),
              reftime_host.data(),
              tau_rc,
              tau_ref,
              bias.data(),
              gain.data(),
              spikes_host.data());
  }

  size_t num_spikes = 0;
  size_t num_spikes_host = 0;
  std::vector<float> reftime_on_spike;
  std::vector<float> reftime_on_spike_host;
  for (size_t i = 0; i < num_neurons; i++) {
    if (spikes[i] == 1.f / dt) {
      num_spikes++;
      reftime_on_spike.push_back(reftime[i]);
    }
    if (spikes_host[i] == 1.f / dt) {
      num_spikes_host++;
      reftime_on_spike_host.push_back(reftime_host[i]);
    }
  }

  std::printf("Number of spikes on host and device: %zu %zu\n", num_spikes_host, num_spikes);
  num_spikes = std::min(num_spikes, num_spikes_host);
  bool ok = true;
  for (size_t i = 0; i < num_spikes; i++) {
    if (std::fabs(reftime_on_spike[i] - reftime_on_spike_host[i]) > 0.1f) {
      std::printf("%f %f\n", reftime_on_spike[i], reftime_on_spike_host[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 2;
}
