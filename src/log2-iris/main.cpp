#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void log2_approx(
    std::vector<float>& inputs,
    std::vector<float>& outputs,
    std::vector<int>& precision,
    int num_inputs,
    int precision_count,
    int repeat)
{
  iris::DMem mem_inputs(inputs.data(), static_cast<size_t>(num_inputs) * sizeof(float));
  iris::DMem mem_outputs(outputs.data(), static_cast<size_t>(num_inputs) * precision_count * sizeof(float));

  const size_t gws[1] = { static_cast<size_t>((num_inputs + 255) / 256) * 256 };
  const size_t lws[1] = { 256 };

  for (int i = 0; i < precision_count; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < repeat; ++k) {
      iris::Task task;
      void* params[5] = {
        &mem_outputs,
        &mem_inputs,
        &i,
        &num_inputs,
        &precision[i]
      };
      int pinfo[5] = {
        iris_w,
        iris_r,
        sizeof(i),
        sizeof(num_inputs),
        sizeof(precision[i])
      };
      task.kernel("compute_log", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
      task.flush_out(mem_outputs);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nIterative approximation with " << precision[i] << " bits of precision\n";
    std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
  }

  mem_outputs.fetch(outputs.data());
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cout << "Usage: ./main <config filename>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::ifstream message_file(argv[1]);
  if (!message_file) {
    std::cerr << "Unable to open config file: " << argv[1] << "\n";
    platform.finalize();
    return 1;
  }

  std::string placeholder;
  message_file >> placeholder;
  long ceilingVal;
  message_file >> ceilingVal;

  message_file >> placeholder;
  int repeat;
  message_file >> repeat;

  message_file >> placeholder;
  int precision_count;
  message_file >> precision_count;

  std::vector<int> precision(precision_count, 0);
  message_file >> placeholder;
  for (int i = 0; i < precision_count; ++i) {
    message_file >> precision[i];
  }

  std::vector<float> inputs;
  for (long i = 1; i <= ceilingVal; ++i) {
    inputs.push_back(static_cast<float>(i));
  }

  const size_t inputs_size = inputs.size();
  std::cout << "Number of precision counts : " << precision_count << std::endl
            << " Number of inputs to evaluate for each precision: " << inputs_size << std::endl
            << " Number of runs for each precision : " << repeat << std::endl;

  std::vector<float> d_output_vals(inputs_size * precision_count);
  log2_approx(inputs, d_output_vals, precision, static_cast<int>(inputs.size()), precision_count, repeat);

  std::vector<float> ref_vals(inputs_size, 0.0f);
  for (size_t i = 0; i < inputs_size; ++i) {
    ref_vals[i] = log2f(inputs[i]);
  }

  std::cout << "-------------- SUMMARY (Device results): --------------" << std::endl << std::endl;
  for (int i = 0; i < precision_count; ++i) {
    std::cout << "----- Iterative approximation with " << precision[i] << " bits of precision -----" << std::endl;
    float s = 0.0f;
    for (size_t j = 0; j < inputs_size; ++j) {
      const float err = d_output_vals[i * inputs_size + j] - ref_vals[j];
      s += err * err;
    }
    s /= inputs.size();
    std::cout << "RMSE : " << sqrtf(s) << std::endl;
  }

  platform.finalize();
  return 0;
}
