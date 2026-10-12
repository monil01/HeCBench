#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void reference(int nthreads,
                      const float* input_data,
                      const float* output_data,
                      const float* output_grad,
                      int channels,
                      int input_height,
                      int input_width,
                      int output_height,
                      int output_width,
                      int ksize_height,
                      int ksize_width,
                      int stride_height,
                      int stride_width,
                      int padding_height,
                      int padding_width,
                      bool exclusive,
                      float* input_grad,
                      bool channel_last)
{
  for (int index = 0; index < nthreads; index++) {
    int w_offset, h_offset, offset_c, batch_idx;
    if (!channel_last) {
      w_offset = index % input_width + padding_width;
      h_offset = (index / input_width) % input_height + padding_height;
      offset_c = (index / input_width / input_height) % channels;
      batch_idx = index / input_width / input_height / channels;
    } else {
      offset_c = index % channels;
      w_offset = (index / channels) % input_width + padding_width;
      h_offset = (index / channels / input_width) % input_height + padding_height;
      batch_idx = index / channels / input_width / input_height;
    }

    const int phstart = (h_offset < ksize_height) ? 0 : (h_offset - ksize_height) / stride_height + 1;
    const int pwstart = (w_offset < ksize_width) ? 0 : (w_offset - ksize_width) / stride_width + 1;
    const int phend = std::min(h_offset / stride_height + 1, output_height);
    const int pwend = std::min(w_offset / stride_width + 1, output_width);

    float gradient = 0.0f;
    const float input = input_data[index];
    (void)input;

    int output_stride;
    if (!channel_last) {
      output_stride = (batch_idx * channels + offset_c) * output_height * output_width;
    } else {
      output_stride = batch_idx * output_height * output_width * channels;
    }

    const float* output_data_t = output_data + output_stride;
    const float* output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = std::min(hstart + ksize_height, input_height);
        int wend = std::min(wstart + ksize_width, input_width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        const int pool_size = exclusive ? (hend - hstart) * (wend - wstart)
                                        : ksize_height * ksize_width;

        const int output_sub_idx = channel_last
            ? (ph * output_width + pw) * channels + offset_c
            : ph * output_width + pw;
        (void)output_data_t;
        gradient += output_grad_t[output_sub_idx] * (1.0f / static_cast<float>(pool_size));
      }
    }
    input_grad[index] = gradient;
  }
}

static void launch_pool(iris::DMem& input,
                        iris::DMem& output,
                        iris::DMem& output_grad,
                        iris::DMem& input_grad,
                        int nthreads,
                        int channels,
                        int input_height,
                        int input_width,
                        int output_height,
                        int output_width,
                        int channel_last)
{
  size_t gws[1] = {((static_cast<size_t>(nthreads) + 255) / 256) * 256};
  size_t lws[1] = {256};
  void* params[] = {&nthreads, &input, &output, &output_grad, &channels, &input_height,
                    &input_width, &output_height, &output_width, &input_grad, &channel_last};
  int pinfo[] = {sizeof(nthreads), iris_r, iris_r, iris_r, sizeof(channels),
                 sizeof(input_height), sizeof(input_width), sizeof(output_height),
                 sizeof(output_width), iris_w, sizeof(channel_last)};
  iris::Task task;
  task.kernel("pool2d_grad_avg", 1, nullptr, gws, lws, 11, params, pinfo);
  task.flush_out(input_grad);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 8) {
    std::printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
    std::printf("<input width> <output height> <output width> <repeat>\n");
    return 1;
  }

  const int batch_size = std::atoi(argv[1]);
  const int input_channels = std::atoi(argv[2]);
  const int input_height = std::atoi(argv[3]);
  const int input_width = std::atoi(argv[4]);
  const int output_height = std::atoi(argv[5]);
  const int output_width = std::atoi(argv[6]);
  const int repeat = std::atoi(argv[7]);

  const int input_numel = batch_size * input_channels * input_height * input_width;
  const int output_numel = batch_size * input_channels * output_height * output_width;
  const int nthreads = input_numel;

  constexpr int ksize_height = 11;
  constexpr int ksize_width = 11;
  constexpr int stride_height = 4;
  constexpr int stride_width = 4;
  constexpr int padding_height = 1;
  constexpr int padding_width = 1;
  constexpr bool exclusive = true;
  const std::string data_format = "NCHW";
  const bool channel_last = (data_format == "NHWC");
  const int channel_last_int = channel_last ? 1 : 0;

  std::vector<float> input(input_numel);
  std::vector<float> output(output_numel);
  std::vector<float> output_grad(output_numel);
  std::vector<float> input_grad(input_numel, 0.0f);
  std::vector<float> input_grad_ref(input_numel, 0.0f);

  std::srand(123);
  for (int i = 0; i < input_numel; ++i) input[i] = static_cast<float>(std::rand()) / RAND_MAX;
  for (int i = 0; i < output_numel; ++i) {
    output[i] = static_cast<float>(std::rand()) / RAND_MAX;
    output_grad[i] = static_cast<float>(input_width * input_height);
  }

  reference(nthreads, input.data(), output.data(), output_grad.data(), input_channels,
            input_height, input_width, output_height, output_width, ksize_height,
            ksize_width, stride_height, stride_width, padding_height, padding_width,
            exclusive, input_grad_ref.data(), channel_last);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  iris::DMem mem_input(input.data(), input.size() * sizeof(float));
  iris::DMem mem_output(output.data(), output.size() * sizeof(float));
  iris::DMem mem_output_grad(output_grad.data(), output_grad.size() * sizeof(float));
  iris::DMem mem_input_grad(input_grad.data(), input_grad.size() * sizeof(float));

  launch_pool(mem_input, mem_output, mem_output_grad, mem_input_grad, nthreads,
              input_channels, input_height, input_width, output_height, output_width,
              channel_last_int);
  mem_input_grad.fetch(input_grad.data());

  bool ok = true;
  for (int i = 0; i < input_numel; ++i) {
    if (std::fabs(input_grad[i] - input_grad_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_pool(mem_input, mem_output, mem_output_grad, mem_input_grad, nthreads,
                input_channels, input_height, input_width, output_height, output_width,
                channel_last_int);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 2;
}
