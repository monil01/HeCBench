#include <cuda.h>

extern "C" __global__ void pool2d_grad_avg(int nthreads,
                                            const float* __restrict__ input_data,
                                            const float* __restrict__ output_data,
                                            const float* __restrict__ output_grad,
                                            int channels,
                                            int input_height,
                                            int input_width,
                                            int output_height,
                                            int output_width,
                                            float* __restrict__ input_grad,
                                            int channel_last)
{
  constexpr int ksize_height = 11;
  constexpr int ksize_width = 11;
  constexpr int stride_height = 4;
  constexpr int stride_width = 4;
  constexpr int padding_height = 1;
  constexpr int padding_width = 1;
  constexpr bool exclusive = true;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < nthreads;
       index += blockDim.x * gridDim.x) {
    int w_offset, h_offset, offset_c, batch_idx;
    int tmp;
    if (!channel_last) {
      w_offset = index % input_width + padding_width;
      tmp = index / input_width;
      h_offset = tmp % input_height + padding_height;
      tmp /= input_height;
      offset_c = tmp % channels;
      batch_idx = tmp / channels;
    } else {
      offset_c = index % channels;
      tmp = index / channels;
      w_offset = tmp % input_width + padding_width;
      tmp /= input_width;
      h_offset = tmp % input_height + padding_height;
      batch_idx = tmp / input_height;
    }

    const int phstart = (h_offset < ksize_height) ? 0 : (h_offset - ksize_height) / stride_height + 1;
    const int pwstart = (w_offset < ksize_width) ? 0 : (w_offset - ksize_width) / stride_width + 1;
    const int phend = min(h_offset / stride_height + 1, output_height);
    const int pwend = min(w_offset / stride_width + 1, output_width);

    float gradient = 0.0f;
    const float input = input_data[index];
    (void)input;

    int output_stride = batch_idx * output_height * output_width * channels;
    if (!channel_last) output_stride += offset_c * output_height * output_width;

    const float* __restrict__ output_data_t = output_data + output_stride;
    const float* __restrict__ output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = min(hstart + ksize_height, input_height);
        int wend = min(wstart + ksize_width, input_width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
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
