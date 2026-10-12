#include <hip/hip_runtime.h>

#include <math.h>

extern "C" __global__ void elementwise(int hiddenSize, int miniBatch,
                                        const float* __restrict__ tmp_h,
                                        int tmp_h_offset,
                                        const float* __restrict__ tmp_i,
                                        int tmp_i_offset,
                                        const float* __restrict__ bias,
                                        int bias_offset,
                                        float* __restrict__ linearGates,
                                        int linear_offset,
                                        float* __restrict__ h_data,
                                        int h_offset,
                                        float* __restrict__ i_data,
                                        int i_offset,
                                        float* __restrict__ c_data,
                                        int c_in_offset,
                                        int c_out_offset) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int numElements = miniBatch * hiddenSize;
  if (index >= numElements) return;

  const float* tmp_h_base = tmp_h + tmp_h_offset;
  const float* tmp_i_base = tmp_i + tmp_i_offset;
  const float* bias_base = bias + bias_offset;
  float* linear_base = linearGates + linear_offset;
  float* h_out = h_data + h_offset;
  float* i_out = i_data + i_offset;
  const float* c_in = c_data + c_in_offset;
  float* c_out = c_data + c_out_offset;

  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;
  float g[4];
  for (int i = 0; i < 4; i++) {
    g[i] = tmp_i_base[i * hiddenSize + gateIndex] + tmp_h_base[i * hiddenSize + gateIndex];
    g[i] += bias_base[i * hiddenSize + index % hiddenSize] +
            bias_base[(i + 4) * hiddenSize + index % hiddenSize];
    linear_base[gateIndex + i * hiddenSize] = g[i];
  }

  float in_gate = 1.0f / (1.0f + expf(-g[0]));
  float forget_gate = 1.0f / (1.0f + expf(-g[1]));
  float in_gate2 = tanhf(g[2]);
  float out_gate = 1.0f / (1.0f + expf(-g[3]));
  float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
  c_out[index] = val;
  val = out_gate * tanhf(val);
  h_out[index] = val;
  i_out[index] = val;
}
