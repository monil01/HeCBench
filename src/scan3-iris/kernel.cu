extern "C" __global__ void exclusive_scan_serial(
    const float* input,
    float* output,
    const int length)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    output[i] = sum;
    sum += input[i];
  }
}
