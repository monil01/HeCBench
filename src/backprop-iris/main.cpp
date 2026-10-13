#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iris/iris.hpp>
#include <vector>
#include <stdexcept>

#include "backprop.h"


#include "reference.h"

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
static int validation_status = 0;
static int hip_device = -1;
static void checked(int status, const char* operation) {
  if (status != IRIS_SUCCESS) throw std::runtime_error(operation);
}
int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <number of input nodes>\n", argv[0]);
    return 255;
  }
  if (atoi(argv[1]) <= 0 || atoi(argv[1]) % 16 != 0) {
    fprintf(stderr, "The number of input nodes must be divided by 16 and positive\n");
    return 255;
  }
  iris::Platform platform;
  try {
    checked(platform.init(&argc, &argv, true), "IRIS initialization failed");
    int count = 0;
    checked(iris_device_count(&count), "Device enumeration failed");
    for (int i = 0; i < count; ++i) {
      int backend = -1, type = 0;
      char name[256] = {};
      checked(iris_device_info(i, iris_backend, &backend, nullptr), "Backend query failed");
      checked(iris_device_info(i, iris_type, &type, nullptr), "Device type query failed");
      checked(iris_device_info(i, iris_name, name, nullptr), "Device name query failed");
      printf("IRIS device %d: %s backend=%d type=%d\n", i, name, backend, type);
      if (backend == iris_hip && (type & iris_gpu) && hip_device < 0) hip_device = i;
    }
    if (hip_device < 0) throw std::runtime_error("No HIP GPU available");
    printf("Selected HIP GPU device %d\n", hip_device);
    setup(argc, argv);
    checked(platform.finalize(), "IRIS finalize failed");
    return validation_status;
  } catch (const std::exception& e) {
    fprintf(stderr, "IRIS failure: %s\n", e.what());
    platform.finalize();
    return 1;
  }
}



int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  float *input_weights_one_dim;
  float *input_weights_one_dim_r;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;

  // Warning: the number of blocks must be less than the maximum grid dimension
  unsigned int num_blocks = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_one_dim_r = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_one_dim_r[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing device offload\n");

  auto start = std::chrono::steady_clock::now();

  // Separate scratch preserves CUDA's restore-before-adjustment behavior.
  const size_t input_bytes = sizeof(float) * (in + 1);
  const size_t weight_bytes = sizeof(float) * (in + 1) * (hid + 1);
  std::vector<float> forward_weights(input_weights_one_dim,
                                    input_weights_one_dim + (in + 1) * (hid + 1));
  iris::DMem d_input(net->input_units, input_bytes);
  iris::DMem d_forward_weights(forward_weights.data(), weight_bytes);
  iris::DMem d_partial(partial_sum, sizeof(float) * num_blocks * WIDTH);
  size_t gws[] = {BLOCK_SIZE, static_cast<size_t>(in)};
  size_t lws[] = {BLOCK_SIZE, BLOCK_SIZE};
  iris::Task forward;
  void* forward_params[] = {&d_input, &d_forward_weights, &d_partial, &hid};
  int forward_info[] = {iris_r, iris_rw, iris_w, sizeof(hid)};
  checked(forward.kernel("kernel_layerforward", 2, nullptr, gws, lws,
                         4, forward_params, forward_info), "Forward kernel setup failed");
  checked(forward.flush_out(d_partial), "Partial sum flush setup failed");
  checked(forward.submit(hip_device, nullptr, true), "Forward HIP task failed");

  for (int j = 1; j <= hid; j++) {
    sum = 0.f;
    for (unsigned int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j-1] ;
    }
#ifdef DEBUG
    printf("j=%d sum=%f\n", j,sum);
#endif
    sum += net->input_weights[0][j];
    net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  iris::DMem d_weights(input_weights_one_dim, weight_bytes);
  iris::DMem d_delta(net->hidden_delta, sizeof(float) * (hid + 1));
  // The reference below requires the original momentum weights.
  std::vector<float> previous(input_weights_prev_one_dim,
                             input_weights_prev_one_dim + (in + 1) * (hid + 1));
  iris::DMem d_previous(previous.data(), weight_bytes);
  iris::Task adjust;
  void* adjust_params[] = {&d_input, &d_weights, &d_delta, &d_previous, &hid};
  int adjust_info[] = {iris_r, iris_rw, iris_r, iris_rw, sizeof(hid)};
  checked(adjust.kernel("kernel_adjust_weights", 2, nullptr, gws, lws,
                        5, adjust_params, adjust_info), "Adjustment kernel setup failed");
  checked(adjust.flush_out(d_weights), "Weight flush setup failed");
  checked(adjust.submit(hip_device, nullptr, true), "Adjustment HIP task failed");

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time = %lf(s)\n", time * 1e-9);

  reference (in, hid, out, net,
             input_weights_one_dim_r,
             input_weights_prev_one_dim,
             partial_sum);

  bool ok = true;
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (!std::isfinite(input_weights_one_dim[i]) ||
        fabsf(input_weights_one_dim[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  validation_status = ok ? 0 : 1;

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++)
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++)
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_one_dim_r);

  return 0;
}
