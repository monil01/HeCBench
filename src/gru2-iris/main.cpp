#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <stdexcept>
#include "reference.h"
static void check(int rc) { if (rc != IRIS_SUCCESS) throw std::runtime_error("IRIS API failure"); }
static int hip_device() {
  int n=0; check(iris_device_count(&n));
  for(int i=0;i<n;++i) { int backend=0,type=0; char name[256]={};
    check(iris_device_info(i,iris_backend,&backend,nullptr));
    check(iris_device_info(i,iris_type,&type,nullptr));
    check(iris_device_info(i,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",i,name,backend,type);
    if(backend==iris_hip && (type&iris_gpu)) return i;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}
static iris::DMem* gru(int device,
    iris::DMem& input, iris::DMem& wih, iris::DMem& whh, iris::DMem& bih, iris::DMem& bhh,
    iris::DMem& h0, iris::DMem& output, iris::DMem& hidden, iris::DMem& hidden_a,
    iris::DMem& layer_a, iris::DMem& layer_b,
    int seq, int batch, int insize, int hidden_size, int layers, int max_input) {
  iris::DMem* read=&h0;
  for(int t=0;t<seq;++t) {
    iris::DMem* write=t%2==0 ? &hidden : &hidden_a;
    size_t output_offset=size_t(t)*batch*hidden_size;
    for(int layer=0;layer<layers;++layer) {
      iris::DMem* x=layer==0 ? &input : (layer%2==1 ? &layer_a : &layer_b);
      iris::DMem* out=layer%2==0 ? &layer_a : &layer_b;
      size_t input_offset=layer==0 ? size_t(t)*batch*insize : 0;
      size_t hidden_offset=size_t(layer)*batch*hidden_size;
      int current_size=layer==0 ? insize : hidden_size;
      int last_layer=layer==layers-1;
      size_t gws[]={size_t(batch)*256,size_t((hidden_size+3)/4)}, lws[]={256,1};
      void* args[]={x,read,&wih,&whh,&bih,&bhh,out,write,&output,
                   &layer,&batch,&current_size,&hidden_size,&max_input,
                   &input_offset,&hidden_offset,&output_offset,&last_layer};
      int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,iris_rw,iris_rw,
                  sizeof(layer),sizeof(batch),sizeof(current_size),sizeof(hidden_size),sizeof(max_input),
                  sizeof(input_offset),sizeof(hidden_offset),sizeof(output_offset),sizeof(last_layer)};
      iris::Task task;
      check(task.kernel("fused_gru_layer_kernel",2,nullptr,gws,lws,18,args,info));
      check(task.submit(device,nullptr,true));
    }
    read=write;
  }
  return read;
}
int main(int argc, char** argv) {
  if (argc != 7) {
    std::printf(
        "Usage: %s <seq_len> <batch_size> <input_size> <hidden_size> "
        "<num_layers> <repeat>\n",
        argv[0]);
    return 1;
  }

  const int seq_len = std::atoi(argv[1]);
  const int batch_size = std::atoi(argv[2]);
  const int input_size = std::atoi(argv[3]);
  const int hidden_size = std::atoi(argv[4]);
  const int num_layers = std::atoi(argv[5]);
  const int repeat = std::atoi(argv[6]);
  if (seq_len <= 0 || batch_size <= 0 || input_size <= 0 ||
      hidden_size <= 0 || num_layers <= 0 || repeat <= 0) {
    std::fprintf(stderr, "All arguments must be positive integers.\n");
    return 1;
  }

  const int max_input_size = std::max(input_size, hidden_size);
  const size_t input_elements = (size_t)seq_len * batch_size * input_size;
  const size_t output_elements = (size_t)seq_len * batch_size * hidden_size;
  const size_t hidden_state_elements =
      (size_t)num_layers * batch_size * hidden_size;
  const size_t w_ih_elements =
      (size_t)num_layers * 3 * hidden_size * max_input_size;
  const size_t w_hh_elements =
      (size_t)num_layers * 3 * hidden_size * hidden_size;
  const size_t bias_elements = (size_t)num_layers * 3 * hidden_size;
  const size_t layer_output_elements = (size_t)batch_size * hidden_size;

  std::vector<float> h_input(input_elements);
  std::vector<float> h_w_ih(w_ih_elements, 0.0f);
  std::vector<float> h_w_hh(w_hh_elements);
  std::vector<float> h_b_ih(bias_elements);
  std::vector<float> h_b_hh(bias_elements);
  std::vector<float> h_h0(hidden_state_elements);
  std::vector<float> h_output(output_elements);
  std::vector<float> h_hidden(hidden_state_elements);
  std::vector<float> h_output_ref(output_elements);
  std::vector<float> h_hidden_ref(hidden_state_elements);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  for (float& v : h_input) v = dist(gen);
  for (float& v : h_w_ih) v = dist(gen);
  for (float& v : h_w_hh) v = dist(gen);
  for (float& v : h_b_ih) v = dist(gen);
  for (float& v : h_b_hh) v = dist(gen);
  for (float& v : h_h0) v = dist(gen);

  iris::Platform platform;
  check(platform.init(&argc,&argv,true));
  int device = hip_device();
  std::vector<float> h_hidden_a(hidden_state_elements), layer_a(layer_output_elements), layer_b(layer_output_elements);
  iris::DMem d_input(h_input.data(),input_elements*sizeof(float));
  iris::DMem d_w_ih(h_w_ih.data(),w_ih_elements*sizeof(float)), d_w_hh(h_w_hh.data(),w_hh_elements*sizeof(float));
  iris::DMem d_b_ih(h_b_ih.data(),bias_elements*sizeof(float)), d_b_hh(h_b_hh.data(),bias_elements*sizeof(float));
  iris::DMem d_h0(h_h0.data(),hidden_state_elements*sizeof(float));
  iris::DMem d_output(h_output.data(),output_elements*sizeof(float));
  iris::DMem d_hidden(h_hidden.data(),hidden_state_elements*sizeof(float)), d_hidden_a(h_hidden_a.data(),hidden_state_elements*sizeof(float));
  iris::DMem d_layer_output_a(layer_a.data(),layer_output_elements*sizeof(float)), d_layer_output_b(layer_b.data(),layer_output_elements*sizeof(float));
  iris::DMem* d_final_hidden = gru(device,
      d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output, d_hidden,
      d_hidden_a, d_layer_output_a, d_layer_output_b, seq_len, batch_size,
      input_size, hidden_size, num_layers, max_input_size);

  iris::Task flush;
  check(flush.flush_out(d_output)); check(flush.flush_out(*d_final_hidden));
  check(flush.submit(device,nullptr,true));
  const auto& final_hidden = d_final_hidden == &d_hidden ? h_hidden : h_hidden_a;
  gru_reference(h_input.data(), h_w_ih.data(), h_w_hh.data(), h_b_ih.data(),
                h_b_hh.data(), h_h0.data(), h_output_ref.data(),
                h_hidden_ref.data(), seq_len, batch_size, input_size,
                hidden_size, num_layers, max_input_size);

  float max_output_error = 0.0f;
  for (size_t i = 0; i < output_elements; ++i) {
    max_output_error = std::max(max_output_error,
                                std::fabs(h_output[i] - h_output_ref[i]));
  }
  float max_hidden_error = 0.0f;
  for (size_t i = 0; i < hidden_state_elements; ++i) {
    max_hidden_error = std::max(max_hidden_error,
                                std::fabs(final_hidden[i] - h_hidden_ref[i]));
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    d_final_hidden = gru(device,
        d_input, d_w_ih, d_w_hh, d_b_ih, d_b_hh, d_h0, d_output, d_hidden,
        d_hidden_a, d_layer_output_a, d_layer_output_b, seq_len, batch_size,
        input_size, hidden_size, num_layers, max_input_size);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::printf("Average execution time of multi_layer_gru: %f (us)\n",
              (elapsed_ns * 1e-3f) / repeat);
  std::printf("max_output_error: %.8e\n", max_output_error);
  std::printf("max_hidden_error: %.8e\n", max_hidden_error);
  const bool ok = (max_output_error < 1e-4f && max_hidden_error < 1e-4f);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  check(platform.finalize());
  return ok ? 0 : 1;
}
