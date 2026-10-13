#include <cstdio>
#include <chrono>
#include <cmath>
#include <string>
#include <iris/iris.hpp>
#include <algorithm>
#include <vector>
#include "layer.h"
int gpu_device=-1;
static FILE* validation_output=nullptr;

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"


static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn(int iter);
static unsigned int classify(double data[28][28]);
static void test();
void forward_pass(double data[28][28]);
void back_pass();

static inline int loaddata(std::string data_dir)
{
  std::string train_image_path = data_dir + "/train-images.idx3-ubyte";
  std::string train_label_path = data_dir + "/train-labels.idx1-ubyte";
  std::string test_image_path  = data_dir + "/t10k-images.idx3-ubyte";
  std::string test_label_path  = data_dir + "/t10k-labels.idx1-ubyte";

  int s1 = mnist_load(train_image_path.c_str(), train_label_path.c_str(),
                      &train_set, &train_cnt);
  int s2 = mnist_load(test_image_path.c_str(), test_label_path.c_str(),
                      &test_set, &test_cnt);
  return s1 | s2;
}

// replace cublas function in the case n = 10
void snrm2(const int n, Buffer &x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));
  x.fetch();
  std::copy(x.host.begin(),x.host.begin()+n,r);
  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <data path> <iterations>\n", argv[0]);
    return 1;
  }

  const char* path = argv[1];
  const int iter = atoi(argv[2]);
  srand(123);
  if (loaddata(path) != 0) return 1;
  if(const char* value=getenv("HECBENCH_MNIST_TRAIN_LIMIT")) train_cnt=std::min(train_cnt,static_cast<unsigned>(atoi(value)));
  if(const char* value=getenv("HECBENCH_MNIST_TEST_LIMIT")) test_cnt=std::min(test_cnt,static_cast<unsigned>(atoi(value)));
  if(!train_cnt || !test_cnt) return 2;
  if(const char* path=getenv("HECBENCH_MNIST_OUTPUT")) {
    validation_output=fopen(path,"wb");if(!validation_output)return 2;
  }
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  gpu_device=device;
  srand(123);
  l_input.bind();l_c1.bind();l_s1.bind();l_f.bind();

  auto t1 = std::chrono::high_resolution_clock::now();
  learn(iter);
  test();
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Total time (learn + test) %lf secs \n", total_time / 1.0e6);
  if(validation_output) {
    for(Layer* layer : {&l_c1,&l_s1,&l_f}) {
      layer->weight.fetch();layer->bias.fetch();
      fwrite(layer->weight.host.data(),sizeof(float),layer->weight.host.size(),validation_output);
      fwrite(layer->bias.host.data(),sizeof(float),layer->bias.host.size(),validation_output);
    }
    fclose(validation_output);
  }
  platform.finalize();
  return 0;
}

// Forward propagation of a single row in dataset
void forward_pass(double data[28][28])
{
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  float input[28][28];
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }

  l_input.setOutput((float *)input);

  launch("fp_preact_c1",64,64,l_input.output, l_c1.preact, l_c1.weight);
  launch("fp_bias_c1",64,64,l_c1.preact, l_c1.bias);
  launch("apply_step_function",64,64,l_c1.preact, l_c1.output, l_c1.O);

  launch("fp_preact_s1",64,64,l_c1.output, l_s1.preact, l_s1.weight);
  launch("fp_bias_s1",64,64,l_s1.preact, l_s1.bias);
  launch("apply_step_function",64,64,l_s1.preact, l_s1.output, l_s1.O);

  launch("fp_preact_f",64,64,l_s1.output, l_f.preact, l_f.weight);
  launch("fp_bias_f",64,64,l_f.preact, l_f.bias);
  launch("apply_step_function",64,64,l_f.preact, l_f.output, l_f.O);
}

// Back propagation to update weights
void back_pass()
{
  launch("bp_weight_f",64,64,l_f.d_weight, l_f.d_preact, l_s1.output);
  launch("bp_bias_f",64,64,l_f.bias, l_f.d_preact);

  launch("bp_output_s1",64,64,l_s1.d_output, l_f.weight, l_f.d_preact);
  launch("bp_preact_s1",64,64,l_s1.d_preact, l_s1.d_output, l_s1.preact);
  launch("bp_weight_s1",64,64,l_s1.d_weight, l_s1.d_preact, l_c1.output);
  launch("bp_bias_s1",64,64,l_s1.bias, l_s1.d_preact);

  launch("bp_output_c1",64,64,l_c1.d_output, l_s1.weight, l_s1.d_preact);
  launch("bp_preact_c1",64,64,l_c1.d_preact, l_c1.d_output, l_c1.preact);
  launch("bp_weight_c1",64,64,l_c1.d_weight, l_c1.d_preact, l_input.output);
  launch("bp_bias_c1",64,64,l_c1.bias, l_c1.d_preact);


  launch("apply_grad",64,64,l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  launch("apply_grad",64,64,l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  launch("apply_grad",64,64,l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
}

static void learn(int iter)
{
  float err;

  fprintf(stdout ,"Learning\n");

  while (iter < 0 || iter-- > 0) {
    err = 0.0f;

    for (unsigned int i = 0; i < train_cnt; ++i) {
      float tmp_err;

      forward_pass(train_set[i].data);

      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      // Euclid distance of train_set[i]
      launch("makeError",10,1,l_f.d_preact, l_f.output, train_set[i].label, 10);
      snrm2(10, l_f.d_preact, tmp_err);
      err += tmp_err;

      back_pass();
    }

    err /= train_cnt;
    fprintf(stdout, "error: %e\n", err);

    if (err < threshold) {
      fprintf(stdout, "Training complete, error less than threshold\n\n");
      break;
    }
  }
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
  forward_pass(data);

  float res[10];
  unsigned int max = 0;

  l_f.output.fetch();
  std::copy(l_f.output.host.begin(),l_f.output.host.end(),res);
  if(validation_output) fwrite(res,sizeof(float),10,validation_output);

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test()
{
  fprintf(stdout ,"Testing\n");

  int error = 0;

  for (unsigned int i = 0; i < test_cnt; ++i) {
    if (classify(test_set[i].data) != test_set[i].label) {
      ++error;
    }
  }

  fprintf(stdout, "Error Rate: %.2lf%%\n",
      double(error) / double(test_cnt) * 100.0);
}
