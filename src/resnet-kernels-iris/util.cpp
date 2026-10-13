#include "util.h"

float* transpose(float* weight, int h, int w) {
  float* new_weight = (float*)malloc(w * h * 4);
  int i, j;
  for (i = 0; i < w; ++i) {
    for (j = 0; j < h; ++j) {
      new_weight[j * w + i] = weight[i * h + j];
    }
  }

  free(weight);
  return new_weight;
}

float* get_parameter(const char* filename, int size) {
  float* parameter = (float*)malloc(size * 4);
  if (!parameter) {
    printf("Bad Malloc\n");
    exit(0);
  }
  FILE* ptr = fopen(filename, "rb");

  if (!ptr) {
    printf("Bad file path %s: %p, %s\n", filename, ptr, strerror(errno));
    exit(0);
  }
  fread(parameter, size * 4, 1, ptr);

  fclose(ptr);
  return parameter;
}

void output_checker(float* A, float* B, int len, int channel, int shift) {
  int error_cnt = 0, i, j, k;
  float max_error = 0;
  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) {
      for (k = 0; k < channel; k++) {
        float diff = fabs(
            A[((i + shift) * (len + 2 * shift) + j + shift) * channel + k] -
            B[(i * len + j) * channel + k]);
        if (diff > 1e-5)
          error_cnt++;
        if (diff > max_error)
          max_error = diff;
      }
    }
  }
  printf("[max_error: %f][error_cnt: %d]\n", max_error, error_cnt);
}

void dump_result(const float* result, int n) {
  const char* path = getenv("RESNET_DUMP");
  if (!path) return;
  FILE* f = fopen(path, "ab");
  if (!f || fwrite(result, sizeof(float), n, f) != size_t(n)) exit(1);
  fclose(f);
}
void launch(const char* name, std::initializer_list<size_t> grid,
            std::initializer_list<size_t> block, std::initializer_list<iris::DMem*> mems) {
  size_t gws[3] = {1,1,1}, lws[3] = {1,1,1};
  int dim = std::max(grid.size(), block.size());
  int i = 0; for (auto n : grid) gws[i++] = n;
  i = 0; for (auto n : block) lws[i++] = n;
  for (i = 0; i < 3; ++i) gws[i] *= lws[i];
  std::vector<void*> params; std::vector<int> info;
  for (auto m : mems) { params.push_back(m); info.push_back(iris_rw); }
  iris::Task task;
  if (task.kernel(name, dim, nullptr, gws, lws, params.size(), params.data(), info.data()) != IRIS_SUCCESS ||
      task.submit(iris_gpu, nullptr, 1) != IRIS_SUCCESS) exit(1);
}

