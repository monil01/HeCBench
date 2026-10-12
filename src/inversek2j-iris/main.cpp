#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>

#define MAX_LOOP 25
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)
#define BLOCK_SIZE 128

using namespace std;

static void invkin_cpu(float* xTarget_in, float* yTarget_in, float* angles, int size)
{
  for (int idx = 0; idx < size; idx++) {
    float angle_out[NUM_JOINTS];
    float xData[NUM_JOINTS_P1];
    float yData[NUM_JOINTS_P1];

    float curr_xTargetIn = xTarget_in[idx];
    float curr_yTargetIn = yTarget_in[idx];

    for (int i = 0; i < NUM_JOINTS; i++) angle_out[i] = 0.0f;
    float angle;
    for (int i = 0; i < NUM_JOINTS_P1; i++) {
      xData[i] = i;
      yData[i] = 0.f;
    }

    for (int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++) {
      for (int iter = NUM_JOINTS; iter > 0; iter--) {
        float pe_x = xData[NUM_JOINTS];
        float pe_y = yData[NUM_JOINTS];
        float pc_x = xData[iter - 1];
        float pc_y = yData[iter - 1];
        float diff_pe_pc_x = pe_x - pc_x;
        float diff_pe_pc_y = pe_y - pc_y;
        float diff_tgt_pc_x = curr_xTargetIn - pc_x;
        float diff_tgt_pc_y = curr_yTargetIn - pc_y;
        float len_diff_pe_pc = sqrtf(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
        float len_diff_tgt_pc = sqrtf(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
        float a_x = diff_pe_pc_x / len_diff_pe_pc;
        float a_y = diff_pe_pc_y / len_diff_pe_pc;
        float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
        float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
        float a_dot_b = a_x * b_x + a_y * b_y;
        if (a_dot_b > 1.f) a_dot_b = 1.f;
        else if (a_dot_b < -1.f) a_dot_b = -1.f;
        angle = acosf(a_dot_b) * (180.f / PI);
        float direction = a_x * b_y - a_y * b_x;
        if (direction < 0.f) angle = -angle;
        if (angle > 30.f) angle = 30.f;
        else if (angle < -30.f) angle = -30.f;
        angle_out[iter - 1] = angle;
        for (int i = 0; i < NUM_JOINTS; i++) {
          if (i < NUM_JOINTS - 1) angle_out[i + 1] += angle_out[i];
        }
      }
    }

    angles[idx * NUM_JOINTS + 0] = angle_out[0];
    angles[idx * NUM_JOINTS + 1] = angle_out[1];
    angles[idx * NUM_JOINTS + 2] = angle_out[2];
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    cerr << "Usage: ./invkin <input file coefficients> <iterations>" << endl;
    return EXIT_FAILURE;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  ifstream coordinate_in_file(argv[1]);
  const int iteration = atoi(argv[2]);
  int data_size = 0;
  if (coordinate_in_file.is_open()) {
    coordinate_in_file >> data_size;
    cout << "# Data Size = " << data_size << endl;
  } else {
    cerr << "File open fails. Please re-enter a file path" << endl;
    platform.finalize();
    return EXIT_FAILURE;
  }

  float* xTarget_in_h = new (nothrow) float[data_size];
  float* yTarget_in_h = new (nothrow) float[data_size];
  float* angle_out_h = new (nothrow) float[data_size * NUM_JOINTS];
  float* angle_out_cpu = new (nothrow) float[data_size * NUM_JOINTS];
  if (!xTarget_in_h || !yTarget_in_h || !angle_out_h || !angle_out_cpu) {
    cerr << "Memory allocation fails!!!" << endl;
    platform.finalize();
    return EXIT_FAILURE;
  }

  float xTarget_tmp, yTarget_tmp;
  int coeff_index = 0;
  while (coeff_index < data_size) {
    coordinate_in_file >> xTarget_tmp >> yTarget_tmp;
    for (int i = 0; i < NUM_JOINTS; i++) angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0f;
    xTarget_in_h[coeff_index] = xTarget_tmp;
    yTarget_in_h[coeff_index++] = yTarget_tmp;
  }

  cout << "# Coordinates are read from file..." << endl;
  cout << "# Memory allocation on GPU is done..." << endl;
  cout << "# Data are transfered to GPU..." << endl;

  iris::DMem xTarget_in_d(xTarget_in_h, data_size * sizeof(float));
  iris::DMem yTarget_in_d(yTarget_in_h, data_size * sizeof(float));
  iris::DMem angle_out_d(angle_out_h, data_size * NUM_JOINTS * sizeof(float));

  size_t lws[1] = {BLOCK_SIZE};
  size_t gws[1] = {static_cast<size_t>(((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < iteration; n++) {
    iris::Task task;
    void* params[4] = {&xTarget_in_d, &yTarget_in_d, &angle_out_d, &data_size};
    int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(data_size)};
    task.kernel("invkin_kernel", 1, nullptr, gws, lws, 4, params, pinfo);
    if (n == iteration - 1) task.flush_out(angle_out_d);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Average kernel execution time " << (time * 1e-3f) / iteration << " (us)\n";

  angle_out_d.fetch(angle_out_h);
  invkin_cpu(xTarget_in_h, yTarget_in_h, angle_out_cpu, data_size);

  int error = 0;
  for (int i = 0; i < data_size; i++) {
    for (int j = 0; j < NUM_JOINTS; j++) {
      if (fabsf(angle_out_h[i * NUM_JOINTS + j] - angle_out_cpu[i * NUM_JOINTS + j]) > 1e-3f)
        error++;
    }
  }

  coordinate_in_file.close();
  delete[] xTarget_in_h;
  delete[] yTarget_in_h;
  delete[] angle_out_h;
  delete[] angle_out_cpu;

  cout << (error ? "FAIL\n" : "PASS\n");

  platform.finalize();
  return 0;
}
