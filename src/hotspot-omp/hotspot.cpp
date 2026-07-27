// hotspot-omp: OpenMP target-offload port of hotspot-cuda.
// Uses a simple per-iteration 5-point stencil sweep instead of the pyramid
// tiling in the CUDA kernel; correctness is verified against the CPU
// reference in reference.h at the CUDA benchmark's declared 1e-3 tolerance.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "hotspot.h"
#include "reference.h"

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
  FILE *fp = fopen(file, "w");
  if (!fp) { printf("Unable to open file %s\n", file); return; }
  int index = 0;
  for (int i = 0; i < grid_rows; i++)
    for (int j = 0; j < grid_cols; j++) {
      fprintf(fp, "%d\t%g\n", index, vect[i*grid_cols+j]);
      index++;
    }
  fclose(fp);
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {
  FILE *fp = fopen(file, "r");
  if (!fp) { printf("The file %s was not opened successfully", file); exit(-1); }
  char str[STR_SIZE];
  float val;
  for (int i = 0; i < grid_rows; i++)
    for (int j = 0; j < grid_cols; j++) {
      if (!fgets(str, STR_SIZE, fp)) { printf("Error reading\n"); exit(-1); }
      if (sscanf(str, "%f", &val) != 1) { printf("bad format"); exit(-1); }
      vect[i*grid_cols+j] = val;
    }
  fclose(fp);
}

void device_stencil(float *curr, float *next, const float *power,
                    int rows, int cols,
                    float step_div_Cap, float Rx_1, float Ry_1, float Rz_1) {
  const float amb_temp = 80.0f;
  #pragma omp target teams distribute parallel for collapse(2)
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      int N = y - 1; if (N < 0) N = 0;
      int S = y + 1; if (S > rows - 1) S = rows - 1;
      int W = x - 1; if (W < 0) W = 0;
      int E = x + 1; if (E > cols - 1) E = cols - 1;
      int idx = y * cols + x;
      float t = curr[idx];
      next[idx] = t + step_div_Cap * ( power[idx]
        + (curr[S*cols+x] + curr[N*cols+x] - 2.f*t) * Ry_1
        + (curr[y*cols+E] + curr[y*cols+W] - 2.f*t) * Rx_1
        + (amb_temp - t) * Rz_1 );
    }
  }
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <rows/cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 7) usage(argc, argv);
  int rows = atoi(argv[1]);
  int cols = atoi(argv[1]);
  int pyramid_height = atoi(argv[2]);
  int total_iterations = atoi(argv[3]);
  if (rows <= 0 || cols <= 0 || pyramid_height <= 0 || total_iterations <= 0) usage(argc, argv);
  char *tfile = argv[4], *pfile = argv[5], *ofile = argv[6];

  printf("Work-group size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int size = rows * cols;
  float *FilesavingTemp = (float*)malloc(size * sizeof(float));
  float *FilesavingPower = (float*)malloc(size * sizeof(float));
  float *result = (float*)malloc(size * sizeof(float));
  float *MatrixTemp_ref[2];
  MatrixTemp_ref[0] = (float*)malloc(size * sizeof(float));
  MatrixTemp_ref[1] = (float*)malloc(size * sizeof(float));

  readinput(FilesavingTemp, rows, cols, tfile);
  readinput(FilesavingPower, rows, cols, pfile);

  // reference
  auto start = std::chrono::steady_clock::now();
  memcpy(MatrixTemp_ref[0], FilesavingTemp, size * sizeof(float));
  int ret = reference(FilesavingPower, MatrixTemp_ref, cols, rows,
                      total_iterations, pyramid_height);
  float *result_ref = MatrixTemp_ref[ret];
  auto end = std::chrono::steady_clock::now();
  double t = std::chrono::duration<double>(end - start).count();
  printf("Total reference execution time %f (s)\n", t);

  // device
  start = std::chrono::steady_clock::now();
  float grid_height = chip_height / rows;
  float grid_width = chip_width / cols;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  float step_div_Cap = step / Cap;
  float Rx_1 = 1.f / Rx, Ry_1 = 1.f / Ry, Rz_1 = 1.f / Rz;

  float *curr = (float*)malloc(size * sizeof(float));
  float *next = (float*)malloc(size * sizeof(float));
  memcpy(curr, FilesavingTemp, size * sizeof(float));

  #pragma omp target enter data map(to: curr[0:size], next[0:size], FilesavingPower[0:size])
  auto kstart = std::chrono::steady_clock::now();
  for (int iter = 0; iter < total_iterations; iter++) {
    device_stencil(curr, next, FilesavingPower, rows, cols, step_div_Cap, Rx_1, Ry_1, Rz_1);
    std::swap(curr, next);
  }
  auto kend = std::chrono::steady_clock::now();
  double kt = std::chrono::duration<double>(kend - kstart).count();
  printf("Total kernel execution time %f (s)\n", kt);
  #pragma omp target exit data map(from: curr[0:size])

  memcpy(result, curr, size * sizeof(float));
  end = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(end - start).count();
  printf("Device offloading time: %.3f (s)\n", dt);

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabsf(result_ref[i] - result[i]) > 1e-3f) { ok = false; break; }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  writeoutput(result, rows, cols, ofile);

  free(curr); free(next);
  free(MatrixTemp_ref[0]); free(MatrixTemp_ref[1]);
  free(FilesavingTemp); free(FilesavingPower); free(result);
  return 0;
}
