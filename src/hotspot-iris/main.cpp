#include <chrono>
#include <cmath>
#include <cstring>
#include <iris/iris.hpp>
#include <vector>
#include <cstdio>
#include <cstdlib>
static int gpu;
static void check(int rc) { if (rc != IRIS_SUCCESS) { fprintf(stderr, "IRIS error %d\n", rc); exit(2); } }
#include "hotspot.h"

#include "reference.h"


void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 ) {
    printf( "Unable to open file %s\n", file );
    return;
  }

  for (i=0; i < grid_rows; i++)
    for (j=0; j < grid_cols; j++)
    {
      sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }

  fclose(fp);
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 ) {
    printf( "The file %s was not opened successfully", file );
    exit(-1);
  }

  for (i=0; i <= grid_rows-1; i++)
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) {
        printf("Error reading file\n");
        exit(-1);
      }
      if (feof(fp)) {
        printf("not enough lines in file");
        exit(-1);
      }
      if ((sscanf(str, "%f", &val) != 1)) {
        printf("invalid file format");
        exit(-1);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);
}

/* compute N time steps */
int compute_tran_temp(
    iris::DMem &MatrixPower,
    iris::DMem *MatrixTemp[2],
    int col, int row,
    int total_iterations, int num_iterations,
    int blockCols, int blockRows, int borderCols, int borderRows)
{
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  int t;
#ifdef DEBUG
  printf("%f %f %f %f %f %f %f\n", grid_height,grid_width,Cap,Rx,Ry,Rz,step);
#endif

  int src = 0, dst = 1;

  // Determine GPU work group grid
  size_t lws[] = {BLOCK_SIZE, BLOCK_SIZE};
  size_t gws[] = {size_t(blockCols)*BLOCK_SIZE, size_t(blockRows)*BLOCK_SIZE};

  for (t = 0; t < total_iterations; t += num_iterations) {

    // Specify kernel arguments
    int iter = MIN(num_iterations, total_iterations - t);

    iris::Task task;
    void* params[] = {&iter, &MatrixPower, MatrixTemp[src], MatrixTemp[dst],
                      &col, &row, &borderCols, &borderRows, &Cap, &Rx, &Ry, &Rz, &step};
    int modes[] = {sizeof(int), iris_r, iris_r, iris_w, sizeof(int), sizeof(int),
                   sizeof(int), sizeof(int), sizeof(float), sizeof(float),
                   sizeof(float), sizeof(float), sizeof(float)};
    check(task.kernel("calc_temp", 2, nullptr, gws, lws, 13, params, modes));
    check(task.submit(gpu, nullptr, true));

    // Swap input and output GPU matrices
    src = 1 - src;
    dst = 1 - dst;
  }

  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char** argv) {

  if (argc < 7) usage(argc, argv);

  int size;
  int grid_rows,grid_cols = 0;
  float *FilesavingTemp,*FilesavingPower;
  char *tfile, *pfile, *ofile;

  int total_iterations = 60;  // this can be overwritten by the commandline argument
  int pyramid_height = 1;     // step size

  if((grid_rows = atoi(argv[1]))<=0||
     (grid_cols = atoi(argv[1]))<=0||
     (pyramid_height = atoi(argv[2]))<=0||
     (total_iterations = atoi(argv[3]))<=0)
    usage(argc, argv);

  printf("Work-group size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  tfile=argv[4];
  pfile=argv[5];
  ofile=argv[6];

  size=sizeof(float)*grid_rows*grid_cols;

  // --------------- pyramid parameters ---------------
  int borderCols = (pyramid_height)*EXPAND_RATE/2;
  int borderRows = (pyramid_height)*EXPAND_RATE/2;
  int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
  int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

  FilesavingTemp = (float *) malloc(size);
  FilesavingPower = (float *) malloc(size);
  float *result = (float *) malloc(size);

  if( !FilesavingPower || !FilesavingTemp || !result) {
    printf("unable to allocate memory");
    exit(-1);
  }

  float *MatrixTemp_ref[2];
  MatrixTemp_ref[0] = (float*) malloc (size);
  MatrixTemp_ref[1] = (float*) malloc (size);
  if( !MatrixTemp_ref[0] || !MatrixTemp_ref[1]) {
    printf("unable to allocate memory");
    exit(-1);
  }

  // Read input data from disk
  readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
  readinput(FilesavingPower, grid_rows, grid_cols, pfile);

  // reference
  auto start = std::chrono::steady_clock::now();

  memcpy(MatrixTemp_ref[0], FilesavingTemp, size);
  int ret = reference(FilesavingPower, MatrixTemp_ref, grid_cols, grid_rows,
                      total_iterations, pyramid_height);
  float *result_ref = MatrixTemp_ref[ret];

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total reference execution time %f (s)\n", time * 1e-9f);

  // device offloading
  start = std::chrono::steady_clock::now();

  iris::Platform platform;
  check(platform.init(&argc, &argv, true));
  int devices = 0;
  check(iris_device_count(&devices));
  gpu = -1;
  for (int d = 0; d < devices; ++d) {
    int type = 0;
    check(iris_device_info(d, iris_type, &type, nullptr));
    if ((type & iris_amd) && gpu < 0) gpu = d;
  }
  if (gpu < 0) { fprintf(stderr, "No AMD GPU\n"); return 2; }
  platform.overview();
  iris::DMem MatrixPower(FilesavingPower, size);
  std::vector<float> temp0(grid_rows*grid_cols), temp1(grid_rows*grid_cols);
  memcpy(temp0.data(), FilesavingTemp, size);
  iris::DMem mem0(temp0.data(), size), mem1(temp1.data(), size);
  iris::DMem *MatrixTemp[] = {&mem0, &mem1};
  auto kstart = std::chrono::steady_clock::now();
  ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows,
                          total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);


  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Total kernel execution time %f (s)\n", ktime * 1e-9f);

  iris::Task flush;
  check(flush.flush_out(*MatrixTemp[ret]));
  check(flush.submit(gpu, nullptr, true));
  memcpy(result, ret == 0 ? temp0.data() : temp1.data(), size);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time: %.3f (s)\n", time * 1e-9f);

  bool ok = true;
  for (int i = 0; i < grid_cols * grid_rows; i++) {
    if (fabsf(result_ref[i] - result[i]) > 1e-3f) {
       ok = false;
       break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Write final output to output file
  writeoutput(result, grid_rows, grid_cols, ofile);

  free(MatrixTemp_ref[0]);
  free(MatrixTemp_ref[1]);
  free(FilesavingTemp);
  free(FilesavingPower);
  free(result);

  return ok ? 0 : 1;
}
