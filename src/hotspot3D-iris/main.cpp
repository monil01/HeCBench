#include <chrono>
#include "iris_support.hpp"
#include <vector>
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees  */
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor  */
#define FACTOR_CHIP  0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw = stepDivCap/ Rx;
  cn               = cs = stepDivCap/ Ry;
  ct               = cb = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  memcpy(tCopy,tIn, size * sizeof(float));

  auto start = std::chrono::steady_clock::now();

  iris::Platform platform;
  int device=init_iris(platform,argc,argv);
  iris::DMem first(tIn,sizeof(float)*size), power(pIn,sizeof(float)*size), second(tOut,sizeof(float)*size);
  iris::DMem* d_tIn=&first; iris::DMem* d_tOut=&second;
  auto kstart = std::chrono::steady_clock::now();

  for(int j = 0; j < iterations; j++)
  {
    iris::Task task;
    void* p[] = { d_tIn, &power, d_tOut, &numCols, &numRows, &layers,
                  &ce, &cw, &cn, &cs, &ct, &cb, &cc, &stepDivCap };
    int info[] = { iris_r, iris_r, iris_w, sizeof(int), sizeof(int), sizeof(int),
                   sizeof(float), sizeof(float), sizeof(float), sizeof(float),
                   sizeof(float), sizeof(float), sizeof(float), sizeof(float) };
    size_t global[] = { size_t(numCols/WG_SIZE_X)*WG_SIZE_X, size_t(numRows/WG_SIZE_Y)*WG_SIZE_Y };
    size_t local[] = { WG_SIZE_X, WG_SIZE_Y };
    check_iris(task.kernel("hotspot3d",2,nullptr,global,local,14,p,info));
    check_iris(task.submit(device,nullptr,true));
    iris::DMem* temp = d_tIn;
    d_tIn = d_tOut;
    d_tOut = temp;
  }


  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

  iris::DMem* d_sel = (iterations & 01) ? d_tIn : d_tOut;
  flush_iris(device,*d_sel);
  if(d_sel==&first) memcpy(tOut,tIn,sizeof(float)*size);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  printf("Device offloading time: %.3f (s)\n",time * 1e-9);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  const int errors=platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}
