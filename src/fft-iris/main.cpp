#include "iris_dmem.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>
struct alignas(8) float2 { float x,y; };
struct alignas(16) double2 { double x,y; };
#include "fft_math.hpp"
#include "reference.h"
int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n", argv[0]);
    printf("Problem size [0-3]: 0=1M, 1=8M, 2=96M, 3=256M\n");
    return 1;
  }

  IRISSession session(&argc,&argv);
  srand(2);
  int i;
  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Convert to MB
  int probSizes[4] = { 1, 8, 96, 256 };
  unsigned long bytes = probSizes[select];
  bytes *= 1024 * 1024;

  // now determine how much available memory will be used
  const int half_n_ffts = bytes / (512*sizeof(T2)*2);
  const int n_ffts = half_n_ffts * 2;
  const int half_n_cmplx = half_n_ffts * 512;
  const unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
  const int n_cmplx = half_n_cmplx*2;

  fprintf(stdout, "used_bytes=%lu, n_cmplx=%d\n", used_bytes, n_cmplx);

  // allocate host memory, in-place FFT/iFFT operations
  T2 *source = (T2*) malloc (used_bytes);
  T2 *reference = (T2*) malloc (used_bytes);

  // init host memory...
  for (i = 0; i < half_n_cmplx; i++) {
    source[i].x = sinf(i / powf(10000, i % 768 / 384));
    source[i].y = cosf(i / powf(10000, i % 768 / 384));
    source[i+half_n_cmplx].x = source[i].x;
    source[i+half_n_cmplx].y = source[i].y;
  }

  memcpy(reference, source, used_bytes);

  T2 *d_source;
  iris_alloc((void**)&d_source, used_bytes);
  iris_copy(d_source, source, used_bytes, IRISHostToDevice);

  iris_launch("fft1D_512", n_ffts, 64, d_source);

  // verify FFT
  fft1D_512_reference<64>(reference, n_ffts);
  iris_copy(source, d_source, used_bytes, IRISDeviceToHost);
  bool error = false;
  for (int i = 0; i < n_cmplx; i++) {
    if (fabs((T)source[i].x - (T)reference[i].x) > EPISON) {
      //std::cout << i << " " << (T)source[i].x << " " << (T)reference[i].x << std::endl;
      error = true;
      break;
    }
    if (fabs((T)source[i].y - (T)reference[i].y) > EPISON) {
      //std::cout << i << " " << (T)source[i].y << " " << (T)reference[i].y << std::endl;
      error = true;
      break;
    }
  }
  std::cout << "FFT " << (error ? "FAIL" : "PASS")  << std::endl;
 
  // execute iFFT
  iris_launch("ifft1D_512", n_ffts, 64, d_source);

  // verify iFFT
  iris_copy(source, d_source, used_bytes, IRISDeviceToHost);
  error = false;
  for (int i = 0; i < n_cmplx; i++) {
    int j = i % half_n_cmplx;
    if (fabs((T)source[i].x - (T)sinf(j / powf(10000, j%768/384))) > EPISON) {
      error = true;
      break;
    }
    if (fabs((T)source[i].y - (T)cosf(j / powf(10000, j%768/384))) > EPISON) {
      error = true;
      break;
    }
  }
  std::cout << "iFFT " << (error ? "FAIL" : "PASS")  << std::endl;

  auto start = std::chrono::steady_clock::now();

  for (int k=0; k<passes; k++) {
    iris_launch("fft1D_512", n_ffts, 64, d_source);
    iris_launch("ifft1D_512", n_ffts, 64, d_source);
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";

  iris_free(d_source);

  free(reference);
  free(source);

  return 0;
}
