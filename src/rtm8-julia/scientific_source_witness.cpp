#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <vector>


#define nx 12
#define ny 12
#define nz 12

inline  int indexTo1D(int x, int y, int z){
  return x + y*nx + z*nx*ny;
}

void rtm8_cpu(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N)
{
#ifdef _OPENMP
  #pragma omp parallel for collapse(3)
#endif
  for (int z = 4; z < nz - 4; z++) {
    for (int y = 4; y < ny - 4; y++) {
      for (int x = 4; x < nx - 4; x++) {
        float div =
          a[0] * current_s[indexTo1D(x,y,z)] +
          a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
              current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
              current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
          a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
              current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
              current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
          a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
              current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
              current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
          a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
              current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
              current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

        next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
          + vsq[indexTo1D(x,y,z)]*div;
        div =
          a[0] * current_r[indexTo1D(x,y,z)] +
          a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
              current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
              current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
          a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
              current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
              current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
          a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
              current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
              current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
          a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
              current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
              current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

        next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
          - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

        image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
      }
    }
  }
}
  

int main(){int N=nx*ny*nz;std::vector<float>v(N,1),cs(N,1),cr(N,1),ns(N,0),nr(N,0),im(N,.5),cr2(N,1),ns2(N,0),nr2(N,0),im2(N,.5);float a[5]={float(-1./560.),float(8./315),-.2f,1.6f,float(-1435./504.)};rtm8_cpu(v.data(),cs.data(),ns.data(),cr.data(),nr.data(),im.data(),a,N);rtm8_cpu(v.data(),cs.data(),cr2.data(),ns2.data(),nr2.data(),im2.data(),a,N);int i=indexTo1D(5,5,5);printf("{\"actual_main_call_image\":%.12g,\"signature_named_call_image\":%.12g,\"actual_current_r_after\":%.12g,\"actual_next_s_after\":%.12g,\"signature_current_r_after\":%.12g,\"signature_next_s_after\":%.12g}\n",im[i],im2[i],cr[i],ns[i],cr2[i],ns2[i]);}