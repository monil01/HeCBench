// Independent serial execution of the source column recurrence; validation only.
#include <cmath>
#include <cstdio>
#include <vector>
// Match HIP's floating-point abs overloads in the shared column equations.
// Without this, global abs(int) truncates rates and changes saturation guards.
using std::abs;
static float rsqrtf(float x) {return 1.f/std::sqrt(x);}
#define __global__
struct Index {int x=0,y=0,z=0;};
Index blockIdx,threadIdx,blockDim{8,8,1};
#define MKX 4
#include "kernel.h"
int main(int argc,char**argv) {
 if(argc!=2)return 1;
 constexpr int nx=60,ny=46,nz=3;
 std::vector<std::vector<float>> a(15);
 for(int i=0;i<15;++i)a[i].assign(i<10?nx*ny*nz:nx*ny,0.001f);
 for(blockIdx.y=0;blockIdx.y<(ny+7)/8;++blockIdx.y)
 for(blockIdx.x=0;blockIdx.x<(nx+7)/8;++blockIdx.x)
 for(threadIdx.y=0;threadIdx.y<8;++threadIdx.y)
 for(threadIdx.x=0;threadIdx.x<8;++threadIdx.x)
 wsm(a[0].data(),a[1].data(),a[2].data(),a[3].data(),a[4].data(),
     a[5].data(),a[6].data(),a[7].data(),a[8].data(),a[9].data(),
     a[10].data(),a[11].data(),a[12].data(),a[13].data(),a[14].data(),
     10.f,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz);
 FILE*f=fopen(argv[1],"wb");if(!f)return 1;
 for(int i:{0,2,3,4,5,6,10,11,12,13,14})
  if(fwrite(a[i].data(),sizeof(float),a[i].size(),f)!=a[i].size())return 1;
 return fclose(f)!=0;
}
