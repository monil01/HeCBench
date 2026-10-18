#define main historical_native_main
#include "../ddbp-cuda/main.cu"
#undef main
#include <fstream>
#include <vector>
int main(){std::vector<double>vol(64,-8192),proj(64,1),tube(1,0),det(1,0);backprojectionDDb(vol.data(),proj.data(),tube.data(),det.data(),-1,1,8,8,1,8,8,1,1,1,1,1,8,0,-0.5);auto e=cudaGetLastError();if(e!=cudaSuccess){fprintf(stderr,"CUDA_ERROR %s\n",cudaGetErrorString(e));return 2;}std::ofstream f("/home/imo/HeCBench_julia/.porting-state/portable_julia/diagnostics/ddbp/native_ones_volume_f64.bin",std::ios::binary);f.write((char*)vol.data(),512);for(int x=0;x<8;x++){for(int y=0;y<8;y++)printf("%.17g%c",vol[x*8+y],y==7?'\n':' ');}return 0;}
