#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include "../grrt-cuda/constants.h"
#undef IMAGE_SIZE
#define IMAGE_SIZE 8
#include "study_cpu_clone.hpp"
int main(){double input[4]={0,acos(.25)/PI*180,IMAGE_SIZE,340e9};double out[IMAGE_SIZE*IMAGE_SIZE*3];for(int mode=1;mode<=2;mode++){if(mode==2)input[1]=45;auto st=std::chrono::steady_clock::now();for(cpu_y=0;cpu_y<IMAGE_SIZE;cpu_y++)for(cpu_x=0;cpu_x<IMAGE_SIZE;cpu_x++){if(mode==1)cpu_task1(out,input,0,0);else cpu_task2(out,input,0,0);}double maxv=0;int finite=0,positive=0;for(int i=0;i<IMAGE_SIZE*IMAGE_SIZE;i++){double v=out[3*i+2];maxv=fmax(maxv,v);finite+=std::isfinite(v);positive+=v>0;}printf("{\"task\":%d,\"finite_pixels\":%d,\"positive_pixels\":%d,\"max\":%.17g,\"CPU_seconds\":%.9g,\"accepted_steps_cumulative\":%lld,\"shrunk_steps_cumulative\":%lld,\"transfer_steps\":%lld,\"transfer_steps_using_rejected_htry\":%lld,\"max_initial_Schwarzschild_null_residual\":%.17g}\n",mode,finite,positive,maxv,std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count(),cpu_accepted_steps,cpu_shrunk_steps,cpu_transfer_steps,cpu_transfer_rejected_steps,cpu_initial_null_max);}}
