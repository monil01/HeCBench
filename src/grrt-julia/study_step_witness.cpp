#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "../grrt-cuda/constants.h"
#undef IMAGE_SIZE
#define IMAGE_SIZE 8
#include "study_cpu_clone.hpp"
#include "study_cpu_clone_corrected.hpp"
int main(){double vin[4]={0,45,IMAGE_SIZE,340e9},source[8*8*3],fixed[8*8*3];for(cpu_y=0;cpu_y<8;cpu_y++)for(cpu_x=0;cpu_x<8;cpu_x++)cpu_task2(source,vin,0,0);for(fixedcpu_y=0;fixedcpu_y<8;fixedcpu_y++)for(fixedcpu_x=0;fixedcpu_x<8;fixedcpu_x++)fixedcpu_task2(fixed,vin,0,0);double ma=0,mr=0;int changed=0;for(int i=0;i<64;i++){double a=source[i*3+2],b=fixed[i*3+2];ma=std::max(ma,fabs(a-b));mr=std::max(mr,fabs(a-b)/std::max(fabs(b),1.0));changed+=a!=b;}printf("{\"changed_pixels\":%d,\"max_absolute_luminosity_difference\":%.17g,\"max_relative_luminosity_difference\":%.17g,\"source_rejected_step_lengths_used\":%lld,\"accepted_step_quad_justification\":\"RK state advances affine parameter by hdid; ds must use that same interval\"}\n",changed,ma,mr,cpu_transfer_rejected_steps);}
