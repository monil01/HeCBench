#define main preserved_native_main
#include "../particles-cuda/main.cu"
#undef main
#include "../particles-cuda/particles.cu"
#include "../particles-cuda/bitonicSort.cu"
#include "study_reference_cpu.h"
int main(){simParams_t p=studyParams();static_assert(sizeof(simParams_t)==104,"params ABI");std::vector<float>pos(4*16384),vel(4*16384);initGrid(pos.data(),vel.data(),p.particleRadius,p.particleRadius*2.f,16384);writeBlob("study_input_pos_f32.bin",pos.data(),pos.size()*4);writeBlob("study_input_vel_f32.bin",vel.data(),vel.size()*4);writeBlob("study_params.bin",&p,sizeof(p));CPUState state;state.pos.assign(pos.begin(),pos.end());state.vel.assign(vel.begin(),vel.end());for(int step=1;step<=8;step++){cpuStep(state,p);if(step==1||step==8){std::vector<double>out=state.pos;out.insert(out.end(),state.vel.begin(),state.vel.end());out.insert(out.end(),state.prevel.begin(),state.prevel.end());char name[100];snprintf(name,100,"study_expected_step%d_f64.bin",step);writeBlob(name,out.data(),out.size()*8);double maxabs=0;bool finite=true;for(double v:out){finite&=std::isfinite(v);maxabs=std::max(maxabs,fabs(v));}printf("CPU_PARTICLES step=%d N=16384 full_all_pairs=1 finite=%d maxstate=%.12g paramsBytes=%zu\n",step,int(finite),maxabs,sizeof(p));fflush(stdout);}}}
