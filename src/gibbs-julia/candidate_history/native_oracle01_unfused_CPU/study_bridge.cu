#define main original_gibbs_main
#include "../../src/gibbs-cuda/main.cu"
#undef main
#include "host_feedback.inc"
#include "cpu_conditional_oracle.h"
#include <cstring>
extern "C" int study_state_bytes(){return sizeof(curandState);}
extern "C" void study_reset(){srand(123);study_sigma=2;}
extern "C" int study_seed_states(void*states,int n){cudaError_t e=cudaMemset(states,0,n*sizeof(curandState));if(e)return e;setup_kernel<<<(n+255)/256,256>>>((curandState*)states,n,123);e=cudaGetLastError();if(e)return e;return cudaDeviceSynchronize();}
extern "C" void study_feedback(float*a,float*b,int n,float flat,float logs){*a=study_sample_a(*a,*b,n,logs);*b=study_sample_b(*a,n,flat);}
extern "C" int study_check(const void*before,const void*after,const float*theta,const int*y,const float*n,float a,float b,int count,double*maxerror){const StudyRNG*bs=(const StudyRNG*)before;const StudyRNG*as=(const StudyRNG*)after;*maxerror=0;int bad=0;for(int i=0;i<count;i++){StudyRNG s=bs[i];float ref=conditional_cpu(s,a+y[i],b+n[i]);double e=fabs(double(theta[i])-ref);*maxerror=fmax(*maxerror,e);bool state=s.d==as[i].d&&s.flag==as[i].flag&&s.flag_double==as[i].flag_double;for(int j=0;j<5;j++)state=state&&s.v[j]==as[i].v[j];if(!state||!std::isfinite(theta[i])||theta[i]<=0||!std::isfinite(ref)||e>3e-5)bad++;}return bad;}
extern "C" int study_native_launch(void*s,float*theta,const int*y,const float*n,float a,float b,int count,float*flat,float*logs){int blocks=(count+255)/256;sample_theta<<<blocks,256>>>((curandState*)s,theta,y,n,a,b,count);cudaError_t e=cudaGetLastError();if(e)return e;sum_blocks<<<blocks,256>>>(theta,flat,logs,count);e=cudaGetLastError();if(e)return e;return cudaDeviceSynchronize();}
