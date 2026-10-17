#include "study_eigenvalue_oracle.h"
#define main original_hecbench_main
#include "../eigenvalue-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=256,repeats=10;const float tol=.001f;std::vector<float>diag(n),off(n-1),initial(2*n),h0(2*n),h1(2*n),c0(2*n),c1(2*n),got(3*n),expected(3*n);for(int i=0;i<n;i++)diag[i]=4.f+(float)i/2.f;for(int i=0;i<n-1;i++)off[i]=(float)(8+i%5)/256.f;float lower,upper;computeGerschgorinInterval(&lower,&upper,diag.data(),off.data(),n);std::fill(initial.begin(),initial.end(),upper);initial[0]=lower;c0=initial;c1=initial;float*cpu[2]={c0.data(),c1.data()};int ci=0,cpuiters=0;while(isComplete(cpu[ci],n,tol)){eigenValueCPUReference(diag.data(),off.data(),n,cpu[ci],cpu[1-ci],tol);ci=1-ci;if(++cpuiters>=100)return 10;}for(int i=0;i<2*n;i++)expected[i]=cpu[ci][i];for(int i=2*n;i<3*n;i++)expected[i]=1.f;
float *dd,*dof,*buf[2];uint*counts;cudaMalloc(&dd,n*4);cudaMalloc(&dof,(n-1)*4);cudaMalloc(&counts,n*4);cudaMalloc(&buf[0],n*2*4);cudaMalloc(&buf[1],n*2*4);cudaMemcpy(dd,diag.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dof,off.data(),(n-1)*4,cudaMemcpyHostToDevice);uint in=0;float*host[2]={h0.data(),h1.data()};int history[51]={0},call=0;
// Original runKernels mutates host interval buffers: preserve observed restart condition after one solve.
h0=initial;h1=initial;host[0]=h0.data();host[1]=h1.data();runKernels(dd,counts,dof,buf,host,n,tol,in);float width_after=0;for(int j=0;j<n;j++)width_after=fmaxf(width_after,h0[2*j+1]-h0[2*j]);printf("ORIGINAL_HOST_MUTATION initial_max_width=%.12g after_solve_buffer0_max_width=%.12g next_run_start_incomplete=%d full_CPU_iterations=%d\n",upper-lower,width_after,isComplete(host[0],n,tol),cpuiters);

 auto launch=[&](){std::copy(initial.begin(),initial.end(),h0.begin());std::copy(initial.begin(),initial.end(),h1.begin());cudaMemcpy(buf[0],host[0],n*2*4,cudaMemcpyHostToDevice);cudaMemcpy(buf[1],host[1],n*2*4,cudaMemcpyHostToDevice);in=0;int steps=0;while(isComplete(host[in],n,tol)){calNumEigenValueInterval<<<1,256>>>(counts,buf[in],dd,dof,n);recalculateEigenIntervals<<<1,256>>>(buf[1-in],buf[in],counts,dd,dof,n,tol);in=1-in;cudaMemcpy(host[in],buf[in],n*2*4,cudaMemcpyDeviceToHost);if(++steps>=100)break;}history[call++]=steps;};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 std::copy(host[in],host[in]+2*n,got.begin());std::vector<uint>hc(n);cudaMemcpy(hc.data(),counts,n*4,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++)got[2*n+i]=hc[i];double distance=0,maxwidth=0;for(int i=0;i<n;i++){double lo=got[2*i],hi=got[2*i+1],ev=independent_eigenvalues[i];if(!(std::isfinite(lo)&&std::isfinite(hi)&&lo<=hi&&hi-lo<tol))return 11;distance=fmax(distance,fmax(fmax(lo-ev,ev-hi),0.));maxwidth=fmax(maxwidth,hi-lo);}if(distance>1e-4||!compare(cpu[ci],host[in],2*n))return 12;printf("STUDY_EIGEN_ACCURACY max_enclosure_distance=%.12g max_width=%.12g CPU_iterations=%d last10_solve_iterations=[",distance,maxwidth,cpuiters);for(int i=call-10;i<call;i++)printf("%s%d",i==call-10?"":",",history[i]);puts("]");
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-06));ok&=std::isfinite((double)got[j])&&e<=1e-06;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"restarted_complete_host_transfer_sturm_bisection_solve_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
