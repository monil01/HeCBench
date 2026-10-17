// Actual unchanged nativeCPU/GPU behavior; independent exact polynomial witnesses.
#include <cuda_runtime.h>
#include "../qem-cuda/gpu_solver.cu"
#include "../qem-cuda/reference.h"
#include <cstdio>
#include <cmath>
int main(){constexpr int n=5;float A[n]={1,1,1,-1,1},B[n]={-9,0,0,0,1},C[n]={28,0,.01f,0,1},D[n]={-33,0,0,1,1},E[n]={0},cpu[n],gpu[n],streamed[n];QuarticMinimumCPU(n,A,B,C,D,E,cpu);QuarticMinimumGPU(n,A,B,C,D,E,gpu);QuarticMinimumGPUStreams(n,A,B,C,D,E,streamed);for(int i=0;i<n;i++){auto f=[&](double x){return ((double(A[i])*x+B[i])*x+C[i])*x*x+D[i]*x+E[i];};printf("WITNESS case=%d A=%.9g B=%.9g C=%.9g D=%.9g CPU_x=%.17g GPU_x=%.17g STREAMS_x=%.17g GPU_F=%.17g finite_cpu=%d finite_gpu=%d finite_streams=%d legacy_checker_accepts=%d\n",i,A[i],B[i],C[i],D[i],cpu[i],gpu[i],streamed[i],f(gpu[i]),int(std::isfinite(cpu[i])),int(std::isfinite(gpu[i])),int(std::isfinite(streamed[i])),int(!(fabsf(gpu[i]-cpu[i])>1e-3f)));}printf("EXACT_PROOF case0 F(x)=x4-9x3+28x2-33x; Fprime=4(x-1)(x-2.75)(x-3); F(1)=-13 F(3)=-9; Apositivecoercive; true_global_x=1; source_wrong_comparison_delta=-12 selects3\n");printf("EXACT_PROOF case1 x4 true_global_x=0; case2 x4+.01x2 true_global_x=0; case3 -x4+x unbounded_below_no_global_minimum; case4 x4+x3+x2+x unique_global_root\n");return cudaGetLastError()!=cudaSuccess;}
