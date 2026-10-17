// Diagnostic only: unchanged native default repeated-state behavior and finite-aware validation.
#define main original_hecbench_main
#include "../chemv-cuda/main.cu"
#undef main
#include <vector>
int main(){std::vector<ComplexFloat> A(AT_SIZE),X(X_SIZE),cpu(Y_SIZE),gpu(Y_SIZE);for(int i=0;i<N;i++){X[i]={float(i+5),float(i*2)};cpu[i]=gpu[i]={float(i*3),float(i+7)};for(int j=0;j<LDAT;j++)A[i*LDAT+j]={float(i+j),float(i+3)};}chemv_cpu(3.14f,1.59f,2.71f,8.28f,A.data(),X.data(),cpu.data());chemv_gpu(3.14f,1.59f,2.71f,8.28f,A.data(),X.data(),gpu.data());int finite_cpu=0,finite_gpu=0;bool original_mismatch=false;for(int i=0;i<N;i++){finite_cpu+=std::isfinite(cpu[i].Re)+std::isfinite(cpu[i].Im);finite_gpu+=std::isfinite(gpu[i].Re)+std::isfinite(gpu[i].Im);original_mismatch|=fabs(cpu[i].Re-gpu[i].Re)>1e-3||fabs(cpu[i].Im-gpu[i].Im)>1e-3;}printf("NATIVE_DEFAULT_DIAGNOSTIC CPUupdates=1 GPUupdates=1000 finite_CPU_components=%d finite_GPU_components=%d total_components=%d original_abs_comparison_mismatch=%s finite_aware_validation_pass=%s\n",finite_cpu,finite_gpu,2*N,original_mismatch?"true":"false",finite_gpu==2*N&&!original_mismatch?"true":"false");return 0;}
