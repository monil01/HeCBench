#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <stdexcept>
#define main archived_main
#include "../kernelLaunch-cuda/main.cu"
#undef main
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class A> __global__ void echo(A args,unsigned char*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=(unsigned char)args.args[i];}
template<class A> void launch(A a,char*out);
template<> void launch(SmallKernelArgs a,char*out){KernelWithSmallArgs<<<1,1>>>(a,out);}
template<> void launch(MediumKernelArgs a,char*out){KernelWithMediumArgs<<<1,1>>>(a,out);}
template<> void launch(LargeKernelArgs a,char*out){KernelWithLargeArgs<<<1,1>>>(a,out);}
template<class A> void diagnostic(int f){A a;for(int i=0;i<(int)sizeof(A);i++)a.args[i]=(char)((37*(i+1)+11*f)%256);int n=sizeof(A);unsigned char*p;check(cudaMalloc(&p,n+32));check(cudaMemset(p,0xa5,n+32));echo<<<(n+255)/256,256>>>(a,p+16,n);check(cudaGetLastError());check(cudaDeviceSynchronize());std::vector<unsigned char>v(n+32);check(cudaMemcpy(v.data(),p,n+32,cudaMemcpyDeviceToHost));for(int i=0;i<n+32;i++)if(v[i]!=(i>=16&&i<n+16?(unsigned char)a.args[i-16]:0xa5))throw std::runtime_error("echo mismatch");check(cudaMemset(p,0xa5,n+32));launch(a,(char*)p+16);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(v.data(),p,n+32,cudaMemcpyDeviceToHost));for(int i=0;i<n+32;i++)if(v[i]!=(i==16?(unsigned char)a.args[0]:0xa5))throw std::runtime_error("nonnull mismatch");check(cudaFree(p));launch(a,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());}
template<class A> double timed(int f){A a;for(int i=0;i<(int)sizeof(A);i++)a.args[i]=(char)((37*(i+1)+11*f)%256);check(cudaDeviceSynchronize());auto s=std::chrono::steady_clock::now();for(int r=0;r<1000;r++)launch(a,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count()/1000;}
int main(){for(int f=0;f<2;f++){diagnostic<SmallKernelArgs>(f);diagnostic<MediumKernelArgs>(f);diagnostic<LargeKernelArgs>(f);}for(int t=1;t<=5;t++){double a[6];for(int f=0;f<2;f++){a[3*f]=timed<SmallKernelArgs>(f);a[3*f+1]=timed<MediumKernelArgs>(f);a[3*f+2]=timed<LargeKernelArgs>(f);}double sum=0;for(double x:a)sum+=x;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":0,\"source_empty_kernel\":false,\"source_no_observable_numerical_outputs\":true,\"no_output_reason\":\"original_null_destination\",\"execution_checked\":true,\"numerical_agreement_kind\":\"no_numerical_outputs_execution_only\",\"max_abs_error\":null,\"max_rel_error\":null,\"numeric_outputs\":[],\"timing_region\":\"sum_six_null_destination_launch_batches_plus_device_completion_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":1000,\"component_time_seconds\":[%.12g,%.12g,%.12g,%.12g,%.12g,%.12g],\"auxiliary_diagnostic\":\"untimed_full_argument_echo_and_guarded_first_byte_store_pass\"}\n",t,sum,a[0],a[1],a[2],a[3],a[4],a[5]);}}
