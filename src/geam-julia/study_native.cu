#define main original_hecbench_main
#include "../geam-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <type_traits>
template<class T>int study(){
 const int nr=1025,nc=1537,repeats=10;const size_t n=size_t(nr)*nc;
 std::vector<T>host(n),expected(n),got(n);for(int row=0;row<nr;row++)for(int col=0;col<nc;col++){T v=T((row*7+col*11+row/17)%13);host[size_t(row)*nc+col]=v;expected[size_t(col)*nr+row]=v;}
 T*input,*output;cudaMalloc(&input,n*sizeof(T));cudaMalloc(&output,n*sizeof(T));cudaMemcpy(input,host.data(),n*sizeof(T),cudaMemcpyHostToDevice);cublasHandle_t handle;if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)return 7;T alpha=1,beta=0;
 auto launch=[&](){cublasStatus_t status;if constexpr(std::is_same<T,float>::value)status=cublasSgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,nr,nc,&alpha,input,nc,&beta,input,nr,output,nr);else status=cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,nr,nc,&alpha,input,nc,&beta,input,nr,output,nr);return status;};
 if(launch()!=CUBLAS_STATUS_SUCCESS||cudaDeviceSynchronize()!=cudaSuccess)return 8;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(output,0,n*sizeof(T));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();bool status=true;for(int k=0;k<repeats;k++)status&=launch()==CUBLAS_STATUS_SUCCESS;status&=cudaDeviceSynchronize()==cudaSuccess;double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),output,n*sizeof(T),cudaMemcpyDeviceToHost);bool ok=status;double error=0;for(size_t j=0;j<n;j++){double e=fabs(double(got[j])-expected[j]);error=fmax(error,e);ok&=std::isfinite(double(got[j]))&&e==0;}allok&=ok;
 char name[80];snprintf(name,sizeof(name),"output_trial%d.bin",trial);FILE*f=fopen(name,"wb");if(!f)return 9;fwrite(got.data(),sizeof(T),n,f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"%s\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",n,error,name,std::is_same<T,float>::value?"float32":"float64",n);
 }cublasDestroy(handle);cudaFree(input);cudaFree(output);return allok?0:1;
}
int main(int argc,char**argv){return argc>1&&strcmp(argv[1],"f64")==0?study<double>():study<float>();}
