#define main original_hecbench_main
#include "../blas-gemmStridedBatched-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int upper=64,m=32,k=32,batch=128,n=upper*batch,repeats=10;std::vector<double> A(upper*upper*batch),B(n),got(n),expected(n,0.0);for(int i=0;i<(int)A.size();i++)A[i]=(i%31-15)/17.0;for(int i=0;i<n;i++)B[i]=((7*i)%17-8)/19.0;
for(int b=0;b<batch;b++)for(int row=0;row<m;row++){double value=0;for(int col=0;col<k;col++)value+=A[b*upper*upper+row+col*upper]*B[b*upper+col];expected[b*upper+row]=value;}
double *da,*db,*dc;cudaMalloc(&da,A.size()*8);cudaMalloc(&db,n*8);cudaMalloc(&dc,n*8);cudaMemcpy(da,A.data(),A.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(db,B.data(),n*8,cudaMemcpyHostToDevice);cublasHandle_t h;if(cublasCreate(&h)!=CUBLAS_STATUS_SUCCESS)return 7;double alpha=1,beta=0;

 auto launch=[&](){if(cublasDgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,1,k,&alpha,da,upper,upper*upper,db,upper,upper,&beta,dc,upper,upper,batch)!=CUBLAS_STATUS_SUCCESS)std::abort();};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(dc,0,n*8);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),dc,n*8,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-06));ok&=std::isfinite((double)got[j])&&e<=1e-06;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(double),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
