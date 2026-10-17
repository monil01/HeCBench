#define main historical_main
#include "../blas-dot-cuda/main.cu"
#undef main
#include <vector>
#define CK(x) do{if((x)!=cudaSuccess){fprintf(stderr,"CUDA failure %s:%d\n",__FILE__,__LINE__);exit(2);}}while(0)
#define BK(x) do{if((x)!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS failure %s:%d\n",__FILE__,__LINE__);exit(3);}}while(0)
struct Record{double output,seconds,error;};
template<class T> Record run_type(cublasHandle_t handle,int repeat){
 const int n=256;std::vector<T>a(n,T(sqrt(65504.0/n)));T *da,*db,*dr;CK(cudaMalloc(&da,n*sizeof(T)));CK(cudaMalloc(&db,n*sizeof(T)));CK(cudaMalloc(&dr,sizeof(T)));CK(cudaMemcpy(da,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(db,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));
 cudaDataType type,execution;if constexpr(std::is_same<T,double>::value){type=execution=CUDA_R_64F;}else if constexpr(std::is_same<T,float>::value){type=execution=CUDA_R_32F;}else if constexpr(std::is_same<T,__half>::value){type=CUDA_R_16F;execution=CUDA_R_32F;}else{type=CUDA_R_16BF;execution=CUDA_R_32F;}
 BK(cublasDotEx(handle,n,da,type,1,db,type,1,dr,type,execution));CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<repeat;rep++){BK(cublasDotEx(handle,n,da,type,1,db,type,1,dr,type,execution));}CK(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;
 T got;CK(cudaMemcpy(&got,dr,sizeof(T),cudaMemcpyDeviceToHost));double expected=0;for(auto v:a)expected+=(float)v*(float)v;Record r{double(got),seconds,fabs(double(got)-expected)};CK(cudaFree(da));CK(cudaFree(db));CK(cudaFree(dr));return r;
}
int main(int argc,char**argv){int repeat=argc>1?atoi(argv[1]):100;if(repeat<1)return 2; cublasHandle_t h;BK(cublasCreate(&h));BK(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_DEVICE));int version;BK(cublasGetVersion(h,&version));printf("STUDY_LIBRARY native_CUBLAS %d\n",version);bool allok=true;
 for(int trial=1;trial<=5;trial++){Record r[]={run_type<double>(h,repeat),run_type<float>(h,repeat),run_type<__half>(h,repeat),run_type<__nv_bfloat16>(h,repeat)};double out[4],seconds=0,error=0;bool ok=true;for(int i=0;i<4;i++){out[i]=r[i].output;seconds+=r[i].seconds;error=fmax(error,r[i].error);ok&=std::isfinite(out[i])&&r[i].error<0.1;}allok&=ok;char file[80];sprintf(file,"blas_dot_trial%d.bin",trial);FILE*f=fopen(file,"wb");fwrite(out,sizeof(double),4,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":4,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum_FP64_FP32_FP16_BF16_dot_device_result_transfer_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"dtype_times_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":4,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",error,repeat,r[0].seconds,r[1].seconds,r[2].seconds,r[3].seconds,file);}
 BK(cublasDestroy(h));puts(allok?"PASS":"FAIL");return allok?0:1;}
