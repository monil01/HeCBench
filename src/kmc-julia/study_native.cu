// Source-faithful cal_km library/host pipeline, named resident fixture.
// Original src/kmc-cuda/main.cu is immutable; source API ordering/precision retained.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDAerror%d line%d\n",int(e),__LINE__);exit(1);}}while(0)
#define B(x) do{auto e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLASerror%d line%d\n",int(e),__LINE__);exit(2);}}while(0)
template<class T>std::vector<T>read(const char*name){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)exit(3);size_t n=f.tellg()/sizeof(T);f.seekg(0);std::vector<T>a(n);f.read((char*)a.data(),n*sizeof(T));if(!f)exit(4);return a;}
struct Device{int n;float*base,*ptr;std::vector<float>initial;Device(int size):n(size),initial(size+32,0){for(int i=0;i<16;i++)initial[i]=initial[n+16+i]=-2047.5f;C(cudaMalloc(&base,(n+32)*4));ptr=base+16;reset();}void reset(){C(cudaMemcpy(base,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}std::vector<float>snapshot(){std::vector<float>x(n+32);C(cudaMemcpy(x.data(),base,x.size()*4,cudaMemcpyDeviceToHost));return x;}~Device(){cudaFree(base);}};
void append(std::vector<double>&out,const std::vector<float>&a){out.insert(out.end(),a.begin(),a.end());}
int main(){const int dim=5001;auto all=read<float>("input_rows_f32.bin");auto prefixRef=read<double>("oracle_prefix_f64.bin");auto expected=read<double>("oracle_output_f64.bin");auto budget=read<double>("oracle_budget_f64.bin");
cublasHandle_t handle;B(cublasCreate(&handle));int version=0;B(cublasGetVersion(handle,&version));cublasMath_t math;B(cublasGetMathMode(handle,&math));Dl_info info={};dladdr((void*)cublasSgemv,&info);printf("STUDY_ENV cublas_version=%d math_mode=%d loaded_path=%s\n",version,int(math),info.dli_fname);fflush(stdout);
const float alpha=1,beta=0,gamma=float(1.0/5000);const double g=double(gamma);
for(int trial=0;trial<=5;trial++){std::vector<double>prefix,outputs;double seconds=0,maxerr=0,maxrelative=0;bool correct=true;
for(int n:{1,257,513}){std::vector<float>tva(all.begin(),all.begin()+n*dim),tr_ar(n*dim);for(int i=0;i<n;i++)for(int j=0;j<dim;j++)tr_ar[j*n+i]=tva[i*dim+j];std::vector<double>sq(n,0),kernel(n*(n+1));for(int i=0;i<n;i++)for(int j=0;j<dim;j++)sq[i]+=pow(tva[i*dim+j],float(2.0));
Device A(n*dim),x(dim),y(n);C(cudaMemcpy(A.ptr,tr_ar.data(),tr_ar.size()*4,cudaMemcpyHostToDevice));prefix.insert(prefix.end(),{double(n),double(dim),double(gamma)});append(prefix,A.snapshot());append(prefix,y.snapshot());std::vector<double>inputVectors,dots;std::vector<float>dp(n);
for(int row=0;row<n;row++){C(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();auto s1=cudaMemcpy(x.ptr,&tva[row*dim],dim*4,cudaMemcpyHostToDevice);auto s2=cublasSgemv(handle,CUBLAS_OP_N,n,dim,&alpha,A.ptr,n,x.ptr,1,&beta,y.ptr,1);auto s3=cudaMemcpy(dp.data(),y.ptr,n*4,cudaMemcpyDeviceToHost);auto stop=std::chrono::steady_clock::now();seconds+=std::chrono::duration<double>(stop-start).count();C(s1);B(s2);C(s3);
// Unchanged source CPU RBF calculation is outside the summed offload intervals.
kernel[row*(n+1)]=row+1;for(int col=0;col<n;col++)kernel[row*(n+1)+col+1]=exp(-g*(sq[row]+sq[col]-double(2)*double(dp[col])));
append(inputVectors,x.snapshot());auto yy=y.snapshot();for(int col=0;col<n;col++)correct=correct&&dp[col]==yy[col+16];append(dots,yy);}
prefix.insert(prefix.end(),inputVectors.begin(),inputVectors.end());outputs.insert(outputs.end(),sq.begin(),sq.end());outputs.insert(outputs.end(),dots.begin(),dots.end());outputs.insert(outputs.end(),kernel.begin(),kernel.end());}
correct=correct&&prefix.size()==prefixRef.size()&&memcmp(prefix.data(),prefixRef.data(),prefix.size()*8)==0&&outputs.size()==expected.size();for(size_t i=0;i<outputs.size();i++){double err=fabs(outputs[i]-expected[i]);maxerr=fmax(maxerr,err);if(expected[i]!=0)maxrelative=fmax(maxrelative,err/fabs(expected[i]));if(!std::isfinite(outputs[i])||err>budget[i]){if(correct)fprintf(stderr,"CPUoracle failure i%zu actual%.17g expected%.17g error%.17g budget%.17g\n",i,outputs[i],expected[i],err,budget[i]);correct=false;}}
if(!correct)return 5;if(trial==0)continue;prefix.insert(prefix.end(),outputs.begin(),outputs.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/kmc_trial"+std::to_string(trial)+".bin";std::ofstream file(path,std::ios::binary);file.write((char*)prefix.data(),prefix.size()*8);file.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"jit_excluded\":true,\"timing_region\":\"sum771_rowH2D_SGEMV_dotD2H_completion_excludes_hostRBF_and_capture\",\"internal_repeats\":771,\"iterations\":771,\"authored_device_kernel_definitions\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,prefix.size(),maxerr,maxrelative,path.c_str(),prefix.size());fflush(stdout);}
B(cublasDestroy(handle));return 0;}
