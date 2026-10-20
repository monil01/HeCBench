// Diagnostic instrumentation only: original cal_km+CLI parsing+GPU calls unchanged.
// Added D2H capture lies inside original timers, so diagnostic times are INVALID.
#define main original_cli_main
#include "../../kmc-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
static std::string outdir;
static int row=0;
template<class T>void store(const std::string&name,const T*p,size_t n,bool append=false){std::ofstream f(outdir+"/"+name,std::ios::binary|(append?std::ios::app:std::ios::trunc));f.write((const char*)p,n*sizeof(T));}
extern "C" cublasStatus_t __real_cublasSgemv_v2(cublasHandle_t,cublasOperation_t,int,int,const float*,const float*,int,const float*,int,const float*,float*,int);
extern "C" cublasStatus_t __wrap_cublasSgemv_v2(cublasHandle_t h,cublasOperation_t op,int m,int n,const float*alpha,const float*A,int lda,const float*x,int incx,const float*beta,float*y,int incy){
 auto status=__real_cublasSgemv_v2(h,op,m,n,alpha,A,lda,x,incx,beta,y,incy);if(status!=CUBLAS_STATUS_SUCCESS)return status;
 cudaError_t e=cudaDeviceSynchronize();if(e!=cudaSuccess)exit(3);
 if(row==0){std::vector<float>a(m*n);if(cudaMemcpy(a.data(),A,a.size()*4,cudaMemcpyDeviceToHost))exit(4);store("actual_A_f32.bin",a.data(),a.size());}
 std::vector<float>v(n),d(m);if(cudaMemcpy(v.data(),x,n*4,cudaMemcpyDeviceToHost)||cudaMemcpy(d.data(),y,m*4,cudaMemcpyDeviceToHost))exit(5);store("actual_vectors_f32.bin",v.data(),v.size(),row!=0);store("actual_dots_f32.bin",d.data(),d.size(),row!=0);row++;return status;
}
int main(int argc,char**argv){outdir=argv[2];std::filesystem::create_directories(outdir);char*p[]={(char*)"kmc",(char*)"-q",(char*)"-v",(char*)"3",argv[1]};char input[1024],modelname[1024];parse_command_line(5,p,input,modelname);read_problem(input);svm_problem kernel;setup_pkm(&kernel);cal_km(&kernel);for(int i=0;i<prob.l;i++)store("actual_kernel_f64.bin",kernel.x[i].values,prob.l+1,i!=0);int geom[]={prob.l,prob.x[0].dim,row};store("geometry_i32.bin",geom,3);std::vector<float>tva;for(int i=0;i<prob.l;i++)for(int j=0;j<prob.x[0].dim;j++)tva.push_back(float(prob.x[i].values[j]));store("parsed_tva_f32.bin",tva.data(),tva.size());float gamma=param.gamma;store("gamma_f32.bin",&gamma,1);free_pkm(&kernel);return 0;}
