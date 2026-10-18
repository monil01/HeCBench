#include "../resnet-kernels-cuda/Kernel128_one.cu"
#include "../resnet-kernels-cuda/Kernel256_one.cu"
#include "../resnet-kernels-cuda/Kernel128_winograd.cu"
#include "../resnet-kernels-cuda/Kernel256_winograd.cu"
#include "../resnet-kernels-cuda/util.cu"
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iomanip>
#include <iostream>
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
std::vector<float> st_input;std::vector<double> gold;
template<class T>std::vector<T> load(const std::string&f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=s.tellg();s.seekg(0);std::vector<T>v(n/sizeof(T));s.read((char*)v.data(),n);return v;}
struct Frame{float *a,*w,*b,*s,*t,*ip,*o;int ci,co,mode,nt,no;};
float* alloc(size_t n){float*p;CK(cudaMalloc(&p,n*4));return p;}
void launch(Frame&f){
 if(f.mode==0){kernel_128_winograd_BtdB<<<dim3(4,4),dim3(128,6),18432>>>(f.a,f.t);kernel_128_OuterProduct_128<<<dim3(36,2),dim3(128,8),40960>>>(f.t,f.w,f.ip);kernel_128_winograd_AtIA<<<dim3(4,4,128),dim3(6,6),144>>>(f.ip,f.b,f.s,f.o);}
 else if(f.mode==1){kernel_256_winograd_BtdB<<<dim3(4,4,2),dim3(128,6),18432>>>(f.a,f.t);kernel_256_OuterProduct_256<<<dim3(36,2),dim3(256,4),49152>>>(f.t,f.w,f.ip);kernel_256_winograd_AtIA<<<dim3(4,4,256),dim3(6,6),144>>>(f.ip,f.b,f.s,f.o);}
 else if(f.mode==2)kernel_512_one_128<<<49,dim3(128,4),44032>>>(f.a,f.w,f.b,f.s,f.o);
 else if(f.mode==3)kernel_128_one_512<<<dim3(49,4),dim3(128,4),37888>>>(f.a,f.w,f.b,f.s,f.o);
 else if(f.mode==4)kernel_1024_one_256<<<49,dim3(256,4),38912>>>(f.a,f.w,f.b,f.s,f.o);
 else kernel_256_one_1024<<<dim3(49,4),dim3(256,4),43008>>>(f.a,f.w,f.b,f.s,f.o);
 CK(cudaGetLastError());
}
int main(int argc,char**argv){int rep=argc>1?atoi(argv[1]):3;std::string base=std::string(__FILE__).substr(0,std::string(__FILE__).find_last_of('/'));
 st_input=load<float>(base+"/fixture.bin");gold=load<double>(base+"/expected_outputs.bin");std::vector<Frame>fs;size_t pos=0;int cis[]={128,256,512,128,1024,256},cos[]={128,256,128,512,256,1024};
 for(int m=0;m<6;++m){Frame f{};f.ci=cis[m];f.co=cos[m];f.mode=m;bool win=m<2;size_t ns[]={size_t((win?290:196)*f.ci),size_t((win?36:1)*f.ci*f.co),size_t(f.co),size_t(f.co)};float**ps[]={&f.a,&f.w,&f.b,&f.s};
 for(int i=0;i<4;++i){*ps[i]=alloc(ns[i]);CK(cudaMemcpy(*ps[i],st_input.data()+pos,ns[i]*4,cudaMemcpyHostToDevice));pos+=ns[i];}
 f.nt=win?576*f.ci:0;f.no=(win?256:196)*f.co;f.t=alloc(f.nt+17);f.ip=alloc(f.nt+17);f.o=alloc(f.no+17);
 for(auto pp:{std::make_pair(f.t,f.nt),std::make_pair(f.ip,f.nt),std::make_pair(f.o,f.no)}){std::vector<float>z(pp.second+17,-12345);if(pp.first==f.o&&win)std::fill(z.begin(),z.begin()+f.no,0);CK(cudaMemcpy(pp.first,z.data(),z.size()*4,cudaMemcpyHostToDevice));}fs.push_back(f);}
 for(auto&f:fs){launch(f);CK(cudaDeviceSynchronize());}
 for(int tr=1;tr<=5;++tr){std::vector<double>times;std::vector<float>got;
 for(auto&f:fs){CK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();for(int k=0;k<rep;++k)launch(f);CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()/rep);
 auto copy=[&](float*p,int n){size_t off=got.size();got.resize(off+n+17);CK(cudaMemcpy(got.data()+off,p,(n+17)*4,cudaMemcpyDeviceToHost));};if(f.mode<2){copy(f.t,f.nt);copy(f.ip,f.nt);}copy(f.o,f.no);}
 bool ok=got.size()==gold.size();double ma=0,mr=0;for(size_t i=0;i<got.size();++i){double e=fabs(got[i]-gold[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(gold[i]),1e-12));ok&=std::isfinite(got[i])&&(gold[i]==-12345?got[i]==-12345:e<=1e-4);}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)st_input.data(),st_input.size()*4);out.write((char*)got.data(),got.size()*4);out.close();double sum=0;for(double t:times)sum+=t;
 std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"time_seconds\":"<<sum<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<got.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"timing_region\":\"sum6_source_convolution_paths_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":"<<rep<<",\"per_path_times_seconds\":[";for(int i=0;i<6;++i)std::cout<<(i?",":"")<<times[i];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float32\",\"count\":"<<st_input.size()+got.size()<<",\"byte_order\":\"little\"}}\n";
 }
}
