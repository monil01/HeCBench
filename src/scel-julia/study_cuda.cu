#define main archived_cli_main
#include "../scel-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
template<class T>std::vector<T>load(const std::string&f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=s.tellg();s.seekg(0);std::vector<T>a(n/sizeof(T));s.read((char*)a.data(),n);return a;}
struct Frame{float*a,*t,*out;int outer,inner,n;std::vector<float>guard;std::vector<float>source[3];};
int main(int argc,char**argv){int rep=argc>1?atoi(argv[1]):20;std::string file=__FILE__,base=file.substr(0,file.find_last_of('/'));auto input=load<float>(base+"/fixture.bin");auto gold=load<double>(base+"/expected_outputs.bin");std::vector<Frame>fs;size_t pos=1;
 for(int c=0;c<int(input[0]);++c){Frame f{};f.outer=input[pos++];f.inner=input[pos++];f.n=input[pos++];CK(cudaMalloc(&f.a,f.n*4));CK(cudaMalloc(&f.t,f.n*4));CK(cudaMalloc(&f.out,(f.outer+17)*4));CK(cudaMemcpy(f.a,input.data()+pos,f.n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(f.t,input.data()+pos+f.n,f.n*4,cudaMemcpyHostToDevice));for(int mode=0;mode<3;++mode){f.source[mode].resize(f.outer+17,-12345);reference(f.outer,f.inner,mode==1,mode==2,input.data()+pos,input.data()+pos+f.n,f.source[mode].data());}pos+=2*f.n;f.guard.assign(f.outer+17,-12345);CK(cudaMemcpy(f.out,f.guard.data(),(f.outer+17)*4,cudaMemcpyHostToDevice));fs.push_back(f);}
 auto launch=[&](Frame&f,int mode){SigmoidCrossEntropyWithLogitsKernel<<<f.outer,256>>>(f.inner,mode==1,mode==2,f.a,f.t,f.out);};
 for(auto&f:fs)for(int mode=0;mode<3;++mode){launch(f,mode);CK(cudaDeviceSynchronize());}
 for(int tr=1;tr<=5;++tr){std::vector<float>got,source;std::vector<double>times;
 for(auto&f:fs)for(int mode=0;mode<3;++mode){CK(cudaMemcpy(f.out,f.guard.data(),(f.outer+17)*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();for(int k=0;k<rep;++k)launch(f,mode);CK(cudaGetLastError());CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()/rep);size_t off=got.size();got.resize(off+f.outer+17);CK(cudaMemcpy(got.data()+off,f.out,(f.outer+17)*4,cudaMemcpyDeviceToHost));source.insert(source.end(),f.source[mode].begin(),f.source[mode].end());}
 bool ok=got.size()==gold.size();double ma=0,mr=0,se=0;for(size_t i=0;i<got.size();++i){double e=fabs(got[i]-gold[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(gold[i]),1e-12));se=std::max(se,fabs(double(got[i])-source[i]));ok&=std::isfinite(got[i])&&(gold[i]==-12345?got[i]==-12345:e<=1e-3)&&fabs(got[i]-source[i])<=1e-3;}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)input.data(),input.size()*4);o.write((char*)got.data(),got.size()*4);o.close();double total=0;for(double t:times)total+=t;
 std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"time_seconds\":"<<total<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<got.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"source_cpu_max_abs_error\":"<<se<<",\"timing_region\":\"sum15_independent_sigmoid_loss_kernel_paths\",\"jit_excluded\":true,\"internal_repeats\":"<<rep<<",\"per_path_times_seconds\":[";for(int c=0;c<15;++c)std::cout<<(c?",":"")<<times[c];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float32\",\"count\":"<<input.size()+got.size()<<",\"byte_order\":\"little\"}}\n";}
}
