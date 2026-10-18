#define main archived_cli_main
#include "../softmax-online-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
template<class T>std::vector<T>load(const std::string&f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=s.tellg();s.seekg(0);std::vector<T>a(n/sizeof(T));s.read((char*)a.data(),n);return a;}
struct Frame{float*in,*out;int n,c,len;std::vector<float>guard,source,host;};
int main(int argc,char**argv){int rep=argc>1?atoi(argv[1]):20;std::string file=__FILE__,base=file.substr(0,file.find_last_of('/'));auto input=load<float>(base+"/fixture.bin");auto gold=load<double>(base+"/expected_outputs.bin");std::vector<Frame>fs;size_t pos=1;int widths[]={32,64,128,256,512,1024};
 for(int k=0;k<5;++k){Frame f{};f.n=input[pos++];f.c=input[pos++];f.len=input[pos++];f.guard.assign(f.len,-12345);f.host.resize(f.len);f.source.resize(f.len,-12345);softmax_forward_cpu(f.source.data(),input.data()+pos,f.n,f.c);CK(cudaMalloc(&f.in,f.len*4));CK(cudaMalloc(&f.out,f.len*4));CK(cudaMemcpy(f.in,input.data()+pos,f.len*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(f.out,f.guard.data(),f.len*4,cudaMemcpyHostToDevice));pos+=f.len;fs.push_back(f);}
 for(auto&f:fs)for(int w:widths){softmax_forward_online_kernel3<<<f.n,w>>>(f.out,f.in,f.n,f.c);CK(cudaDeviceSynchronize());}
 std::vector<float>got;got.reserve(gold.size());
 for(int tr=1;tr<=5;++tr){got.clear();std::vector<double>times;bool ok=true;double mass=0,shift=0,se=0;
 for(auto&f:fs)for(int w:widths){CK(cudaMemcpy(f.out,f.guard.data(),f.len*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();for(int k=0;k<rep;++k)softmax_forward_online_kernel3<<<f.n,w>>>(f.out,f.in,f.n,f.c);CK(cudaGetLastError());CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()/rep);CK(cudaMemcpy(f.host.data(),f.out,f.len*4,cudaMemcpyDeviceToHost));got.insert(got.end(),f.host.begin(),f.host.end());for(int row=0;row<f.n;++row){double sum=0;for(int j=0;j<f.c;++j){float v=f.host[row*f.c+j];sum+=v;ok&=std::isfinite(v)&&v>=0&&v<=1+1e-6;se=std::max(se,fabs(double(v)-f.source[row*f.c+j]));ok&=fabs(v-f.source[row*f.c+j])<=1e-4+fabs(f.source[row*f.c+j])*FLT_EPSILON;}mass=std::max(mass,fabs(sum-1));ok&=fabs(sum-1)<=1e-5;}for(int dst:{2,3})if(dst<f.n){ok&=memcmp(f.host.data()+f.c,f.host.data()+dst*f.c,f.c*4)==0;for(int j=0;j<f.c;++j)shift=std::max(shift,fabs(double(f.host[f.c+j])-f.host[dst*f.c+j]));}}
 double ma=0,mr=0;for(size_t i=0;i<got.size();++i){double e=fabs(got[i]-gold[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(gold[i]),1e-12));ok&=std::isfinite(got[i])&&(gold[i]==-12345?got[i]==-12345:e<=1e-4+fabs(gold[i])*FLT_EPSILON);}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)input.data(),input.size()*4);o.write((char*)got.data(),got.size()*4);o.close();double total=0;for(double t:times)total+=t;
 std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"time_seconds\":"<<total<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<got.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"max_row_sum_abs_error\":"<<mass<<",\"max_exact_shift_pair_abs_error\":"<<shift<<",\"source_cpu_max_abs_error\":"<<se<<",\"timing_region\":\"sum30_source_block_online_kernel4_paths\",\"jit_excluded\":true,\"internal_repeats\":"<<rep<<",\"per_path_times_seconds\":[";for(int c=0;c<30;++c)std::cout<<(c?",":"")<<times[c];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float32\",\"count\":"<<input.size()+got.size()<<",\"byte_order\":\"little\"}}\n";}
}
