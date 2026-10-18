#define main archived_cli_main
#include "../secp256k1-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
template<class T>std::vector<T>load(const std::string&f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=s.tellg();s.seekg(0);std::vector<T>a(n/sizeof(T));s.read((char*)a.data(),n);return a;}
int main(int argc,char**argv){static_assert(sizeof(secp256k1_ge_storage)==64&&sizeof(uint)==4&&sizeof(ulong)==8);int rep=argc>1?atoi(argv[1]):3;std::string file=__FILE__,base=file.substr(0,file.find_last_of('/'));auto input=load<unsigned>(base+"/fixture.bin");auto gold=load<unsigned char>(base+"/expected_bytes.bin");secp256k1_ge_storage*points[3];unsigned char*out[3];std::vector<unsigned char>guard(49,165);for(int c=0;c<3;++c){CK(cudaMalloc(&points[c],512*64));CK(cudaMalloc(&out[c],49));CK(cudaMemcpy(points[c],input.data()+1+c*8192,512*64,cudaMemcpyHostToDevice));CK(cudaMemcpy(out[c],guard.data(),49,cudaMemcpyHostToDevice));secp256k1<<<1,1>>>(points[c],out[c]);CK(cudaDeviceSynchronize());}
 for(int tr=1;tr<=5;++tr){std::vector<unsigned char>got;std::vector<double>times;
 for(int c=0;c<3;++c){CK(cudaMemcpy(out[c],guard.data(),49,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();for(int k=0;k<rep;++k)secp256k1<<<1,1>>>(points[c],out[c]);CK(cudaGetLastError());CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()/rep);size_t off=got.size();got.resize(off+49);CK(cudaMemcpy(got.data()+off,out[c],49,cudaMemcpyDeviceToHost));}
 bool ok=got==gold;int ma=0;for(size_t i=0;i<got.size();++i)ma=std::max(ma,abs(int(got[i])-int(gold[i])));
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";auto artifact=input;for(auto v:got)artifact.push_back(v);std::ofstream o(path,std::ios::binary);o.write((char*)artifact.data(),artifact.size()*4);o.close();double total=0;for(double t:times)total+=t;
 std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"time_seconds\":"<<total<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<got.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":0,\"timing_region\":\"sum3_complete_512point_field_chains_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":"<<rep<<",\"per_path_times_seconds\":[";for(int c=0;c<3;++c)std::cout<<(c?",":"")<<times[c];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"uint32\",\"count\":"<<artifact.size()<<",\"byte_order\":\"little\"}}\n";}
}
