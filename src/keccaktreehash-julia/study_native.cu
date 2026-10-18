#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "../keccaktreehash-cuda/KeccakTree.h"
#undef NB_INPUT_BLOCK
#define NB_INPUT_BLOCK 3
#undef NB_THREADS_BLOCKS
#define NB_THREADS_BLOCKS 3
#include "../keccaktreehash-cuda/KeccakF.cu"
#include "../keccaktreehash-cuda/KeccakTreeCPU.cu"
#include "../keccaktreehash-cuda/KeccakTreeGPU.cu"
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Case{std::vector<uint32_t>input,output,want,trace;uint32_t*dd;uint32_t*dout;uint32_t*dc;};
int main(){std::vector<Case>cases;std::vector<uint32_t>prefix;
 for(int id=1;id<=2;++id){Case c;c.input.resize(4608);c.output.resize(1536);c.trace.resize(200);for(uint32_t i=0;i<4608;++i)c.input[i]=id==1?0:(i*uint32_t(2654435761))^(i>>3)^uint32_t(0xa5a55a5a);KeccakTreeCPU(c.input.data(),c.output.data());c.want=c.output;uint32_t state[25]={};for(int it=0;it<8;++it){Keccak_top(state,c.output.data(),192);c.want.insert(c.want.end(),state,state+25);}check(cudaMalloc(&c.dd,4608*4));check(cudaMalloc(&c.dout,1536*4));check(cudaMalloc(&c.dc,22*4));check(cudaMemcpy(c.dc,KeccakF_RoundConstantsCPU,22*4,cudaMemcpyHostToDevice));prefix.insert(prefix.end(),c.input.begin(),c.input.end());prefix.insert(prefix.end(),KeccakF_RoundConstantsCPU,KeccakF_RoundConstantsCPU+22);prefix.insert(prefix.end(),25+1536,0);for(uint32_t v:{64,3,3,8,32,32})prefix.push_back(v);cases.push_back(std::move(c));}
 auto run=[&](Case&c,std::vector<uint32_t>&out,bool&ok){check(cudaMemset(c.dout,0,1536*4));uint32_t state[25]={};check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int it=0;it<8;++it){KeccakTreeGPU(c.input.data(),c.dd,c.output.data(),c.dout,c.dc);Keccak_top(state,c.output.data(),192);std::copy(state,state+25,c.trace.begin()+it*25);}check(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<uint32_t>got=c.output;got.insert(got.end(),c.trace.begin(),c.trace.end());ok=ok&&(got==c.want);out.insert(out.end(),got.begin(),got.end());return elapsed;};
 for(auto&c:cases){std::vector<uint32_t>out;bool ok=true;run(c,out,ok);if(!ok)return 3;}
 for(int trial=1;trial<=5;++trial){auto out=prefix;bool ok=true;double elapsed=0;for(auto&c:cases)elapsed+=run(c,out,ok);std::string name="trial_"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":0,\"timing_region\":\"sum_two_eight_pass_transfer_leaf_gpu_root_cpu_pipelines\",\"jit_excluded\":true,\"internal_repeats\":8,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"uint32\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
 for(auto&c:cases){cudaFree(c.dd);cudaFree(c.dout);cudaFree(c.dc);}}
