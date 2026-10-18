#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "../radixsort-cuda/RadixSort.cu"
#include "../radixsort-cuda/Scan.cu"
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
using U=uint32_t;
std::vector<U> fixture(int n){std::vector<U>a(n);for(U i=0;i<U(n);++i){if(n==32768)a[i]=i%11==0?0:(i%13==0?0xffffffffu:(i%17==0?(0x80000000u|(i&255)):((i*2654435761u)^(i>>7)^0xa5a55a5au)));else a[i]=U((n-1-i)%1024)|(((i/1024)%16)<<28);}return a;}
std::vector<U> digit_sort(const std::vector<U>&a,int begin,int length,int bit){U counts[16]={},offset[16]={};for(int i=0;i<length;++i)++counts[(a[begin+i]>>bit)&15];for(int d=1;d<16;++d)offset[d]=offset[d-1]+counts[d-1];std::vector<U>out(length);for(int i=0;i<length;++i){U key=a[begin+i];out[offset[(key>>bit)&15]++]=key;}return out;}
std::vector<U> oracle(std::vector<U>keys){int n=keys.size(),nb=n/256,cn=n/16;std::vector<U>tmp(n),counts(cn),blocks(cn),scan(cn),buffer(cn/1024);for(int rep=0;rep<3;++rep)for(int bit=0;bit<32;bit+=4){for(int base=0;base<n;base+=512){auto chunk=digit_sort(keys,base,512,bit);std::copy(chunk.begin(),chunk.end(),tmp.begin()+base);}std::fill(counts.begin(),counts.end(),0);std::fill(blocks.begin(),blocks.end(),0);for(int g=0;g<nb;++g)for(int i=0;i<256;++i){U digit=(tmp[g*256+i]>>bit)&15;if(counts[digit*nb+g]==0)blocks[g*16+digit]=i;++counts[digit*nb+g];}U running=0;for(int i=0;i<cn;++i){scan[i]=running;running+=counts[i];}for(int g=0;g<cn/1024;++g)buffer[g]=scan[g*1024];keys=digit_sort(keys,0,n,bit);}std::vector<U>out=keys;for(auto*x:{&tmp,&counts,&blocks,&scan,&buffer})out.insert(out.end(),x->begin(),x->end());return out;}
struct Case{int n,cn,bn;std::vector<U>input,want;U*keys;U*tmp;U*counts;U*blocks;U*scan;U*buffer;};
int main(){std::vector<Case>cases;std::vector<U>prefix;
 for(int n:{32768,65536}){Case c;c.n=n;c.cn=n/16;c.bn=c.cn/1024;c.input=fixture(n);c.want=oracle(c.input);auto sorted=c.input;std::sort(sorted.begin(),sorted.end());if(!std::equal(sorted.begin(),sorted.end(),c.want.begin()))return 6;check(cudaMalloc(&c.keys,n*4));check(cudaMalloc(&c.tmp,n*4));check(cudaMalloc(&c.counts,c.cn*4));check(cudaMalloc(&c.blocks,c.cn*4));check(cudaMalloc(&c.scan,c.cn*4));check(cudaMalloc(&c.buffer,c.bn*4));prefix.insert(prefix.end(),c.input.begin(),c.input.end());prefix.insert(prefix.end(),n+3*c.cn+c.bn,0);for(U v:{U(n),32u,3u,128u,256u,4u,1u})prefix.push_back(v);cases.push_back(std::move(c));}
 auto run=[&](Case&c,std::vector<U>&out,bool&ok){check(cudaMemcpy(c.keys,c.input.data(),c.n*4,cudaMemcpyHostToDevice));check(cudaMemset(c.tmp,0,c.n*4));for(U*x:{c.counts,c.blocks,c.scan})check(cudaMemset(x,0,c.cn*4));check(cudaMemset(c.buffer,0,c.bn*4));check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<3;++rep)radixSortKeys(c.keys,c.tmp,c.counts,c.blocks,c.scan,c.buffer,c.n,32,1);check(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/3;std::vector<U>got;for(auto pair:std::vector<std::pair<U*,int>>{{c.keys,c.n},{c.tmp,c.n},{c.counts,c.cn},{c.blocks,c.cn},{c.scan,c.cn},{c.buffer,c.bn}}){size_t begin=got.size();got.resize(begin+pair.second);check(cudaMemcpy(got.data()+begin,pair.first,pair.second*4,cudaMemcpyDeviceToHost));}if(got!=c.want){size_t index=0;while(index<got.size()&&got[index]==c.want[index])++index;std::cerr<<"Mismatch index "<<index<<" got "<<got[index]<<" expected "<<c.want[index]<<std::endl;ok=false;}out.insert(out.end(),got.begin(),got.end());return elapsed;};
 for(auto&c:cases){std::vector<U>out;bool ok=true;run(c,out,ok);if(!ok)return 3;}
 for(int trial=1;trial<=5;++trial){auto out=prefix;bool ok=true;double elapsed=0;for(auto&c:cases)elapsed+=run(c,out,ok);std::string name="trial_"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":0,\"timing_region\":\"sum_two_three_full48kernel_sort_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"uint32\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
 for(auto&c:cases){for(U*x:{c.keys,c.tmp,c.counts,c.blocks,c.scan,c.buffer})cudaFree(x);}}
