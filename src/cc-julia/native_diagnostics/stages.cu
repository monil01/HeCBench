#define main original_cc_main
#include "../../cc-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include "../study_cases.h"
static void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(3);}}
static std::vector<int> load(const std::string&f,int count){std::ifstream s(f,std::ios::binary);std::vector<int>x(count);s.read((char*)x.data(),count*4);if(!s||s.peek()!=EOF)throw std::runtime_error("input size "+f);return x;}
static std::vector<int> host(int*p,int n){std::vector<int>x(n);ck(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
static void cat(std::vector<int>&a,const std::vector<int>&b){a.insert(a.end(),b.begin(),b.end());}
__global__ void observe_params(int n,int e,int blocks,int*p){p[0]=n;p[1]=e;p[2]=blocks;}
struct Work{Case c;int *idx,*adj,*stat,*wl,*params;std::vector<int>A,B;};
static void pipeline(Work&w){auto c=w.c;init<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16);compute1<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);compute2<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);compute3<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16,w.wl+16);flatten<<<c.blocks,256>>>(c.n,w.idx+16,w.adj+16,w.stat+16);}

int main(){for(auto c:cases){auto A=load("case"+std::to_string(c.id)+"_idx_i32.bin",c.ic),B=load("case"+std::to_string(c.id)+"_adj_i32.bin",c.ac);int *a,*b,*s,*wl;ck(cudaMalloc(&a,c.ic*4));ck(cudaMalloc(&b,c.ac*4));ck(cudaMalloc(&s,c.n*4));ck(cudaMalloc(&wl,c.n*4));ck(cudaMemcpy(a,A.data(),c.ic*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(b,B.data(),c.ac*4,cudaMemcpyHostToDevice));for(int stage=0;stage<5;stage++){if(stage==0)init<<<c.blocks,256>>>(c.n,a+16,b+16,s);if(stage==1)compute1<<<c.blocks,256>>>(c.n,a+16,b+16,s,wl);if(stage==2)compute2<<<c.blocks,256>>>(c.n,a+16,b+16,s,wl);if(stage==3)compute3<<<c.blocks,256>>>(c.n,a+16,b+16,s,wl);if(stage==4)flatten<<<c.blocks,256>>>(c.n,a+16,b+16,s);ck(cudaDeviceSynchronize());int tl,th;ck(cudaMemcpyFromSymbol(&tl,topL,4));ck(cudaMemcpyFromSymbol(&th,topH,4));auto x=host(s,c.n);printf("STUDY_TRACE {\"case\":%d,\"stage\":%d,\"topL\":%d,\"topH\":%d,\"labels\":[",c.id,stage,tl,th);for(int i=0;i<c.n;i++)printf("%s%d",i?",":"",x[i]);printf("]}\n");}ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(s));ck(cudaFree(wl));}return 0;}
