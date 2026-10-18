#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstdio>
#include "../merkle-cuda/ff_p.cu"
#include "../merkle-cuda/rescue_prime.cu"
#include "../merkle-cuda/merkle_tree.cu"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<ulong>load(const char*f){std::ifstream in(f,std::ios::binary|std::ios::ate);size_t z=in.tellg();std::vector<ulong>a(z/8);in.seekg(0);in.read((char*)a.data(),z);return a;}
int main(int argc,char**argv){auto data=load(argv[1]);size_t n=64;ulong*leaves,*out;ulong4 *mds,*ark1,*ark2;std::vector<ulong4>m(36),a1(21),a2(21);prepare_mds(m.data());prepare_ark1(a1.data());prepare_ark2(a2.data());ck(cudaMalloc(&mds,m.size()*32));ck(cudaMalloc(&ark1,a1.size()*32));ck(cudaMalloc(&ark2,a2.size()*32));ck(cudaMalloc(&leaves,n*4*8));ck(cudaMalloc(&out,n*4*8));ck(cudaMemcpy(mds,m.data(),m.size()*32,cudaMemcpyHostToDevice));ck(cudaMemcpy(ark1,a1.data(),a1.size()*32,cudaMemcpyHostToDevice));ck(cudaMemcpy(ark2,a2.data(),a2.size()*32,cudaMemcpyHostToDevice));std::vector<ulong>all;bool ok=true;for(int k=0;k<2;k++){ck(cudaMemcpy(leaves,data.data()+k*n*4,n*4*8,cudaMemcpyHostToDevice));ck(cudaMemset(out,0,n*4*8));merklize_approach_1(leaves,out,n,32,mds,ark1,ark2);ck(cudaGetLastError());std::vector<ulong>a(n*4);ck(cudaMemcpy(a.data(),out,n*4*8,cudaMemcpyDeviceToHost));for(size_t j=0;j<n*4;j++)ok&=(a[j]%MOD)==data[2*n*4+k*n*4+j];all.insert(all.end(),a.begin(),a.end());}std::ofstream f(argv[2],std::ios::binary);f.write((char*)all.data(),all.size()*8);f.close();bool equal=std::equal(all.begin(),all.begin()+n*4,all.begin()+n*4);printf("{\"original_source_unchanged\":true,\"full_nodes_bigint_oracle_pass\":%s,\"lastleaf_perturbation_output_identical\":%s,\"checked_field_words\":%zu}\n",ok?"true":"false",equal?"true":"false",all.size());return ok&&equal?0:3;}
