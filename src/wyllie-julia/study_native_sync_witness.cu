// Diagnostic only: unchanged native packedpointer kernel, exact independent rank checks.
#define main legacy_main
#include "../wyllie-cuda/main.cu"
#undef main
#include <cstdint>
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):1027,random=argc>2?atoi(argv[2]):1;std::vector<int>next(n),gold(n);if(random)random_list(next);else ordered_list(next);int node=0;for(int rank=n-1;rank>=0;rank--){if(node<0||node>=n)return 3;gold[node]=rank;node=next[node];}if(node!=NIL)return 3;std::vector<long>input(n),got(n);for(int i=0;i<n;i++){unsigned long bits=(uint64_t(uint32_t(next[i]))<<32)|uint32_t(next[i]==NIL?0:1);memcpy(&input[i],&bits,8);}long*device;cudaMalloc(&device,n*8);for(int trial=1;trial<=5;trial++){cudaMemcpy(device,input.data(),n*8,cudaMemcpyHostToDevice);wyllie<<<(n+255)/256,256>>>(device,n);auto err=cudaDeviceSynchronize();cudaMemcpy(got.data(),device,n*8,cudaMemcpyDeviceToHost);int bad=0;for(int i=0;i<n;i++)bad+=((got[i]&MASK)!=gold[i]);printf("WITNESS trial=%d n=%d random=%d wrong_ranks=%d cuda_status=%d\n",trial,n,random,bad,int(err));fflush(stdout);if(bad||err!=cudaSuccess)return 2;}cudaFree(device);}
