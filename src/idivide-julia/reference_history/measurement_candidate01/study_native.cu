#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include "../idivide-cuda/fastdiv.h"
#include "../idivide-cuda/kernels.h"
template<class D> __global__ void capture(const int*xs,int*out,int n,D d){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int x=xs[i],q=x/d;out[2+i]=q;out[2+n+i]=x-q*(int)d;if(i==0)out[1]=(int)d;}}
int main(){const int g=-123456789,R=5;int ds[]={1,-1,2,-2,3,-3,5,-5,7,-7,11,-11,16,-16,31,-31,127,-127,255,-255,1024,-1024,65535,-65535,65537,-65537,2147483647,-2147483647};FILE*f=fopen("input_i32.bin","rb");if(!f)return 2;fseek(f,0,SEEK_END);int n=ftell(f)/4;rewind(f);std::vector<int>host(n+2,g);fread(host.data()+1,4,n,f);fclose(f);int stride=2*n+3;std::vector<int>got(56*stride,g),checks(28*6,g),back(n+2),oracle(28*2*n);f=fopen("oracle_i32.bin","rb");if(!f)return 3;fread(oracle.data(),4,oracle.size(),f);fclose(f);int *x,*out,*err;cudaMalloc(&x,host.size()*4);cudaMalloc(&out,got.size()*4);cudaMalloc(&err,checks.size()*4);cudaMemcpy(x,host.data(),host.size()*4,cudaMemcpyHostToDevice);
 auto suite=[&](){for(int r=0;r<R;r++)for(int j=0;j<28;j++){capture<int><<<(n+255)/256,256>>>(x+1,out+(j*2)*stride,n,ds[j]);capture<int_fastdiv><<<(n+255)/256,256>>>(x+1,out+(j*2+1)*stride,n,int_fastdiv(ds[j]));}return cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;};if(!suite())return 4;bool all=true;
 for(int t=1;t<=5;t++){std::fill(got.begin(),got.end(),g);cudaMemcpy(out,got.data(),got.size()*4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();bool ok=suite();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost);cudaMemcpy(back.data(),x,back.size()*4,cudaMemcpyDeviceToHost);ok &= back==host;
  // Original unchanged functional check, distinct from timed capture maps.
  for(int j=0;j<28;j++){std::fill(checks.begin()+j*6+1,checks.begin()+j*6+5,0);}cudaMemcpy(err,checks.data(),checks.size()*4,cudaMemcpyHostToDevice);for(int j=0;j<28;j++)check<<<16,256>>>(int_fastdiv(ds[j]),err+j*6+1);ok &=cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;std::vector<int>checkback(checks.size());cudaMemcpy(checkback.data(),err,checks.size()*4,cudaMemcpyDeviceToHost);ok &=checkback==checks;
  int bad=0;for(int j=0;j<28;j++)for(int m=0;m<2;m++){int base=(j*2+m)*stride;bad+=got[base]!=g||got[base+stride-1]!=g||got[base+1]!=ds[j];for(int k=0;k<2*n;k++)bad+=got[base+2+k]!=oracle[j*2*n+k];}ok &=bad==0;all &=ok;std::vector<int>artifact=back;artifact.insert(artifact.end(),got.begin(),got.end());artifact.insert(artifact.end(),checkback.begin(),checkback.end());char path[80];snprintf(path,80,"output_trial%d.bin",t);f=fopen(path,"wb");fwrite(artifact.data(),4,artifact.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"observable_56_quotient_remainder_maps_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",artifact.size(),bad?1:0,R,path,artifact.size());
 }cudaFree(x);cudaFree(out);cudaFree(err);puts(all?"PASS":"FAIL");return all?0:1;}
