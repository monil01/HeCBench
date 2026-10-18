#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#define main preserved_original_main
#include "../resize-cuda/main.cu"
#undef main
#ifndef OUTDIR
#error OUTDIR required
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
int main(){
 for(int ci=0;ci<3;++ci){int iw=ci==0?4:5,ih=ci==0?4:1,ow=ci==0?2:(ci==1?3:5),oh=ci==0?2:1;std::vector<unsigned char>h(8*iw*ih),v(8*ow*oh);for(int c=0;c<8;++c)for(int y=0;y<ih;++y)for(int x=0;x<iw;++x)h[(c*ih+y)*iw+x]=(unsigned char)(ci==0?10*x+20*y+c:5*c+x);unsigned char *a,*b;CK(cudaMalloc(&a,h.size()));CK(cudaMalloc(&b,v.size()));CK(cudaMemcpy(a,h.data(),h.size(),cudaMemcpyHostToDevice));CK(cudaMemset(b,255,v.size()));const float fx=iw/ow,fy=ih/oh;
  if(ci==0)resize_bilinear<unsigned char,8><<<29184,256>>>(b,v.size(),oh,ow,a,ih,iw,fy,fx,true);else resize<unsigned char,8><<<29184,256>>>(b,v.size(),oh,ow,a,ih,iw,fy,fx,true,true);CK(cudaDeviceSynchronize());CK(cudaMemcpy(v.data(),b,v.size(),cudaMemcpyDeviceToHost));
  std::string path=std::string(OUTDIR)+"/case"+std::to_string(ci+1)+"_input_output_u8.bin";std::ofstream f(path,std::ios::binary);f.write((char*)h.data(),h.size());f.write((char*)v.data(),v.size());f.close();printf("RESIZE_WITNESS {\"case\":%d,\"input_width\":%d,\"input_height\":%d,\"output_width\":%d,\"output_height\":%d,\"channels\":8,\"native_host_fx\":%.9g,\"native_host_fy\":%.9g,\"path\":\"%s\",\"input_bytes\":%zu,\"output_bytes\":%zu,\"channel0_output\":[",ci+1,iw,ih,ow,oh,fx,fy,path.c_str(),h.size(),v.size());for(int i=0;i<ow*oh;++i)printf("%s%u",i?",":"",(unsigned)v[i]);printf("]}\n");CK(cudaFree(a));CK(cudaFree(b));
 }
}
