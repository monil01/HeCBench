#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#ifdef STUDY_FIX_VECTOR
#include "study_corrected_reverse.cuh"
#else
#include "../reverse2D-cuda/reverse.cuh"
#endif
struct Case {int nr,nc,width,ratio;bool rm,ar;void*in;void*out;std::vector<uint8_t> input,got,expected;};
uint64_t word(int i){uint64_t x=(uint64_t)(i+1)*0x9e3779b97f4a7c15ULL;x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;return x;}
template<class T>void launch(Case&c){
 int len=c.ar?((c.nr+1)/2)*c.nc:c.nr*((c.nc+1)/2);const T*in=(const T*)((uint8_t*)c.in+32);T*out=(T*)((uint8_t*)c.out+32);
 #define GO(V) reverseKernel<T,V><<<(len/V+255)/256,256>>>(out,in,c.nr,c.nc,c.rm,c.ar,len)
 switch(c.ratio){case 16:GO(16);break;case 8:if constexpr(sizeof(T)<=2){GO(8);}break;case 4:if constexpr(sizeof(T)<=4){GO(4);}break;case 2:GO(2);break;default:GO(1);break;}
 #undef GO
}
// Avoid instantiating impossible 16-element vectors for wide types.
template<>void launch<uint16_t>(Case&c){int len=c.ar?((c.nr+1)/2)*c.nc:c.nr*((c.nc+1)/2);auto*in=(uint16_t*)((uint8_t*)c.in+32);auto*out=(uint16_t*)((uint8_t*)c.out+32);
 #define GO(V) reverseKernel<uint16_t,V><<<(len/V+255)/256,256>>>(out,in,c.nr,c.nc,c.rm,c.ar,len)
 switch(c.ratio){case 8:GO(8);break;case 4:GO(4);break;case 2:GO(2);break;default:GO(1);break;}
 #undef GO
}
template<>void launch<uint32_t>(Case&c){int len=c.ar?((c.nr+1)/2)*c.nc:c.nr*((c.nc+1)/2);auto*in=(uint32_t*)((uint8_t*)c.in+32);auto*out=(uint32_t*)((uint8_t*)c.out+32);
 #define GO(V) reverseKernel<uint32_t,V><<<(len/V+255)/256,256>>>(out,in,c.nr,c.nc,c.rm,c.ar,len)
 switch(c.ratio){case 4:GO(4);break;case 2:GO(2);break;default:GO(1);break;}
 #undef GO
}
template<>void launch<uint64_t>(Case&c){int len=c.ar?((c.nr+1)/2)*c.nc:c.nr*((c.nc+1)/2);auto*in=(uint64_t*)((uint8_t*)c.in+32);auto*out=(uint64_t*)((uint8_t*)c.out+32);
 if(c.ratio==2)reverseKernel<uint64_t,2><<<(len/2+255)/256,256>>>(out,in,c.nr,c.nc,c.rm,c.ar,len);else reverseKernel<uint64_t,1><<<(len+255)/256,256>>>(out,in,c.nr,c.nc,c.rm,c.ar,len);
}
int main(int argc,char**argv){bool vector=argc>1&&atoi(argv[1]);int shapes[2][2];if(vector){shapes[0][0]=32;shapes[0][1]=64;shapes[1][0]=64;shapes[1][1]=96;}else{shapes[0][0]=17;shapes[0][1]=19;shapes[1][0]=33;shapes[1][1]=35;}
 std::vector<Case> cases;const int repeats=7;
 for(auto&shape:shapes)for(int width:{1,2,4,8})for(bool rm:{true,false})for(bool ar:{true,false}){
  Case c{};c.nr=shape[0];c.nc=shape[1];c.width=width;c.rm=rm;c.ar=ar;int n=c.nr*c.nc,bytes=n*width;c.input.assign(bytes+64,0xa5);c.expected.assign(bytes+64,0xa5);c.got.resize(bytes+64);
  for(int i=0;i<n;i++){uint64_t x=word(i);memcpy(c.input.data()+32+i*width,&x,width);}
  for(int i=0;i<n;i++){int row=rm?i/c.nc:i%c.nr,col=rm?i%c.nc:i/c.nr;int sr=ar?c.nr-row-1:row,sc=ar?col:c.nc-col-1;int j=rm?sr*c.nc+sc:sc*c.nr+sr;memcpy(c.expected.data()+32+i*width,c.input.data()+32+j*width,width);}
  int bytesrow=(rm?c.nc:c.nr)*width;c.ratio=16/width;while(c.ratio>1&&bytesrow%(c.ratio*width))c.ratio/=2;
  if(cudaMalloc(&c.in,bytes+64)!=cudaSuccess||cudaMalloc(&c.out,bytes+64)!=cudaSuccess)return 2;cudaMemcpy(c.in,c.input.data(),bytes+64,cudaMemcpyHostToDevice);cases.push_back(std::move(c));
 }
 auto suite=[&](){for(auto&c:cases)for(int r=0;r<repeats;r++){switch(c.width){case 1:launch<uint8_t>(c);break;case 2:launch<uint16_t>(c);break;case 4:launch<uint32_t>(c);break;case 8:launch<uint64_t>(c);break;}if(cudaGetLastError()!=cudaSuccess)return false;}return cudaDeviceSynchronize()==cudaSuccess;};
 if(!suite())return 3;bool allok=true;
 for(int trial=1;trial<=5;trial++){
  for(auto&c:cases)cudaMemset(c.out,0xa5,c.got.size());cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();bool ok=suite();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/repeats;
  std::vector<uint8_t> artifact;for(auto&c:cases){std::vector<uint8_t>back(c.input.size());ok &=cudaMemcpy(back.data(),c.in,back.size(),cudaMemcpyDeviceToHost)==cudaSuccess;ok &=back==c.input;artifact.insert(artifact.end(),back.begin(),back.end());}
  size_t mismatches=0;for(auto&c:cases){ok &=cudaMemcpy(c.got.data(),c.out,c.got.size(),cudaMemcpyDeviceToHost)==cudaSuccess;for(size_t i=0;i<c.got.size();i++)mismatches+=c.got[i]!=c.expected[i];artifact.insert(artifact.end(),c.got.begin(),c.got.end());}ok &=mismatches==0;allok &=ok;
  char path[80];snprintf(path,sizeof(path),"output_trial%d.bin",trial);FILE*f=fopen(path,"wb");if(!f)return 4;fwrite(artifact.data(),1,artifact.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"mismatched_output_bytes\":%zu,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"32case_reverse_kernels_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",artifact.size(),mismatches,ok?"0":"null",repeats,path,artifact.size());
 }for(auto&c:cases){cudaFree(c.in);cudaFree(c.out);}puts(allok?"PASS":"FAIL");return allok?0:1;
}
