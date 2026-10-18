#define main hecbench_original_main
#include "../popcount-cuda/main.cu"
#undef main
#include <vector>
#include <cstdint>
static_assert(sizeof(unsigned long)==8,"LP64 UInt64 native input contract");
int main(){
 const int n=4099,repeats=11;const int32_t guard=-1234567890;
 std::vector<unsigned long> host(n),inputback(n);FILE*f=fopen("study_input.bin","rb");if(!f)return 2;
 if(fread(host.data(),8,n,f)!=(size_t)n)return 3;fclose(f);
 std::vector<int32_t> expected(n),got(6*(n+2)),artifact(2*n+got.size());
 for(int i=0;i<n;i++)expected[i]=popcount_ref(host[i]);
 unsigned char table[256];for(int i=0;i<256;i++)table[i]=popcount_ref(i);
 unsigned long* data;int32_t*out;if(cudaMalloc(&data,n*8)!=cudaSuccess||cudaMalloc(&out,got.size()*4)!=cudaSuccess)return 4;
 if(cudaMemcpy(data,host.data(),n*8,cudaMemcpyHostToDevice)!=cudaSuccess)return 5;
 auto suite=[&](){for(int which=1;which<=6;which++)for(int r=0;r<repeats;r++){
   int32_t*target=out+(which-1)*(n+2)+1;
   switch(which){case 1:pc1<<<(n+255)/256,256>>>(data,target,n);break;
    case 2:pc2<<<(n+255)/256,256>>>(data,target,n);break;
    case 3:pc3<<<(n+255)/256,256>>>(data,target,n);break;
    case 4:pc4<<<(n+255)/256,256>>>(data,target,n);break;
    case 5:if(cudaMemcpyToSymbol(lut,table,256)!=cudaSuccess)return false;pc5<<<(n+255)/256,256>>>(data,target,n);break;
    case 6:pc6<<<(n+255)/256,256>>>(data,target,n);break;}
   if(cudaGetLastError()!=cudaSuccess)return false;
 }return cudaDeviceSynchronize()==cudaSuccess;};
 if(!suite())return 6;bool allok=true;
 for(int trial=1;trial<=5;trial++){
  std::fill(got.begin(),got.end(),guard);cudaMemcpy(out,got.data(),got.size()*4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();bool ok=suite();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  ok &=cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess;
  ok &=cudaMemcpy(inputback.data(),data,n*8,cudaMemcpyDeviceToHost)==cudaSuccess;ok &=inputback==host;
  for(int w=0;w<6;w++){ok &=got[w*(n+2)]==guard&&got[w*(n+2)+n+1]==guard;for(int i=0;i<n;i++)ok &=got[w*(n+2)+i+1]==expected[i];}
  memcpy(artifact.data(),inputback.data(),n*8);memcpy(artifact.data()+2*n,got.data(),got.size()*4);
  char path[80];snprintf(path,sizeof(path),"output_trial%d.bin",trial);f=fopen(path,"wb");if(!f)return 7;if(fwrite(artifact.data(),4,artifact.size(),f)!=artifact.size())return 8;fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"six_algorithms_including_pc5_LUT_upload_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",artifact.size(),ok?"0":"null",repeats,path,artifact.size());allok &=ok;
 }cudaFree(data);cudaFree(out);puts(allok?"PASS":"FAIL");return allok?0:1;
}
