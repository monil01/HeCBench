#include <cstring>
#define main original_hecbench_main
#include "../fpc-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=262144,repeats=10;bool masked=argc>1&&strcmp(argv[1],"fpc2")==0;const ulong patterns[16]={0x00000000UL,0x0000007fUL,0xffffff80UL,0x0000fffeUL,0xffff0001UL,0x12340000UL,0x00010001UL,0x00aa00bbUL,0xababababUL,0x01010101UL,0x12345678UL,0x80000000UL,0x80000001UL,0xffffffffUL,0x00010000UL,0x00000100UL};const unsigned bytes[16]={1,1,1,2,2,2,2,2,1,1,4,2,4,1,2,2};std::vector<ulong>host(n);std::vector<int>got(1),expected(1,0);for(int i=0;i<n;i++){host[i]=patterns[i%16];expected[0]+=bytes[i%16];}if(FPCCompress(host.data(),n)!=(unsigned)expected[0])return 6;ulong*input;unsigned*output;cudaMalloc(&input,n*sizeof(ulong));cudaMalloc(&output,4);cudaMemcpy(input,host.data(),n*sizeof(ulong),cudaMemcpyHostToDevice);printf("STUDY_PATTERN_CONTRACT words=%d low32_patterns=16 expected_payload_bytes=%d device_container_bytes=%d source_uncompressed_word_bytes=4 observable_scalar_count=1 mode=%s\n",n,expected[0],n*8,masked?"fpc2":"fpc");

 auto launch=[&](){cudaMemsetAsync(output,0,4);if(masked)fpc2_kernel<<<1024,256>>>(input,output);else fpc_kernel<<<1024,256>>>(input,output);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(output,0,4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 unsigned value;cudaMemcpy(&value,output,4,cudaMemcpyDeviceToHost);got[0]=(int)value;
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"counter_reset_and_fpc_kernel_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
