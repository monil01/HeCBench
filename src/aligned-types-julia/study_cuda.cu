#include <cuda_runtime.h>
#include <vector>
#include <functional>
#define main legacy_main
#include "../aligned-types-cuda/main.cu"
#undef main
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
struct StudyCase{const char*name;int packed,size,alignment;std::function<void()>launch;};
template<class T>StudyCase make_case(const char*name,int packed,unsigned char*dst,unsigned char*src,int bytes){int n=bytes/sizeof(T);printf("STUDY_LAYOUT {\"name\":\"%s\",\"sizeof\":%d,\"alignment\":%d,\"packed_bytes\":%d}\n",name,int(sizeof(T)),int(alignof(T)),packed);return {name,packed,int(sizeof(T)),int(alignof(T)),[=](){testKernel<T><<<(n+255)/256,256>>>((T*)dst,(T*)src,n);}};}
int main(int argc,char**argv){int bytes=196608,repeat=argc>1?atoi(argv[1]):100;std::vector<unsigned char>input(bytes),got(bytes),output;for(int i=0;i<bytes;i++)input[i]=(i*37+(i/256)*13+17)%256;unsigned char *src,*dst;CK(cudaMalloc(&src,bytes));CK(cudaMalloc(&dst,bytes));CK(cudaMemcpy(src,input.data(),bytes,cudaMemcpyHostToDevice));std::vector<StudyCase>cases;
cases.push_back(make_case<uchar_misaligned>("uchar_misaligned",1,dst,src,bytes));
cases.push_back(make_case<uchar4_misaligned>("uchar4_misaligned",4,dst,src,bytes));
cases.push_back(make_case<ushort_misaligned>("ushort_misaligned",2,dst,src,bytes));
cases.push_back(make_case<uint2_misaligned>("uint2_misaligned",8,dst,src,bytes));
cases.push_back(make_case<uint3_misaligned>("uint3_misaligned",12,dst,src,bytes));
cases.push_back(make_case<uint4_misaligned>("uint4_misaligned",16,dst,src,bytes));
cases.push_back(make_case<uint8_misaligned>("uint8_misaligned",32,dst,src,bytes));
cases.push_back(make_case<uchar4_aligned>("uchar4_aligned",4,dst,src,bytes));
cases.push_back(make_case<uint_aligned>("uint_aligned",4,dst,src,bytes));
cases.push_back(make_case<uint2_aligned>("uint2_aligned",8,dst,src,bytes));
cases.push_back(make_case<uint3_aligned>("uint3_aligned",12,dst,src,bytes));
cases.push_back(make_case<uint4_aligned>("uint4_aligned",16,dst,src,bytes));
cases.push_back(make_case<uint8_aligned>("uint8_aligned",32,dst,src,bytes));
for(auto&c:cases)c.launch();CK(cudaDeviceSynchronize());bool all=true;for(int trial=1;trial<=5;trial++){std::vector<double>times;output.clear();bool ok=true;int ma=0;for(auto&c:cases){CK(cudaMemset(dst,0,bytes));CK(cudaDeviceSynchronize());auto t0=std::chrono::steady_clock::now();for(int k=0;k<repeat;k++)c.launch();CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/repeat);CK(cudaMemcpy(got.data(),dst,bytes,cudaMemcpyDeviceToHost));for(int e=0;e<bytes/c.size;e++)for(int j=0;j<c.packed;j++){int idx=e*c.size+j,d=abs(int(got[idx])-int(input[idx]));ma=std::max(ma,d);ok&=d==0;output.push_back(got[idx]);}}all&=ok;char file[100];sprintf(file,"aligned_trial%d.bin",trial);FILE*f=fopen(file,"wb");fwrite(output.data(),1,output.size(),f);fclose(f);double sum=0;for(double t:times)sum+=t;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum_thirteen_ABI_layout_copy_regions_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"layout_times_seconds\":[",trial,sum,ok?"true":"false",int(output.size()),ma,repeat);for(int i=0;i<times.size();i++)printf("%s%.12g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",file,int(output.size()));}CK(cudaFree(src));CK(cudaFree(dst));puts(all?"PASS":"FAIL");return all?0:1;}
