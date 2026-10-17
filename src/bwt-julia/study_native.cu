#define main original_hecbench_main
#include <algorithm>
#include "../bwt-cuda/bwt.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=1021,table_size=1024,repeats=3;std::vector<unsigned char> sequence(n),output(n);uint32_t state=0x12345678;const char alphabet[4]={'A','T','C','G'};for(int i=0;i<n;i++){state=state*1664525u+1013904223u;sequence[i]=alphabet[(state>>28)&3u];}
std::string s(sequence.begin(),sequence.end());std::vector<std::string> rotations(n);std::vector<int> order(n),hosttable(table_size);for(int i=0;i<n;i++){rotations[i]=s.substr(i)+s.substr(0,i);order[i]=i;}std::sort(order.begin(),order.end(),[&](int a,int b){return rotations[a]<rotations[b];});for(int i=1;i<n;i++)if(rotations[order[i-1]]==rotations[order[i]])return 8;
std::vector<int> got(n+table_size),expected(n+table_size,-1);for(int i=0;i<n;i++){expected[i]=sequence[(n+order[i]-1)%n];expected[n+i]=order[i];}
int *table;char *input,*out;cudaMalloc(&table,table_size*4);cudaMalloc(&input,n);cudaMalloc(&out,n);cudaMemcpy(input,sequence.data(),n,cudaMemcpyHostToDevice);

 auto launch=[&](){generate_table<<<4,256>>>(table,table_size,n);for(int k=2;k<=table_size;k<<=1)for(int j=k>>1;j>0;j>>=1)bitonic_sort_step<<<4,256>>>(table,table_size,j,k,input,n);reconstruct_sequence<<<4,256>>>(table,input,out,n);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(table,0,table_size*4);cudaMemset(out,0,n);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(output.data(),out,n,cudaMemcpyDeviceToHost);cudaMemcpy(hosttable.data(),table,table_size*4,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++)got[i]=output[i];for(int i=0;i<table_size;i++)got[n+i]=hosttable[i];
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
