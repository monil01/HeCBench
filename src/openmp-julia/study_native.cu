#define main legacy_main
#include "openmp-cuda/main.cu"
#undef main
#include <vector>
#include <chrono>
int main(){const int n=262144;std::vector<int>host(n);auto run=[&](int threads){omp_set_dynamic(0);omp_set_num_threads(threads);
#pragma omp parallel
{int lane=omp_get_thread_num(),count=omp_get_num_threads(),per=n/count;int*sub=host.data()+lane*per;for(int i=0;i<per;i++)sub[i]=i+lane*per;int*d;cudaMalloc(&d,per*4);cudaMemcpy(d,sub,per*4,cudaMemcpyHostToDevice);addConstant<<<per/256,256>>>(d,3,37);cudaMemcpy(sub,d,per*4,cudaMemcpyDeviceToHost);cudaFree(d);}
if(cudaDeviceSynchronize()!=cudaSuccess)exit(2);};int fs[]={1,2,4,8,16,32};for(int f:fs)run(f);for(int trial=1;trial<=5;trial++){double total=0;bool ok=true;std::vector<int>all;for(int outer=0;outer<2;outer++)for(int f:fs){cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();run(f);total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();for(int i=0;i<n;i++)ok=ok&&host[i]==i+36;all.insert(all.end(),host.begin(),host.end());}char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*out=fopen(path,"wb");fwrite(all.data(),4,all.size(),out);fclose(out);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.15g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"timing_region\":\"sum_twelve_host_concurrency_end_to_end_cases\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",all.size(),ok?0:1,path,all.size());if(!ok)return 1;}}
