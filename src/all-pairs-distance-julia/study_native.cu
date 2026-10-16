// Named k3 (CUB reduction) study. Native k1/k2 are not measured or translated.
#define main original_hecbench_main
#include "../all-pairs-distance-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 const int repeats=argc>1?atoi(argv[1]):10;if(repeats<1)return 2;
 const size_t ni=(size_t)INSTANCES*ATTRIBUTES,no=(size_t)INSTANCES*INSTANCES;
 std::vector<char> host(ni);std::vector<int> data(ni),expected(no,0),got(no);
 for(int r=0;r<INSTANCES;r++)for(int k=0;k<ATTRIBUTES;k++){int v=((17*k)^(13*r)^(k>>3))%3;host[r*ATTRIBUTES+k]=v;data[r*ATTRIBUTES+k]=v;}
 CPU(data.data(),expected.data());char*a;int*b;if(cudaMalloc(&a,ni)!=cudaSuccess||cudaMalloc(&b,no*sizeof(int))!=cudaSuccess)return 3;cudaMemcpy(a,host.data(),ni,cudaMemcpyHostToDevice);
 auto launch=[&](){k3<<<dim3(INSTANCES,INSTANCES),THREADS>>>(a,b);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(b,0,no*sizeof(int));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),b,no*sizeof(int),cudaMemcpyDeviceToHost);long long maxabs=0;bool ok=status==cudaSuccess;for(size_t i=0;i<no;i++){long long e=llabs((long long)got[i]-expected[i]);if(e>maxabs)maxabs=e;ok &=e==0;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),no,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",no,maxabs,repeats,filename,no);
 }cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
