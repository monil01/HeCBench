#define main original_hecbench_main
#include "../layout-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 bool soa=argc>1&&std::string(argv[1])=="soa";int repeats=argc>2?atoi(argv[2]):10;if(repeats<1)return 2;
 const size_t n=(size_t)TREE_NUM*TREE_SIZE;std::vector<int>host(n),expected(TREE_NUM),got(TREE_NUM);
 for(int r=0;r<TREE_NUM;r++){unsigned sum=0;for(int k=0;k<TREE_SIZE;k++){int v=r*TREE_SIZE+k+(r*k)%17-8;host[soa?(size_t)k*TREE_NUM+r:(size_t)r*TREE_SIZE+k]=v;sum+=(unsigned)v;}memcpy(&expected[r],&sum,sizeof(sum));}
 int*a,*b;if(cudaMalloc(&a,n*sizeof(int))!=cudaSuccess||cudaMalloc(&b,TREE_NUM*sizeof(int))!=cudaSuccess)return 3;cudaMemcpy(a,host.data(),n*sizeof(int),cudaMemcpyHostToDevice);
 auto launch=[&](){if(soa)SoAKernel<TREE_SIZE><<<TREE_NUM/GROUP_SIZE,GROUP_SIZE>>>((ApplesOnTrees*)a,b);else AoSKernel<TREE_SIZE><<<TREE_NUM/GROUP_SIZE,GROUP_SIZE>>>((AppleTree*)a,b);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(b,0,TREE_NUM*sizeof(int));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),b,TREE_NUM*sizeof(int),cudaMemcpyDeviceToHost);long long maxabs=0;bool ok=status==cudaSuccess;for(int i=0;i<TREE_NUM;i++){long long e=llabs((long long)got[i]-expected[i]);if(e>maxabs)maxabs=e;ok &=e==0;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),TREE_NUM,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",TREE_NUM,maxabs,repeats,filename,TREE_NUM);
 }cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
