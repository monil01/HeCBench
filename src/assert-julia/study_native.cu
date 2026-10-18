#define main original_assert_main
#include "../assert-cuda/main.cu"
#undef main
#include <cstring>
int main(int argc,char**argv){
 if(argc>1 && !strcmp(argv[1],"--negative")){
  testKernel<<<2,32>>>(64);auto safe=cudaDeviceSynchronize();if(safe!=cudaSuccess)return 2;
  testKernel<<<2,32>>>(60);auto fail=cudaDeviceSynchronize();
  printf("ASSERT_STATUS %d\n",int(fail));fflush(stdout);return fail==cudaErrorAssert?0:3;
 }
 auto run=[](int k){ if(k==1)perfKernel<<<1000,256>>>();else perfKernel2<<<1000,256>>>();
 auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"CUDA status %d\n",int(e));exit(2);}};
 run(1);run(2);
 for(int t=1;t<=5;t++){double times[2];for(int k=1;k<=2;k++){
 auto s=std::chrono::steady_clock::now();run(k);times[k-1]=std::chrono::duration<double>(std::chrono::steady_clock::now()-s).count();}
 printf("PERF_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"component_time_seconds\":[%.17g,%.17g]}\n",t,times[0]+times[1],times[0],times[1]);fflush(stdout);}
}
