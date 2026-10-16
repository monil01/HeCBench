#define main original_hecbench_main
#include "../affine-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int repeats=argc>1?atoi(argv[1]):10;if(repeats<1)return 2;const size_t n=(size_t)X_SIZE*Y_SIZE;
 std::vector<unsigned short>host(n),expected(n),got(n);for(int y=0;y<Y_SIZE;y++)for(int x=0;x<X_SIZE;x++)host[y*X_SIZE+x]=3*x+5*y;
 affine_reference(host.data(),expected.data());unsigned short*a,*b;if(cudaMalloc(&a,n*sizeof(*a))!=cudaSuccess||cudaMalloc(&b,n*sizeof(*b))!=cudaSuccess)return 3;cudaMemcpy(a,host.data(),n*sizeof(*a),cudaMemcpyHostToDevice);
 auto launch=[&](){affine<<<dim3(32,32),dim3(16,16)>>>(a,b);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(b,0,n*sizeof(*b));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),b,n*sizeof(*b),cudaMemcpyDeviceToHost);long long maxabs=0;bool ok=status==cudaSuccess;std::vector<int>flat(n);for(size_t i=0;i<n;i++){long long e=llabs((long long)got[i]-expected[i]);if(e>maxabs)maxabs=e;ok &=e<=1;flat[i]=got[i];}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(flat.data(),sizeof(int),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",n,maxabs,repeats,filename,n);
 }cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
