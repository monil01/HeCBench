#define main hecbench_original_main
#include "mrg32k3a-cuda/main.cu"
#undef main
std::vector<float> generate_mrg(int n,bool host){
 std::vector<float>result(n);curandGenerator_t gen=nullptr;
 if(host)run_on_host(n,0ULL,1234ULL,CURAND_ORDERING_PSEUDO_BEST,CURAND_RNG_PSEUDO_MRG32K3A,gen,result);
 else run_on_device(n,0ULL,1234ULL,CURAND_ORDERING_PSEUDO_BEST,CURAND_RNG_PSEUDO_MRG32K3A,gen,result);
 CURAND_CHECK(curandDestroyGenerator(gen));return result;
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):1;generate_mrg(n,true);generate_mrg(n,false);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  auto start=std::chrono::steady_clock::now();std::vector<float>host;for(int r=0;r<repeat;r++)host=generate_mrg(n,true);auto stop=std::chrono::steady_clock::now();double hosttime=std::chrono::duration<double>(stop-start).count()/repeat;
  cudaDeviceSynchronize();start=std::chrono::steady_clock::now();std::vector<float>result;for(int r=0;r<repeat;r++)result=generate_mrg(n,false);cudaDeviceSynchronize();stop=std::chrono::steady_clock::now();double time=std::chrono::duration<double>(stop-start).count()/repeat;
  float error=0;bool correct=true;for(int i=0;i<n;i++){float d=fabsf(result[i]-host[i]);error=fmaxf(error,d);correct &= std::isfinite(result[i])&&d<=1e-3f;}
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(result.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"timing_region\":\"vendor_rng_allocate_create_configure_generate_d2h_release_destroy\",\"jit_excluded\":true,\"internal_repeats\":%d,\"host_time_seconds\":%.12g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,correct?"true":"false",n,error,repeat,hosttime,filename,n);if(!correct)return 1;
 }return 0;
}
