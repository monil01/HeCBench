#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#define main historical_main
#include "../blockScan-cuda/main.cu"
#undef main
constexpr int GUARD=0x13572468, STRIDE=1041, PATHS=54, REPEATS=3;
template<cub::BlockScanAlgorithm ALG> void dispatch(int shape,const int*a,int*o){switch(shape){
case 0:BlockPrefixSumKernel<1024,1,ALG><<<1,1024>>>(a,o);break;
case 1:BlockPrefixSumKernel<512,2,ALG><<<1,512>>>(a,o);break;
case 2:BlockPrefixSumKernel<256,4,ALG><<<1,256>>>(a,o);break;
case 3:BlockPrefixSumKernel<128,8,ALG><<<1,128>>>(a,o);break;
case 4:BlockPrefixSumKernel<64,16,ALG><<<1,64>>>(a,o);break;
case 5:BlockPrefixSumKernel<32,32,ALG><<<1,32>>>(a,o);break;}}
int main(){std::vector<int> input(3072);FILE*f=fopen("/home/imo/HeCBench_julia/src/blockScan-julia/fixture.bin","rb");if(!f||fread(input.data(),4,input.size(),f)!=input.size())return 4;fclose(f);
std::vector<int>expected(STRIDE*PATHS,GUARD);for(int fixture=0;fixture<3;fixture++)for(int alg=0;alg<3;alg++)for(int shape=0;shape<6;shape++){int path=(fixture*3+alg)*6+shape;int64_t acc=0;for(int i=0;i<1024;i++){expected[path*STRIDE+8+i]=int(acc);acc+=input[fixture*1024+i];if(acc<INT32_MIN||acc>INT32_MAX)return 5;}expected[path*STRIDE+8+1024]=int(acc);}
int *di,*dout;GPU_CHECK(cudaMalloc(&di,input.size()*4));GPU_CHECK(cudaMalloc(&dout,expected.size()*4));GPU_CHECK(cudaMemcpy(di,input.data(),input.size()*4,cudaMemcpyHostToDevice));std::vector<int>copied(input.size());GPU_CHECK(cudaMemcpy(copied.data(),di,input.size()*4,cudaMemcpyDeviceToHost));if(copied!=input)return 6;
auto launch=[&](){for(int fixture=0;fixture<3;fixture++)for(int alg=0;alg<3;alg++)for(int shape=0;shape<6;shape++){int path=(fixture*3+alg)*6+shape;const int*a=di+fixture*1024;int*o=dout+path*STRIDE+8;if(alg==0)dispatch<cub::BLOCK_SCAN_RAKING>(shape,a,o);else if(alg==1)dispatch<cub::BLOCK_SCAN_RAKING_MEMOIZE>(shape,a,o);else dispatch<cub::BLOCK_SCAN_WARP_SCANS>(shape,a,o);}};
std::vector<int>got(expected.size(),GUARD);GPU_CHECK(cudaMemcpy(dout,got.data(),got.size()*4,cudaMemcpyHostToDevice));launch();GPU_CHECK(cudaDeviceSynchronize());
for(int trial=1;trial<=5;trial++){std::fill(got.begin(),got.end(),GUARD);GPU_CHECK(cudaMemcpy(dout,got.data(),got.size()*4,cudaMemcpyHostToDevice));GPU_CHECK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();for(int i=0;i<REPEATS;i++)launch();GPU_CHECK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()/REPEATS;GPU_CHECK(cudaMemcpy(got.data(),dout,got.size()*4,cudaMemcpyDeviceToHost));bool ok=got==expected;int64_t maxerr=0;for(size_t i=0;i<got.size();i++)maxerr=std::max(maxerr,std::abs(int64_t(got[i])-expected[i]));std::vector<int>art=input;art.insert(art.end(),got.begin(),got.end());const char*dir=getenv("STUDY_OUTPUT_DIR");std::string file=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";FILE*out=fopen(file.c_str(),"wb");fwrite(art.data(),4,art.size(),out);fclose(out);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":56214,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"complete54path_exclusive_scan_suite_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":59286,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",(long long)maxerr,file.c_str());}cudaFree(di);cudaFree(dout);}
