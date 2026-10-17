#define main original_surfel_main
#include "../surfel-cuda/main.cu"
#undef main
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include "study_fixture.h"
template<size_t N>std::vector<float>decode(const uint32_t(&a)[N]){std::vector<float>v(N);memcpy(v.data(),a,4*N);return v;}
int main(){auto src=decode(STUDY_SRC_BITS),focals=decode(STUDY_FOCALS_BITS),expected=decode(STUDY_EXPECTED_BITS);float*s,*d;cudaMalloc(&s,4*src.size());cudaMalloc(&d,4*1024);cudaMemcpy(s,src.data(),4*src.size(),cudaMemcpyHostToDevice);for(float f:focals){surfel_render<float><<<dim3(2,2),dim3(16,16)>>>(s,257,f,32,32,d);surfel_render_tile<float,256><<<dim3(2,2),dim3(16,16)>>>(s,257,f,32,32,d);}cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){std::vector<float>got,one(1024),cpu(1024);double total=0,error=0;bool ok=true;for(int fi=0;fi<3;fi++){float f=focals[fi];reference<float>(src.data(),257,f,32,32,cpu.data());for(bool tiled:{false,true}){cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();for(int repeat=0;repeat<10;repeat++){if(tiled)surfel_render_tile<float,256><<<dim3(2,2),dim3(16,16)>>>(s,257,f,32,32,d);else surfel_render<float><<<dim3(2,2),dim3(16,16)>>>(s,257,f,32,32,d);}cudaDeviceSynchronize();total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/10;cudaMemcpy(one.data(),d,4096,cudaMemcpyDeviceToHost);for(int i=0;i<1024;i++){error=std::max(error,std::abs(double(one[i])-expected[fi*1024+i]));ok&=std::isfinite(one[i])&&std::abs(double(one[i])-cpu[i])<1e-3;}got.insert(got.end(),one.begin(),one.end());}}
 ok&=error<1e-3;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/trial_"+std::to_string(trial)+".bin";FILE*file=fopen(path.c_str(),"wb");fwrite(got.data(),4,got.size(),file);fclose(file);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":6144,\"max_abs_error\":%.17g,\"timing_region\":\"sum_three_focals_base_tiled_sync_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":6144,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",error,path.c_str());if(!ok)return 8;}}
