#define main original_hecbench_main
#include "../meanshift-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int repeats=1000,n=10000,padded=10048;bool tiled=argc>1&&std::string(argv[1])=="tiled";std::vector<float>host(padded*3);std::vector<double>expected(n*3);FILE*f=fopen("/home/imo/HeCBench_julia/src/meanshift-julia/study_input_f32.bin","rb");if(!f||fread(host.data(),4,host.size(),f)!=host.size())return 5;fclose(f);f=fopen("/home/imo/HeCBench_julia/src/meanshift-julia/study_expected_f64.bin","rb");if(!f||fread(expected.data(),8,expected.size(),f)!=expected.size())return 6;fclose(f);auto cpu_expected=expected;float*a,*b;cudaMalloc(&a,host.size()*4);cudaMalloc(&b,host.size()*4);cudaMemcpy(a,host.data(),host.size()*4,cudaMemcpyHostToDevice);cudaMemset(b,0,host.size()*4);
 auto launch=[&](){if(tiled)mean_shift::gpu::mean_shift_tiling<<<157,64>>>(a,b);else mean_shift::gpu::mean_shift<<<157,64>>>(a,b);cudaDeviceSynchronize();std::swap(a,b);};for(int warm=0;warm<1000;warm++)launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemcpy(a,host.data(),host.size()*4,cudaMemcpyHostToDevice);cudaMemset(b,0,host.size()*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 std::vector<float>out(n*3);cudaMemcpy(out.data(),a,n*3*4,cudaMemcpyDeviceToHost);std::vector<double>got(host.begin(),host.end());got.insert(got.end(),out.begin(),out.end());expected.assign(host.begin(),host.end());expected.insert(expected.end(),cpu_expected.begin(),cpu_expected.end());for(int i=0;i<n;i++)for(int j=0;j<3;j++)if(!std::isfinite(out[3*i+j])||fabs(out[3*i+j]-100*(i%3))>sqrt(10.))return 7;
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.01));ok&=std::isfinite((double)got[j])&&e<=0.01;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(double),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
