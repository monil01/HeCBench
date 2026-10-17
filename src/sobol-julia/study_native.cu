#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include "../sobol-cuda/sobol_primitives.cu"
#include "../sobol-cuda/sobol_gold.cu"
#include "../sobol-cuda/sobol_gpu.cu"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
struct Case{int n,d;float*out;};
static std::vector<unsigned> expected(Case c,const std::vector<unsigned>&dirs){std::vector<unsigned>ret;for(int dim=0;dim<c.d;++dim)for(int i=0;i<c.n;++i){unsigned gray=unsigned(i)^(unsigned(i)>>1),x=0;for(int bit=0;bit<32;++bit)if(gray&(1u<<bit))x^=dirs[dim*32+bit];float y=float(x)*2.3283064e-10f;unsigned bits;memcpy(&bits,&y,4);ret.push_back(bits);}return ret;}
int main(){std::vector<unsigned>dirs(3200);initSobolDirectionVectors(100,dirs.data());unsigned*dd;ck(cudaMalloc(&dd,3200*4));ck(cudaMemcpy(dd,dirs.data(),3200*4,cudaMemcpyHostToDevice));std::vector<Case>cases;for(auto shape:std::vector<std::pair<int,int>>{{1,7},{257,7},{16387,7},{1027,100}}){Case c{shape.first,shape.second,nullptr};ck(cudaMalloc(&c.out,c.n*c.d*4));cases.push_back(c);}for(auto c:cases)sobolGPU(1,c.n,c.d,dd,c.out);for(int trial=1;trial<=5;++trial){std::vector<unsigned>values=dirs;bool ok=true;double elapsed=0.;for(auto c:cases){elapsed+=sobolGPU(10,c.n,c.d,dd,c.out)*1e-9/10;ck(cudaGetLastError());std::vector<unsigned>got(c.n*c.d);ck(cudaMemcpy(got.data(),c.out,got.size()*4,cudaMemcpyDeviceToHost));ok&=got==expected(c,dirs);values.insert(values.end(),got.begin(),got.end());}std::string path="trial_"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(values.data(),4,values.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":222415,\"max_abs_error\":0,\"timing_region\":\"sum_four_sobol_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":222415,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",path.c_str());if(!ok)return 3;}for(auto c:cases)cudaFree(c.out);cudaFree(dd);return 0;}
