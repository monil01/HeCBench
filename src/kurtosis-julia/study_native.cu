#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "../kurtosis-cuda/StoreTypedefs.h"
#include "../kurtosis-cuda/StoreElement.h"
#include "../kurtosis-cuda/kurtosis.h"
int main(){static_assert(sizeof(storeElement)==24&&sizeof(kurtosisResult)==20);int sizes[]={1,257,4099,4099,513,513,1024};const int G=16,R=10;size_t total=0;for(int n:sizes)total+=n;std::vector<storeElement>fixture(total);FILE*f=fopen("fixtures_aos24.bin","rb");if(!f||fread(fixture.data(),24,total,f)!=total)return 2;fclose(f);std::vector<double>oracle(35);f=fopen("oracle_f64.bin","rb");if(!f||fread(oracle.data(),8,35,f)!=35)return 3;fclose(f);bool all=true;
 for(int trial=0;trial<=5;trial++){bool ok=true;double maxerr=0;size_t offset=0;int ci=0;std::vector<double>prefix,outputs,times;
 for(int n:sizes){std::vector<storeElement>h(n+2*G),actual(n+2*G);std::memset(h.data(),0,h.size()*24);for(auto &x:h){x.tag=-777;x.metric=19;x.time=65261;x.value=-1234.5f;}std::memcpy(h.data()+G,fixture.data()+offset,n*24);storeElement*dev;cudaMalloc(&dev,h.size()*24);cudaMemcpy(dev,h.data(),h.size()*24,cudaMemcpyHostToDevice);void*ptr=nullptr;if(trial==0){kurtosis(dev+G,n,1,&ptr);delete(kurtosisResult*)ptr;ok &=cudaDeviceSynchronize()==cudaSuccess;cudaFree(dev);offset+=n;ci++;continue;}cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();size_t bytes=kurtosis(dev+G,n,R,&ptr);ok &=cudaDeviceSynchronize()==cudaSuccess;times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R);auto a=*(kurtosisResult*)ptr;double out[]={double(a.count),a.mean,a.m2,a.m3,a.m4};ok &=bytes==20&&a.count==n;double mean=oracle[5*ci+1],maxx=0,sums[5]={double(n),0,0,0,0};for(int i=0;i<n;i++){double v=fixture[offset+i].value;maxx=fmax(maxx,fabs(v));double e=fabs(v-mean);for(int p=1;p<=4;p++)sums[p]+=pow(e,p);}double mb=1e-5+4e-6*fmax(1.,maxx);for(int j=1;j<5;j++){double ref=oracle[5*ci+j],budget=j==1?mb:3e-5*(1+sums[j])+j*mb*sums[j-1],err=fabs(out[j]-ref);maxerr=fmax(maxerr,err);ok &=std::isfinite(out[j])&&err<=budget;}if(ci==0||ci==4||ci==5)ok &=a.m2==0&&a.m3==0&&a.m4==0;cudaMemcpy(actual.data(),dev,actual.size()*24,cudaMemcpyDeviceToHost);ok &=std::memcmp(actual.data(),h.data(),h.size()*24)==0;std::vector<unsigned>bits(actual.size()*6);std::memcpy(bits.data(),actual.data(),bits.size()*4);for(unsigned v:bits)prefix.push_back(v);outputs.push_back(bytes);for(double v:out)outputs.push_back(v);delete(kurtosisResult*)ptr;cudaFree(dev);offset+=n;ci++;
 }if(!trial){if(!ok)return 4;continue;}all &=ok;std::vector<double>artifact=prefix;artifact.insert(artifact.end(),outputs.begin(),outputs.end());char path[80];snprintf(path,80,"output_trial%d.bin",trial);f=fopen(path,"wb");if(!f)return 5;fwrite(artifact.data(),8,artifact.size(),f);fclose(f);double sum=0;for(double t:times)sum+=t;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum7_full_reduce_API_hostwindows_R10\",\"jit_excluded\":true,\"path_times_seconds\":[",trial,sum,ok?"true":"false",artifact.size(),maxerr);for(size_t j=0;j<times.size();j++)printf("%s%.12g",j?",":"",times[j]);printf("],\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path,artifact.size());
 }puts(all?"PASS":"FAIL");return all?0:1;}
