#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#define main archived_pointwise_main
#include "../pointwise-cuda/main.cu"
#undef main
#include "study_cases.h"
const int G=16;const float GUARD=12345.5f;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>std::vector<T>readv(const char*n){std::ifstream f(std::string("/home/imo/HeCBench_julia/src/pointwise-julia/")+n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixture read");size_t size=f.tellg();f.seekg(0);std::vector<T>v(size/sizeof(T));f.read((char*)v.data(),size);return v;}
struct State{StudyCase c;std::vector<std::vector<float>>host;std::vector<float*>dev;};
double sequence(State&a){auto c=a.c;int hiddenSize=c.h,miniBatch=c.b,seqLength=c.s,numLayers=c.l,numElements=c.h*c.b;float*h_data=a.dev[0]+G,*i_data=a.dev[1]+G,*c_data=a.dev[2]+G,*bias=a.dev[3]+G,*tmp_h=a.dev[4]+G,*tmp_i=a.dev[5]+G,*linearGates=a.dev[6]+G;dim3 grids_p((numElements+255)/256),blocks(256);int lStart=0,lEnd=0,rStart=0,rEnd=0,recurBatchSize=2;double ktime=0;
  while (true) {
    // Many layer "scheduling".
    if (lEnd == 0) {
      lStart = 0;
      lEnd = 1;
      rStart = 0;
    }
    else {
      // Move "up" and "left"
      lStart++;
      lEnd++;

      rStart -= recurBatchSize;

      // Over the top or off the left, reset to layer 0
      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;

        lStart = 0;
        lEnd = 1;
      }

      // Off the right, step up
      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++;
        lEnd++;
        rStart -= recurBatchSize;
      }

      // Over the top or off the left, done!
      if (lEnd > numLayers || rStart < 0) {
        break;
      }
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;

    auto start = std::chrono::steady_clock::now();

    for (int layer = lStart; layer < lEnd; layer++) {
      for (int i = rStart; i < rEnd; i++)
        elementwise <<< grids_p, blocks >>>
        (hiddenSize, miniBatch,
         tmp_h + 4 * layer * numElements,
         tmp_i + 4 * i * numElements,
         bias + 8 * layer * hiddenSize,
         linearGates + 4 * (i * numElements + layer * seqLength * numElements),
         h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
         i_data + i * numElements + (layer + 1) * seqLength * numElements,
         c_data + i * numElements + layer * (seqLength + 1) * numElements,
         c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements);
      gpuErrCheck(cudaGetLastError());
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }


check(cudaGetLastError());check(cudaDeviceSynchronize());return ktime*1e-9;}
void check_source_init(const std::vector<float>&expected){int n=expected.size();std::vector<float>v(n+2*G,GUARD);float*p;check(cudaMalloc(&p,v.size()*4));check(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));init<<<(n+255)/256,256>>>(p+G,n);check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<n+2*G;i++)if(v[i]!=(i>=G&&i<G+n?expected[i-G]:GUARD))throw std::runtime_error("source LCG mismatch");check(cudaFree(p));}
int main(){auto inputs=readv<float>("inputs_f32.bin");auto gold=readv<double>("oracle_f64.bin");std::vector<State>states;std::vector<double>prefix;long long source_init_checked=0;for(auto c:cases){State a;a.c=c;for(int k=0;k<7;k++){std::vector<float>v(c.sizes[k]+2*G,GUARD);int f=-1;if(k==2)f=0;if(k>=3&&k<=5)f=k-2;if(f>=0){std::copy(inputs.begin()+c.offs[f],inputs.begin()+c.offs[f]+c.sizes[k],v.begin()+G);if(c.mode==0){std::vector<float>payload(v.begin()+G,v.end()-G);check_source_init(payload);source_init_checked+=payload.size();}}float*p;check(cudaMalloc(&p,v.size()*4));check(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));a.host.push_back(v);a.dev.push_back(p);prefix.insert(prefix.end(),v.begin(),v.end());}states.push_back(a);}for(auto&a:states)sequence(a);for(int trial=1;trial<=5;trial++){bool ok=true;double maxerr=0,sum=0;std::vector<double>out,times;for(auto&a:states){auto c=a.c;for(int k=0;k<7;k++)check(cudaMemcpy(a.dev[k],a.host[k].data(),a.host[k].size()*4,cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());double dt=sequence(a);times.push_back(dt);sum+=dt;long long off=c.go;for(int k=0;k<7;k++){std::vector<float>v(a.host[k].size());check(cudaMemcpy(v.data(),a.dev[k],v.size()*4,cudaMemcpyDeviceToHost));for(size_t i=0;i<v.size();i++){double ref=gold[off+i],err=fabs(v[i]-ref);ok&=std::isfinite(v[i])&&err<=2e-5+2e-6*fabs(ref);maxerr=fmax(maxerr,err);if(i<G||i>=v.size()-G)ok&=v[i]==GUARD;if(k>=3&&k<=5)ok&=v[i]==a.host[k][i];}off+=v.size();out.insert(out.end(),v.begin(),v.end());}}std::string path="/tmp/pointwise_"+std::to_string(getpid())+"_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.write((char*)out.data(),out.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"native_source_init_checked_values\":%lld,\"jit_excluded\":true,\"timing_region\":\"sum_five_source_wavefront_group_launch_windows_plus_device_completion\",\"component_time_seconds\":[",trial,sum,ok?"true":"false",maxerr,source_init_checked);for(int i=0;i<(int)times.size();i++)printf("%s%.12g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",path.c_str(),prefix.size()+out.size(),prefix.size()+out.size());}}
