#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>
#include "reference_history/common.h"
#include "reference_history/aig_manager.h"
#include "independent_truth_gold.h"
using Clock=std::chrono::steady_clock;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Graph{int np,no,nn;std::vector<int> a,b,o,refs;};
static Graph capture(AIGMan &m){
    Graph g;int objects;
    if(!m.deviceAllocated)throw std::runtime_error("observer requires explicitly device-staged source state");
    ck(cudaMemcpy(&objects,m.d_pnObjs,4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(&g.np,m.d_pnPIs,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&g.no,m.d_pnPOs,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&g.nn,m.d_pnNodes,4,cudaMemcpyDeviceToHost));
    if(objects!=g.nn+g.np+1)throw std::runtime_error("node count");
    g.a.resize(objects);g.b.resize(objects);g.refs.resize(objects);g.o.resize(g.no);
    ck(cudaMemcpy(g.a.data(),m.d_pFanin0,objects*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(g.b.data(),m.d_pFanin1,objects*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(g.refs.data(),m.d_pNumFanouts,objects*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(g.o.data(),m.d_pOuts,g.no*4,cudaMemcpyDeviceToHost));ck(cudaDeviceSynchronize());return g;
}
static std::vector<uint64_t> truth(const Graph &g){
    const int words=((1<<g.np)+63)/64;std::vector<uint64_t> all(g.a.size()*words),result;
    // Internal source convention is node0=TRUE; AIGER complement convention differs only at0/1.
    uint64_t mask=g.np<6?(uint64_t(1)<<(1<<g.np))-1:~uint64_t(0);
    for(int w=0;w<words;w++)all[w]=mask;
    for(int j=0;j<g.np;j++)for(int k=0;k<(1<<g.np);k++)if((k>>j)&1)all[(j+1)*words+k/64]|=uint64_t(1)<<(k%64);
    auto lit=[&](int x,int w){if(x<0||size_t(x/2)>=g.a.size())throw std::runtime_error("literal bounds");return all[(x/2)*words+w] ^ (x&1?mask:0);};
    for(size_t j=g.np+1;j<g.a.size();j++){if(g.a[j]<0||g.b[j]<0||size_t(g.a[j]/2)>=j||size_t(g.b[j]/2)>=j)throw std::runtime_error("topological order");for(int w=0;w<words;w++)all[j*words+w]=lit(g.a[j],w)&lit(g.b[j],w);}
    for(int o:g.o)for(int w=0;w<words;w++)result.push_back(lit(o,w));return result;
}
static void ints(FILE *f,const std::vector<int>&v){fwrite(v.data(),4,v.size(),f);}
static void serialize(FILE *f,const Graph &g,const std::vector<uint64_t>&t){int head[]={g.np,g.no,g.nn};fwrite(head,4,3,f);ints(f,g.a);ints(f,g.b);ints(f,g.o);ints(f,g.refs);fwrite(t.data(),8,t.size(),f);}
static const uint64_t *gold(int n){switch(n){case 4:return gold_pi4;case 6:return gold_pi6;case 9:return gold_pi9;default:return gold_pi12;}}
static void stage(AIGMan&m,int k){switch(k){case 0:case 4:m.balance(1);break;case 1:case 5:m.rewrite(false,true);break;case 2:m.refactor(true,false,3);break;case 3:case 10:case 14:m.strash(false,true);break;case 6:case 7:case 11:case 12:m.rewrite(true,true);break;case 8:case 13:m.balance(0);break;case 9:m.refactor(true,true,3);break;default:throw std::runtime_error("stage");}}
int main(int argc,char **argv){try{
    if(argc!=2)throw std::runtime_error("usage native_study benchmark-parent");
    ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize,size_t(2048)*1024*1024));
    for(int trial=0;trial<=5;trial++){
        FILE *f=nullptr;std::string name="output_trial"+std::to_string(trial)+".bin";if(trial)f=fopen(name.c_str(),"wb");
        double elapsed=0;size_t checked=0;bool valid=true;
        // All initial device datasets form one exact prefix before any computed output.
        std::vector<Graph> initial;for(int n:{4,6,9,12}){AIGMan m(0);m.readFile((std::string(argv[1])+"/controls/control_pi"+std::to_string(n)+".aig").c_str());m.toDevice();initial.push_back(capture(m));}
        for(const auto &g:initial){auto t=truth(g);if(f)serialize(f,g,t);}
        for(int n:{4,6,9,12}){
            AIGMan m(0);m.readFile((std::string(argv[1])+"/controls/control_pi"+std::to_string(n)+".aig").c_str());
            for(int k=0;k<15;k++){
                // Explicit stage-setup adaptation: source toDevice/preparation outside ROI,
                // same in every model. Source host algorithm bodies remain unchanged.
                if(!m.deviceAllocated)m.toDevice();ck(cudaDeviceSynchronize());
                auto input=capture(m);auto inputTruth=truth(input);valid &= memcmp(inputTruth.data(),gold(n),inputTruth.size()*8)==0;
                auto start=Clock::now();stage(m,k);ck(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(Clock::now()-start).count();
                if(!m.deviceAllocated)m.toDevice();auto output=capture(m);auto t=truth(output);valid &= memcmp(t.data(),gold(n),t.size()*8)==0;checked+=size_t(1<<n)*t.size()/(((1<<n)+63)/64);
                if(f){serialize(f,input,inputTruth);serialize(f,output,t);}
            }
        }
        if(f){long bytes=ftell(f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"host_inclusive_60_source_stages_excluding_explicit_device_staging_and_observation\",\"jit_excluded\":true,\"internal_repeats\":1,\"full_boolean_oracle_pass\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%ld,\"byte_order\":\"little\"}}\n",trial,elapsed,valid?"true":"false",checked,valid?"true":"false",name.c_str(),bytes/4);fflush(stdout);if(!valid)return 2;}
    }return 0;
}catch(std::exception &e){fprintf(stderr,"study error %s\n",e.what());return 1;}}
