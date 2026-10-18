#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <algorithm>
using std::string;
class OptionParser;
#include "../qtclustering-cuda/kernels_common.h"
#include "../qtclustering-cuda/kernels_compact_storage.h"
#ifndef FIXTURE
#error FIXTURE required
#endif
#ifndef STUDYDIR
#error STUDYDIR required
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double> load(const std::string&name){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f)exit(3);auto n=f.tellg();f.seekg(0);std::vector<double>a((size_t)n/8);f.read((char*)a.data(),n);return a;}
template<class T>struct Buffer{T*base;T*p;size_t n;T guard;std::vector<T> initial;Buffer(std::vector<T>h):n(h.size()),initial(h){guard=std::is_same<T,int8_t>::value?(T)-99:(T)-12345;CK(cudaMalloc(&base,(n+16)*sizeof(T)));p=base+8;reset();}void reset(){std::vector<T>h(n+16,guard);std::copy(initial.begin(),initial.end(),h.begin()+8);CK(cudaMemcpy(base,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>a(n);CK(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}bool guards(std::vector<double>&out){T a[8],b[8];CK(cudaMemcpy(a,base,8*sizeof(T),cudaMemcpyDeviceToHost));CK(cudaMemcpy(b,p+n,8*sizeof(T),cudaMemcpyDeviceToHost));bool ok=true;for(T x:a){ok&=x==guard;out.push_back(x);}for(T x:b){ok&=x==guard;out.push_back(x);}return ok;}};
template<class T>void append(std::vector<double>&a,const std::vector<T>&b){for(T v:b)a.push_back(v);}
struct Case{int N,D,ci;float threshold;Buffer<float>dist,cache;Buffer<int>idx,degree,ungrp,card,result;Buffer<int8_t>ai,clustered;std::vector<double>gold;
Case(int n,int d,int c,float th,std::vector<int>ix,std::vector<float>ds):N(n),D(d),ci(c),threshold(th),dist(ds),cache(std::vector<float>(n*d,0)),idx(ix),degree(std::vector<int>(n,-12345)),ungrp(std::vector<int>(n)),card(std::vector<int>(2*n,-12345)),result(std::vector<int>(n,-12345)),ai(std::vector<int8_t>(n*n,0)),clustered(std::vector<int8_t>(n,0)){for(int i=0;i<n;++i)ungrp.initial[i]=i;gold=load(std::string(STUDYDIR)+"/oracle_case"+std::to_string(ci)+".bin");}
void reset(){dist.reset();cache.reset();idx.reset();degree.reset();ungrp.reset();card.reset();result.reset();ai.reset();clustered.reset();}
};
struct Outcome{bool ok;std::vector<double>art,times;std::vector<std::string>labels;int iterations,remaining;};
Outcome execute(Case&c){c.reset();CK(cudaDeviceSynchronize());Outcome o;o.ok=true;int pc=c.N,it=0;std::vector<double>logical={(double)c.N,(double)c.D,0},aux;auto stage=[&](const char*name,auto launch){auto t=std::chrono::steady_clock::now();launch();CK(cudaDeviceSynchronize());o.times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());o.labels.push_back("case"+std::to_string(c.ci)+":"+name+":"+std::to_string(it));};
 stage("degree",[&](){compute_degrees<<<c.N,64>>>(c.idx.p,c.degree.p,c.N,c.D);});append(logical,c.degree.get());
 while(pc>0&&it<=c.N){++it;int blocks=pc;
  stage("qtc",[&](){QTC_device<<<blocks,64>>>(c.dist.p,(char*)c.ai.p,(char*)c.clustered.p,c.idx.p,c.card.p,c.ungrp.p,c.cache.p,c.degree.p,pc,c.N,c.D,c.threshold,0,1,blocks);});
  if(blocks>1)stage("reduce",[&](){reduce_card_device<<<1,1>>>(c.card.p,blocks);});
  auto card=c.card.get();int count=card[0],seed=card[1];if(!(count>=1&&count<=pc&&seed>=0&&seed<c.N)){o.ok=false;break;}
  stage("trim",[&](){trim_ungrouped_pnts_indr_array<<<1,64>>>(seed,c.ungrp.p,c.dist.p,c.result.p,(char*)c.ai.p,(char*)c.clustered.p,c.idx.p,c.card.p,c.cache.p,c.degree.p,pc,c.N,c.D,c.threshold);});
  stage("update",[&](){update_clustered_pnts_mask<<<1,64>>>((char*)c.clustered.p,(char*)c.ai.p,c.N);});
  int rem=pc-count;for(int v:{pc,count,seed,rem})logical.push_back(v);auto rs=c.result.get(),un=c.ungrp.get();auto cl=c.clustered.get();for(int i=0;i<count;++i)logical.push_back(rs[i]);for(int i=0;i<rem;++i)logical.push_back(un[i]);append(logical,cl);
  append(aux,c.degree.get());append(aux,un);append(aux,c.ai.get());append(aux,c.cache.get());append(aux,cl);append(aux,c.card.get());append(aux,rs);pc=rem;if(count<=1)break;
 }
 logical[2]=it;o.ok &=logical==c.gold;o.iterations=it;o.remaining=pc;o.art.push_back(logical.size());append(o.art,logical);o.art.push_back(aux.size());append(o.art,aux);
 auto ix=c.idx.get();auto ds=c.dist.get();o.ok&=ix==c.idx.initial&&ds==c.dist.initial;append(o.art,ix);append(o.art,ds);
 o.ok&=c.dist.guards(o.art);o.ok&=c.idx.guards(o.art);o.ok&=c.degree.guards(o.art);o.ok&=c.ungrp.guards(o.art);o.ok&=c.ai.guards(o.art);o.ok&=c.cache.guards(o.art);o.ok&=c.clustered.guards(o.art);o.ok&=c.card.guards(o.art);o.ok&=c.result.guards(o.art);return o;
}
int main(){auto pre=load(FIXTURE);size_t pos=1;std::vector<Case*>cases;int nc=(int)pre[0];for(int ci=1;ci<=nc;++ci){int N=pre[pos++],D=pre[pos++];float th=pre[pos++];pos+=2*N;std::vector<int>ix(N*D);std::vector<float>ds(N*D);for(int&v:ix)v=pre[pos++];for(float&v:ds)v=(float)pre[pos++];cases.push_back(new Case(N,D,ci,th,ix,ds));}if(pos!=pre.size())return 3;
 for(auto c:cases)execute(*c);
 for(int trial=1;trial<=5;++trial){auto art=pre;std::vector<double>times;std::vector<std::string>labels;std::vector<int>its,rems;bool ok=true;double total=0;for(auto c:cases){auto o=execute(*c);ok&=o.ok;append(art,o.art);append(times,o.times);labels.insert(labels.end(),o.labels.begin(),o.labels.end());its.push_back(o.iterations);rems.push_back(o.remaining);}for(double t:times)total+=t;
  std::string file="output_trial"+std::to_string(trial)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)art.data(),art.size()*8);f.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_complete_QT_degree_candidate_reduce_trim_update_launch_completion_pipeline\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,total,ok?"true":"false",art.size(),file.c_str(),art.size());for(size_t i=0;i<times.size();++i)printf("%s%.17g",i?",":"",times[i]);printf("],\"path_labels\":[");for(size_t i=0;i<labels.size();++i)printf("%s\"%s\"",i?",":"",labels[i].c_str());printf("],\"convergence_iterations\":[");for(size_t i=0;i<its.size();++i)printf("%s%d",i?",":"",its[i]);printf("],\"source_stop_remaining\":[");for(size_t i=0;i<rems.size();++i)printf("%s%d",i?",":"",rems[i]);printf("]}\n");
 }
 return 0;
}
