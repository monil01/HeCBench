#pragma once
// Independently authored source-behavior oracle. Known terminal-run and shuffle
// defects are intentionally represented and separately witnessed, not endorsed.
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
using Stats=std::array<double,18>;
inline std::array<double,3> direction_stat(const std::vector<uint8_t>&a){
 uint64_t transitions=0,longest_finished=0,current=1,increases=0;bool prev=a[0]<=a[1];
 for(size_t j=1;j+1<a.size();++j){increases+=prev;bool next=a[j]<=a[j+1];if(prev==next)++current;else{++transitions;longest_finished=std::max(longest_finished,current);current=1;}prev=next;}
 increases+=prev;return {double(1+transitions),double(longest_finished),double(std::max(increases,uint64_t(a.size())-increases))};
}
inline std::array<double,2> median_stat(const std::vector<uint8_t>&a,double med){
 uint64_t runs=1,finished=0,current=1;bool prev=a[0]>=med;
 for(size_t j=1;j<a.size();++j){bool next=a[j]>=med;if(prev==next)++current;else{++runs;finished=std::max(finished,current);current=1;}prev=next;}return{double(runs),double(finished)};
}
inline Stats independent_stats(const std::vector<uint8_t>&a,uint32_t size,double mean,double med,bool gpu_fma){
 Stats s{};double cumulative=0;
 for(size_t i=0;i<a.size();++i){cumulative+=a[i];double e=gpu_fma?std::fma(-double(i+1),mean,cumulative):cumulative-double(i+1)*mean;s[0]=std::max(s[0],std::abs(e));}
 std::vector<uint8_t>packed,hamming;const auto*dir=&a;const auto*coll=&a;
 if(size==1){if(a.size()%8)throw std::runtime_error("binary source only certified divisible8");packed.resize(a.size()/8);hamming.resize(a.size()/8);for(size_t i=0;i<packed.size();++i)for(int j=0;j<8;++j){packed[i]|=(a[8*i+j]&1)<<(7-j);hamming[i]+=a[8*i+j];}dir=&hamming;coll=&packed;}
 auto d=direction_stat(*dir);s[1]=d[0];s[2]=d[1];s[3]=d[2];auto m=median_stat(a,med);s[4]=m[0];s[5]=m[1];
 size_t start=0,cnt=0,maxdist=0;uint64_t distances=0;while(start<coll->size()){std::array<bool,256>seen{};size_t k=start;for(;k<coll->size();++k){uint8_t x=(*coll)[k];if(seen[x]){size_t distance=k-start;distances+=distance;maxdist=std::max(maxdist,distance);++cnt;start=k+1;break;}seen[x]=true;}if(k==coll->size())break;}
 if(!cnt)throw std::runtime_error("control has no collisions; source would be nonfinite");s[6]=double(distances)/cnt;s[7]=maxdist;
 const int lags[]={1,2,8,16,32};for(int k=0;k<5;++k){uint64_t equal=0,product=0;for(size_t i=0;i+lags[k]<dir->size();++i){equal+=(*dir)[i]==(*dir)[i+lags[k]];product+=uint32_t((*dir)[i])*(*dir)[i+lags[k]];}s[8+k]=equal;s[13+k]=product;}return s;
}
inline std::vector<uint8_t> lossless_pack(const std::vector<uint8_t>&a,bool binary){
 if(!binary)return a;std::vector<uint8_t>out((a.size()+7)/8,0);for(size_t i=0;i<a.size();++i){if(a[i]>1)throw std::runtime_error("nonbinary value not packable");out[i/8]|=a[i]<<(i%8);}return out;
}
struct Oracle{std::vector<uint8_t> packed;std::vector<Stats> stats;std::array<uint32_t,54>counts{};bool binary;};
inline Oracle make_oracle(const std::vector<uint8_t>&input,uint32_t size,uint32_t N,double mean,double median,const double*baseline){
 Oracle o;o.binary=std::all_of(input.begin(),input.end(),[](uint8_t x){return x<=1;});o.stats.resize(N);std::vector<uint8_t>raw(input.size()*N);uint64_t sum=0;for(auto x:input)sum+=x;
 std::atomic<uint32_t>cursor{0};std::vector<std::thread>workers;unsigned nt=std::min(32u,std::max(1u,std::thread::hardware_concurrency()));
 for(unsigned w=0;w<nt;++w)workers.emplace_back([&]{std::vector<uint8_t>x;for(;;){uint32_t tid=cursor.fetch_add(1);if(tid>=N)break;x=input;uint64_t state=sum^uint64_t(tid);for(size_t j=x.size()-1;j>0;--j){state=(state*UINT64_C(2806196910506780709)+1)&UINT64_C(0x7fffffffffffffff);size_t k=uint32_t(state)%j;std::swap(x[j],x[k]);}o.stats[tid]=independent_stats(x,size,mean,median,true);for(size_t i=0;i<x.size();++i)raw[i*N+tid]=x[i];}});
 for(auto&w:workers)w.join();o.packed=lossless_pack(raw,o.binary);
 for(auto&s:o.stats)for(int k=0;k<18;++k){double x=k? s[k]:double(float(s[k]));double y=k?baseline[k]:double(float(baseline[k]));int rank=x>y?0:x==y?1:2;++o.counts[k*3+rank];}return o;
}
