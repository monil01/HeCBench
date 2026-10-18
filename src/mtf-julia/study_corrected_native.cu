// SECONDARY corrected online-MTF algorithm; original source is byte-preserved.
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/copy.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include "study_fixtures.h"
struct Case{int n,id;std::vector<unsigned char>word,output;thrust::host_vector<unsigned char>alpha;thrust::device_vector<unsigned char>d;std::vector<unsigned char>ref,refalpha;Case(const std::vector<unsigned char>&w,int i):n(w.size()),id(i),word(n+64,0xa5),output(n+64,0xa5),alpha(320,0xa5),d(320,0xa5){std::copy(w.begin(),w.end(),word.begin()+32);std::vector<int>last(256,-1);for(int k=0;k<n;k++){int b=w[k],rank=last[b]<0?b:0;for(int a=0;a<256;a++){if(last[b]<0){if(a>b&&last[a]>=0)rank++;}else if(last[a]>last[b])rank++;}ref.push_back(rank);last[b]=k;}std::vector<int>order(256);for(int a=0;a<256;a++)order[a]=a;std::sort(order.begin(),order.end(),[&](int a,int b){return last[a]==last[b]?a<b:last[a]>last[b];});for(auto a:order)refalpha.push_back(a);}};
bool guards(Case&c){bool ok=true;auto check=[&](const auto&raw){for(size_t i=0;i<32;i++){ok&=raw[i]==0xa5;ok&=raw[raw.size()-1-i]==0xa5;}};check(c.word);check(c.output);check(c.alpha);thrust::host_vector<unsigned char>copy=c.d;check(copy);return ok;}
int main(){std::vector<Case*>cases;std::vector<unsigned int>prefix;int id=0;for(auto w:fixtures()){auto*c=new Case(w,id);for(auto n:{id,c->n,1})prefix.push_back(n);for(auto b:w)prefix.push_back(b);cases.push_back(c);id++;}
auto run=[&](Case&c,std::vector<unsigned int>&out){cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<256;j++)c.alpha[j+32]=j;auto h=c.alpha.begin()+32;auto dev=c.d.begin()+32;for(int i=0;i<c.n;i++){thrust::copy(h,h+256,dev);auto found=thrust::find(dev,dev+256,c.word[i+32]);int rank=found-dev;if(rank>=256)throw std::runtime_error("AlphabetFind");c.output[i+32]=rank;if(rank>0){thrust::copy(dev,found,h+1);c.alpha[32]=c.word[i+32];}}cudaDeviceSynchronize();double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();bool ok=cudaGetLastError()==cudaSuccess;std::vector<unsigned char>a(256);for(int j=0;j<256;j++)a[j]=j;for(int i=0;i<c.n;i++){unsigned char rank=c.output[i+32],symbol=a[rank];ok&=rank==c.ref[i]&&symbol==c.word[i+32];std::rotate(a.begin(),a.begin()+rank,a.begin()+rank+1);out.push_back(rank);}for(int j=0;j<256;j++){ok&=c.alpha[j+32]==c.refalpha[j];out.push_back(c.alpha[j+32]);}bool g=guards(c);return std::make_tuple(time,ok&&g,g);};std::vector<unsigned int>tmp;for(auto c:cases)run(*c,tmp);for(int trial=1;trial<=5;trial++){auto out=prefix;double time=0;bool ok=true;std::vector<double>times;for(auto c:cases){auto[t,pass,g]=run(*c,out);time+=t;ok&=pass;times.push_back(t);std::cout<<"STUDY_GUARDS trial="<<trial<<" case="<<c->id<<" exact="<<(g?"true":"false")<<std::endl;}std::cout<<"STUDY_COMPONENT_SECONDS ";for(auto t:times)std::cout<<std::setprecision(17)<<t<<",";std::cout<<std::endl;std::string file="mtf_"+std::to_string(trial)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<std::setprecision(17)<<time<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":0,\"timing_region\":\"sum9_preallocated_corrected_online_hostdriven_mtf\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\""<<file<<"\",\"dtype\":\"uint32\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}}
