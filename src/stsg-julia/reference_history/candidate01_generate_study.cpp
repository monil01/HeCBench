#include <cmath>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <cstdint>
#include "study_fixture_definition.h"
struct ID{int x=0,y=0,z=0;};ID blockIdx,blockDim,threadIdx;
#define __global__
#include "../stsg-cuda/Filter.cu"
int main(){std::vector<double>fixture{3},oracle;FILE*j=fopen("fixture_manifest.json","w");assert(j);fprintf(j,"{\"cases\":[");for(int which=0;which<3;which++){STSGFixture f(which);int X=f.nx,Y=f.ny,B=f.nb,Z=f.years,N=X*Y,W=(2*f.win+1)*(2*f.win+1);for(double x:{double(X),double(Y),double(B),double(Z),double(f.win),double(f.cosyear),double(f.sampcorr),double(f.snow)})fixture.push_back(x);for(short x:f.raw)fixture.push_back(x);for(unsigned char x:f.qa)fixture.push_back(x);
std::vector<float>nd(N*B*Z),qa(N*B*Z),ref(N*B),res(N*W*4),r3(N*W*3),out(N*B*Z),vin(N*B);std::vector<int>res1(N*B),index(N*W*3);auto each=[&](auto call){for(int jj=0;jj<Y;jj++)for(int ii=0;ii<X;ii++){blockIdx.x=ii;blockIdx.y=jj;blockDim.x=blockDim.y=1;call();}};each([&](){Short_to_Float(f.raw.data(),f.qa.data(),X,Y,B,Z,nd.data(),qa.data());});each([&](){Generate_NDVI_reference(f.cosyear,2,nd.data(),qa.data(),X,Y,B,Z,ref.data(),r3.data(),res1.data());});std::fill(r3.begin(),r3.end(),0);each([&](){Compute_d_res(nd.data(),qa.data(),ref.data(),0,Y,0,0,X,Y,B,Z,f.win,res.data());});each([&](){STSG_filter(nd.data(),qa.data(),ref.data(),0,Y,0,0,X,Y,B,Z,f.win,f.sampcorr,f.snow,out.data(),vin.data(),res.data(),r3.data(),index.data());});
for(auto&v:{nd,qa,ref,res,r3,out,vin})for(float x:v){assert(std::isfinite(x));oracle.push_back(x);}for(auto&v:{res1,index})for(int x:v)oracle.push_back(x);int zeros=0;for(float x:out)zeros+=x==0;fprintf(j,"%s{\"case\":%d,\"nx\":%d,\"ny\":%d,\"bands\":%d,\"years\":%d,\"win\":4,\"output_values\":%zu,\"zero_outputs\":%d,\"quality_scope\":\"%s\"}",which?",":"",which,X,Y,B,Z,out.size(),zeros,which==1?"mixed0good1marginal2snow3cloud":which==2?"manufacturedlowNDVIthresholdcontrol":"allgood");}
std::ofstream f("study_fixture.bin",std::ios::binary);f.write((char*)fixture.data(),fixture.size()*8);std::ofstream o("study_oracle.bin",std::ios::binary);o.write((char*)oracle.data(),oracle.size()*8);fprintf(j,"],\"fixture_count\":%zu,\"source_CPU_oracle_count\":%zu,\"atol\":0.00002,\"rtol\":0.00002,\"source_derived_CPU_oracle_not_independent_math\":true,\"scope\":\"All4unchangedSTSGkernels,singlefullimage,noGDALrasterIO/chunkstreaming,12bands/3or4years/window4,actualsourceSGcoefficients\"}\n",fixture.size(),oracle.size());fclose(j);printf("STSG3CASES fixture=%zu sourceCPUoracle=%zu\n",fixture.size(),oracle.size());}
