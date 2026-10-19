#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
struct Case{int n,f,lo,hi,restarts;std::vector<float>x;};
inline std::vector<Case>cases(){std::vector<Case>v;for(int n:{17,257,4099}){Case c{n,n==17?3:(n==257?4:7),n==17?2:5,n==17?4:15,n==17?2:10,{}};for(int i=0;i<c.n;i++)for(int f=0;f<c.f;f++)c.x.push_back(2.f*(i%5)+(f+1)*(i%5+1)/32.f+(n==17?0.f:((i%7)-3)/256.f));v.push_back(c);}return v;}
inline int nearest(const std::vector<float>&x,int i,int f,const std::vector<float>&cent,int k){int id=0;float best=std::numeric_limits<float>::max();for(int j=0;j<k;j++){float d=0;for(int a=0;a<f;a++){float z=x[i*f+a]-cent[j*f+a];d+=z*z;}if(d<best){best=d;id=j;}}return id;}
inline void update(const Case&c,std::vector<float>&cent,const std::vector<int>&mem,std::vector<int>&prev,int&delta){int k=cent.size()/c.f;std::vector<float>s(k*c.f,0);std::vector<int>counts(k,0);delta=0;for(int i=0;i<c.n;i++){int j=mem[i];if(j<0||j>=k)throw std::runtime_error("memberrange");counts[j]++;if(mem[i]!=prev[i]){delta++;prev[i]=mem[i];}for(int a=0;a<c.f;a++)s[j*c.f+a]+=c.x[i*c.f+a];}for(int j=0;j<k;j++)if(counts[j])for(int a=0;a<c.f;a++)cent[j*c.f+a]=s[j*c.f+a]/counts[j];}
inline float rmse(const Case&c,const std::vector<float>&cent){float sum=0;for(int i=0;i<c.n;i++){int j=nearest(c.x,i,c.f,cent,cent.size()/c.f);float d=0;for(int a=0;a<c.f;a++){float z=c.x[i*c.f+a]-cent[j*c.f+a];d+=z*z;}sum+=d;}return std::sqrt(sum/c.n);}
inline constexpr int prefixcount=1+3*5+17*3+257*4+4099*7;
extern "C" int study_check(const char*,double*);
