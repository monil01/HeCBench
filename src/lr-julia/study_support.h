#pragma once
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>
struct Case{int n,repeat;std::vector<float>d;};
inline std::vector<Case> cases(){std::vector<Case>v;for(int n:{63,126,96453}){Case c{n,n==96453?7:3,{}};for(int i=0;i<n;i++){float x=(i%63-31)/16.f;float noise=n==63?0.f:((i*17)%7-3)/64.f;c.d.push_back(x);c.d.push_back(1.25f+.75f*x+noise);}v.push_back(c);}return v;}
inline void coefficients(const std::vector<float>&v,int n,float&m,float&a,float&b){float x=0,y=0,xy=0,xx=0;for(size_t i=0;i<v.size();i+=4){x+=v[i];y+=v[i+1];xy+=v[i+2];xx+=v[i+3];}float nf=n;double denom=nf*xx-x*x;a=(y*xx-x*xy)/denom;b=(nf*xy-x*y)/denom;m=y/nf;}
inline std::vector<float> tree(const Case&c,int stage,float m=0,float a=0,float b=0){int fields=stage==1?4:2;std::vector<float>o;for(int g=0;g<c.n/63;g++){float s[63][4];for(int l=0;l<63;l++){float x=c.d[2*(g*63+l)],y=c.d[2*(g*63+l)+1];if(stage==1){s[l][0]=x;s[l][1]=y;s[l][2]=x*y;s[l][3]=x*x;}else{s[l][0]=(y-m)*(y-m);float e=x*b+a;s[l][1]=(e-m)*(e-m);}}for(int i=31,old=63;i>0;old=i,i/=2){for(int l=0;l<i;l++)for(int f=0;f<fields;f++){s[l][f]+=s[l+i][f];if(l==i-1&&old%2)s[l][f]+=s[old-1][f];}}for(int f=0;f<fields;f++)o.push_back(s[0][f]);}return o;}
inline constexpr int prefixcount=1+3*2+2*(63+126+96453);
inline constexpr int outcount=prefixcount+6*(1+2+1531)+3*(4+384);
extern "C" int study_check(const char*,double*);
