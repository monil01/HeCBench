#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
struct ICoverage { bool pass=true; int missing=0,extra=0; };
inline int polynomial_sign(double x,int i){if(!(x>=.25&&x<=4))return 2;int exponent;double fraction=std::frexp(x,&exponent);uint64_t n=uint64_t(std::ldexp(fraction,53));uint64_t d=uint64_t(1)<<(53-exponent);__int128 nn=n,dd=d;__int128 v=__int128(65536)*nn*nn-__int128(131072+i)*nn*dd+__int128(65536)*dd*dd;return(v>0)-(v<0);}
inline void roots_in(double lo,double up,int i,bool&lower,bool&upper){if(i==0){lower=upper=lo<=1&&up>=1;return;}lower=lo<=1&&up>=.25&&(lo<=.25||polynomial_sign(lo,i)>=0)&&(up>=1||polynomial_sign(up,i)<=0);upper=up>=1&&(lo<=1||polynomial_sign(lo,i)<=0)&&(up>=4||polynomial_sign(up,i)>=0);}
inline ICoverage check_coverage(const void*raw,const int*counts){ICoverage c;for(int i=0;i<65536;i++){int k=counts[i];if(k<0||k>128){c.pass=false;c.missing++;continue;}bool lower=false,upper=false;for(int j=0;j<k;j++){double v[2];std::memcpy(v,(const unsigned char*)raw+16*(size_t(j)*65536+i),16);if(!std::isfinite(v[0])||!std::isfinite(v[1])||v[0]>v[1]){c.pass=false;continue;}bool l=false,u=false;roots_in(v[0],v[1],i,l,u);lower|=l;upper|=u;if(!l&&!u)c.extra++;}if(!lower||!upper){c.missing++;c.pass=false;}}return c;}
