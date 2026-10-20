#pragma once
// Independent CPU block-twist MT oracle and LongDouble Box–Muller witness.
// No CUDA or source GPU kernels are included here.
#include <cstdint>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
struct MTOracleMetrics{uint64_t checked,uniform_mismatches,gaussian_failures;double max_abs_error,max_bound_fraction,sum,sum_squares,min_value,max_value;};
inline uint32_t mt_float_bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
inline MTOracleMetrics mt_check(const uint32_t*parameters,const float*uniform,const float*gaussian,int n){MTOracleMetrics out{};out.min_value=1e300;out.max_value=-1e300;const long double unit=0x1p-24L;const float pi=3.14159265358979f;for(int stream=0;stream<4096;stream++){auto*p=parameters+4*stream;std::array<uint32_t,19>state;state[0]=p[3];for(uint32_t k=1;k<19;k++)state[k]=1812433253U*(state[k-1]^(state[k-1]>>30))+k;int cursor=19;for(int j=0;j<n;j++){if(cursor==19){for(int k=0;k<19;k++){uint32_t x=(state[k]&0xfffffffeU)|(state[(k+1)%19]&1U);state[k]=state[(k+9)%19]^(x>>1)^((x&1U)?p[0]:0U);}cursor=0;}uint32_t x=state[cursor++];x^=x>>12;x^=(x<<7)&p[1];x^=(x<<15)&p[2];x^=x>>18;float ref=(float(x)+1.f)/4294967296.f;size_t offset=stream+size_t(j)*4096;if(mt_float_bits(ref)!=mt_float_bits(uniform[offset]))out.uniform_mismatches++;out.checked++;}
 for(int j=0;j<n;j+=2){size_t i=stream+size_t(j)*4096;long double u1=uniform[i],u2=uniform[i+4096];if(!(u1>0&&u1<=1&&u2>0&&u2<=1)){out.gaussian_failures+=2;continue;}long double radius=std::sqrt(-2.L*std::log(u1));float phi_float=(2.f*pi)*float(u2);long double phi=phi_float;long double refs[]={radius*std::cos(phi),radius*std::sin(phi)};long double budget=32.L*unit*(1.L+radius+std::abs(std::log(u1)));for(int k=0;k<2;k++){double actual=gaussian[i+size_t(k)*4096];long double error=std::abs((long double)actual-refs[k]);if(!std::isfinite(actual)||error>budget)out.gaussian_failures++;out.max_abs_error=std::max(out.max_abs_error,double(error));out.max_bound_fraction=std::max(out.max_bound_fraction,double(error/budget));out.sum+=actual;out.sum_squares+=actual*actual;out.min_value=std::min(out.min_value,actual);out.max_value=std::max(out.max_value,actual);out.checked++;}}}return out;}
extern "C" inline int mt_check_abi_impl(const uint32_t*p,const float*u,const float*g,int n,MTOracleMetrics*m){if(!p||!u||!g||!m||n<2||n%2)return 2;*m=mt_check(p,u,g,n);return(m->uniform_mismatches||m->gaussian_failures)?1:0;}
