#include <cmath>
#include <cstdio>
#include <cstring>
#define __device__
#define __host__
inline float __int_as_float(unsigned x){float v;std::memcpy(&v,&x,4);return v;}
inline double __longlong_as_double(unsigned long long x){double v;std::memcpy(&v,&x,8);return v;}
#include "../interval-cuda/cuda_interval_rounded_arith.h"
int main(){rounded_arith<double>r;double a=1,b=std::ldexp(1.,-53),c=3*b;std::printf("{\"upper_case\":{\"a\":\"%a\",\"b\":\"%a\",\"actual_add_up\":\"%a\"},\"lower_case\":{\"a\":\"%a\",\"b\":\"%a\",\"actual_add_down\":\"%a\"}}\n",a,b,r.add_up(a,b),a,c,r.add_down(a,c));}
