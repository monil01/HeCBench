#include <cmath>
#include <cstdio>
#include <type_traits>
#include "../grrt-cuda/constants.h"
int main(){long double distance=C_sgrA_d*C_pc;long double u=10.L*C_sgrA_mbh*C_rgeo/distance;long double old=::atan(u),full=std::atan(u);printf("{\"global_atan_longdouble\":%s,\"argument\":%.21Lg,\"global_angle\":%.21Lg,\"std_extended_angle\":%.21Lg,\"relative_luminosity_prefactor_difference\":%.21Lg}\n",std::is_same<decltype(::atan(u)),long double>::value?"true":"false",u,old,full,(old*old-full*full)/(full*full));}
