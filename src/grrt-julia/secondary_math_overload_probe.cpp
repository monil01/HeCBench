#include <cmath>
#include <type_traits>
#include <cstdio>
#include <cfloat>
int main(){printf("{\"LDBL_MANT_DIG\":%d,\"global_sin_longdouble\":%s,\"global_log_longdouble\":%s,\"global_acos_longdouble\":%s,\"std_acos_longdouble\":%s}\n",LDBL_MANT_DIG,std::is_same_v<decltype(::sin(1.L)),long double>?"true":"false",std::is_same_v<decltype(::log(1.L)),long double>?"true":"false",std::is_same_v<decltype(::acos(1.L)),long double>?"true":"false",std::is_same_v<decltype(std::acos(1.L)),long double>?"true":"false");}
