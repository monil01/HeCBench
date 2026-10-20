#include "study_oracle.h"
extern "C" int mt_check_abi(const uint32_t*p,const float*u,const float*g,int n,MTOracleMetrics*m){return mt_check_abi_impl(p,u,g,n,m);}
