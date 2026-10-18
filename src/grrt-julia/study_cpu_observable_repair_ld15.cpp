#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>
#include "../grrt-cuda/constants.h"
#undef IMAGE_SIZE
#define IMAGE_SIZE 8
namespace d12{using Real=double;using std::sin;using std::cos;using std::sqrt;using std::pow;using std::exp;using std::expm1;using std::fabs;using std::acos;using std::atan2;using std::log;using std::floor;using std::fmax;
#define GR_ESCAL 1e12
#include "study_cpu_observable_repair.hpp"
#undef GR_ESCAL
}
namespace d14{using Real=double;using std::sin;using std::cos;using std::sqrt;using std::pow;using std::exp;using std::expm1;using std::fabs;using std::acos;using std::atan2;using std::log;using std::floor;using std::fmax;
#define GR_ESCAL 1e14
#include "study_cpu_observable_repair.hpp"
#undef GR_ESCAL
}
namespace d15{using Real=double;using std::sin;using std::cos;using std::sqrt;using std::pow;using std::exp;using std::expm1;using std::fabs;using std::acos;using std::atan2;using std::log;using std::floor;using std::fmax;
#define GR_ESCAL 1e15
#include "study_cpu_observable_repair.hpp"
#undef GR_ESCAL
}
namespace ld14{using Real=long double;using std::sin;using std::cos;using std::sqrt;using std::pow;using std::exp;using std::expm1;using std::fabs;using std::acos;using std::atan2;using std::log;using std::floor;using std::fmax;
#define GR_ESCAL 1e14
#include "study_cpu_observable_repair.hpp"
#undef GR_ESCAL
}
namespace ld15{using Real=long double;using std::sin;using std::cos;using std::sqrt;using std::pow;using std::exp;using std::expm1;using std::fabs;using std::acos;using std::atan2;using std::log;using std::floor;using std::fmax;
#define GR_ESCAL 1e15
#include "study_cpu_observable_repair.hpp"
#undef GR_ESCAL
}
template<class T,class F1,class F2>void run(const char*name,F1 f1,F2 f2,int&x,int&y,long long&steps,long long&ev,long long&segs,long long&neg,long long&qa,long long&qr,long long&qf,T&res){T input[4]={T(0),T(acos(.25)/PI*180),T(8),T(340e9)},out[192];std::vector<T>reds,lums;for(y=0;y<8;y++)for(x=0;x<8;x++)f1(out,input,0,0);for(int i=0;i<64;i++)reds.push_back(out[3*i+2]);input[1]=45;for(y=0;y<8;y++)for(x=0;x<8;x++){f2(out,input,0,0);if(x==7){fprintf(stderr,"case=%s row=%d quad_accepted=%lld rejected=%lld failed=%lld\n",name,y,qa,qr,qf);fflush(stderr);}}for(int i=0;i<64;i++)lums.push_back(out[3*i+2]);printf("{\"case\":\"%s\",\"accepted_steps\":%lld,\"localized_events\":%lld,\"midpoint_segments\":%lld,\"negative_transfer_increments\":%lld,\"quadrature_accepted\":%lld,\"quadrature_rejected\":%lld,\"quadrature_failed\":%lld,\"max_plane_residual\":%.21Lg,\"redshift\":[",name,steps,ev,segs,neg,qa,qr,qf,(long double)res);for(int i=0;i<64;i++)printf("%s%.21Lg",i?",":"",(long double)reds[i]);printf("],\"luminosity\":[");for(int i=0;i<64;i++)printf("%s%.21Lg",i?",":"",(long double)lums[i]);printf("]}\n");fflush(stdout);}
// CPU-only completion ofcase truncated byprevious120secondbudget; sameheader/math.
int main(){
run("ld15",ld15::fixedcpu_task1,ld15::fixedcpu_task2,ld15::fixedcpu_x,ld15::fixedcpu_y,ld15::fixedcpu_accepted_steps,ld15::localized_events,ld15::midpoint_segments,ld15::negative_transfer_increments,ld15::quad_accepted,ld15::quad_rejected,ld15::quad_failed,ld15::max_event_theta_residual);
}
