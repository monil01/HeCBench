#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
using std::abs;
struct Index{unsigned x,y,z;};static Index blockIdx{0,0,0},threadIdx{0,0,0},blockDim{8,8,1};
float rsqrtf(float x){return 1.f/std::sqrt(x);}
#define __global__
#define MKX 4
#ifndef WSM_WITNESS_HEADER
#define WSM_WITNESS_HEADER "../wsm5-cuda/kernel.h"
#endif
#include WSM_WITNESS_HEADER
int main(){std::vector<float>th(3,280.f),pii(3,1.f),q(3,.0001f),qc(3,0.f),qi(3,0.f),qr(3,0.f),qs(3,0.f),den(3,1.f),p(3,90000.f),dz(3,500.f);float rain=0.f,rainncv=0.f,sr=0.f,snow=0.f,snowncv=0.f;wsm(th.data(),pii.data(),q.data(),qc.data(),qi.data(),qr.data(),qs.data(),den.data(),p.data(),dz.data(),&rain,&rainncv,&sr,&snow,&snowncv,10.f,1,1,1,1,1,3,1,1,1,1,1,3,1,1,1,1,1,3);std::cout<<std::setprecision(17)<<"rain="<<rain<<" rainncv="<<rainncv<<" snow="<<snow<<" snowncv="<<snowncv<<" sr="<<sr<<" th="<<th[0]<<" q="<<q[0]<<"\n";}
