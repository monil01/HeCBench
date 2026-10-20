#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#define PI 3.1415926535897931f
#include "independent_stage_oracle.h"
int main(int argc,char**argv){
 GMMState x;x.N={4};x.pi={1};x.constant={float(-std::log(2.0*double(PI)))};x.avgvar={1e-6f};x.means={0,0};x.R={1,0,0,1};x.Rinv=x.R;x.memberships=std::vector<float>(4,1);
 std::vector<float>data={-1,-1,1,1,-1,1,-1,1};
 auto out=x;gmm_scientific_stage(2,1,2,4,1,1,data,x,out,{});
 for(auto &v:out.memberships)v=x.constant[0]-1;gmm_scientific_stage(3,1,2,4,1,24,data,x,out,{});
 auto before=out;out.memberships=std::vector<float>(4,argc>1?.5f:1.f);std::vector<float>likelihood(24,0);likelihood[23]=4*(x.constant[0]-1);gmm_scientific_stage(4,1,2,4,24,1,data,before,out,likelihood);
 before=out;out.pi={4};gmm_scientific_stage(5,1,2,4,1,1,data,before,out,{});
 gmm_scientific_stage(6,1,2,4,1,2,data,before,out,{});
 out.R={4+1e-6f,0,0,4+1e-6f};gmm_scientific_stage(7,1,2,4,1,3,data,before,out,{});
 printf("ANALYTIC_SYMMETRIC_ONE_GAUSSIAN_PASS checked%zu\n",oracle_checked);return 0;
}
