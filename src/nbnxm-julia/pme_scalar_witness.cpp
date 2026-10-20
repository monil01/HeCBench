#include <cstdio>
#include <cmath>
#include <algorithm>
float pme_unfused(const float z2)
{
  constexpr float FN6 = -1.7357322914161492954e-8F;
  constexpr float FN5 = 1.4703624142580877519e-6F;
  constexpr float FN4 = -0.000053401640219807709149F;
  constexpr float FN3 = 0.0010054721316683106153F;
  constexpr float FN2 = -0.019278317264888380590F;
  constexpr float FN1 = 0.069670166153766424023F;
  constexpr float FN0 = -0.75225204789749321333F;

  constexpr float FD4 = 0.0011193462567257629232F;
  constexpr float FD3 = 0.014866955030185295499F;
  constexpr float FD2 = 0.11583842382862377919F;
  constexpr float FD1 = 0.50736591960530292870F;
  constexpr float FD0 = 1.0F;

  const float z4 = z2 * z2;
        float polyFD0 = FD4 * z4 + FD2;
  const float polyFD1 = FD3 * z4 + FD1;
  polyFD0 = polyFD0 * z4 + FD0;
  polyFD0 = polyFD1 * z2 + polyFD0;
  polyFD0 = 1.0F / polyFD0;

  float polyFN0 = FN6 * z4 + FN4;
  float polyFN1 = FN5 * z4 + FN3;
  polyFN0       = polyFN0 * z4 + FN2;
  polyFN1       = polyFN1 * z4 + FN1;
  polyFN0       = polyFN0 * z4 + FN0;
  polyFN0       = polyFN1 * z2 + polyFN0;

  return polyFN0 * polyFD0;
}

float pme_fused(float z2){
 const float z4=z2*z2;
 float fd0=fmaf(.0011193462567257629232f,z4,.11583842382862377919f);
 float fd1=fmaf(.014866955030185295499f,z4,.50736591960530292870f);
 fd0=fmaf(fd0,z4,1.f);fd0=fmaf(fd1,z2,fd0);fd0=1.f/fd0;
 float fn0=fmaf(-1.7357322914161492954e-8f,z4,-.000053401640219807709149f);
 float fn1=fmaf(1.4703624142580877519e-6f,z4,.0010054721316683106153f);
 fn0=fmaf(fn0,z4,-.019278317264888380590f);fn1=fmaf(fn1,z4,.069670166153766424023f);
 fn0=fmaf(fn0,z4,-.75225204789749321333f);fn0=fmaf(fn1,z2,fn0);return fn0*fd0;
}
long double exact(long double x){long double pi=acosl(-1.L);if(x<.01L){long double s=0,t=1;for(int k=0;k<40;k++){s+=t/(2*k+3);t*=-x/(k+1);}return -4/sqrtl(pi)*s;}long double z=sqrtl(x);return 2*expl(-x)/(sqrtl(pi)*x)-erfl(z)/(x*z);}
int main(){float beta=3.12341f,upper=beta*beta;long double ma[2]={},mr[2]={};float worst[2]={};int samples=100001;
 for(int k=0;k<samples;k++){float x=upper*(double(k)/(samples-1));long double gold=exact(x);float vals[2]={pme_unfused(x),pme_fused(x)};for(int m=0;m<2;m++){long double err=fabsl(vals[m]-gold);if(err>ma[m]){ma[m]=err;worst[m]=x;}mr[m]=std::max(mr[m],err/fabsl(gold));}}
 printf("{\"beta_float32\":%.9g,\"z2_upper_float32\":%.9g,\"samples\":%d,\"longdouble_mantissa_bits\":%d,\"variants\":[",beta,upper,samples,__LDBL_MANT_DIG__);
 for(int m=0;m<2;m++)printf("%s{\"arithmetic\":\"%s\",\"max_abs_error\":%.17Lg,\"max_rel_error\":%.17Lg,\"worst_z2\":%.9g,\"sampled_budget_3e_minus6_pass\":%s}",m?",":"",m?"explicitFloat32FMA":"sourceFloat32unfused",ma[m],mr[m],worst[m],ma[m]<=3e-6L?"true":"false");
 printf("],\"scope\":\"Dense sampled CPU polynomial-versus-analytic Ewald-correction control; not a rigorous global bound or actual GPU instruction trace.\"}\n");return(ma[0]<=3e-6L&&ma[1]<=3e-6L)?0:2;}
