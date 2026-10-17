#include <cmath>
#include <cstdio>
#include <initializer_list>
extern double Fresnel_Sine_Integral(double);
int main(){double ma=0;for(double x:{0.25,0.5,1.,2.,4.,7.,8.}){const int steps=20000;double h=x/steps,s=0;for(int i=0;i<=steps;i++){double t=h*i;s+=(i==0||i==steps?1:i%2?4:2)*sin(t*t);}double gold=s*h/3*sqrt(2/acos(-1.)),got=Fresnel_Sine_Integral(x);double e=fabs(got-gold);ma=fmax(ma,e);printf("FRESNEL_MATH_WITNESS x=%.8g native_host=%.17g independent_Simpson=%.17g abs_error=%.12g\n",x,got,gold,e);}printf("FRESNEL_MATH_SUMMARY max_abs_error=%.12g source_tolerance=1e-6\n",ma);}
