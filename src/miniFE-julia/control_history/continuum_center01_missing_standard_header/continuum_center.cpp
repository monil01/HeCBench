// Independent unit-cube continuum solution at the center. Harmonic one-hot-face
// part is exactly1/6 by cube symmetry. Constant unit-forcing/zero-boundary part
// has sine coefficients64/(pi^5*l*m*n*(l^2+m^2+n^2)), odd l,m,n.
#include <cmath>
#include <cstdio>
#include <cfloat>
int main(){const long double pi=acosl(-1.L);long double previous=0;bool first=true;printf("{\"harmonic_hot_face_center\":%.21Lg,\"mantissa_bits\":%d,\"truncations\":[",1.L/6,LDBL_MANT_DIG);
 for(int terms:{16,32,64,128}){
  long double sum=0,correction=0;
  for(int p=0;p<terms;p++)for(int q=0;q<terms;q++)for(int r=0;r<terms;r++){
   long double a=2*p+1,b=2*q+1,c=2*r+1,term=((p+q+r)%2?-1.L:1.L)/(a*b*c*(a*a+b*b+c*c));long double y=term-correction,t=sum+y;correction=(t-sum)-y;sum=t;
  }
  long double forcing=64/powl(pi,5)*sum,total=1.L/6+forcing;
  printf("%s{\"odd_modes_per_axis\":%d,\"unit_forcing_center\":%.21Lg,\"full_Poisson_center\":%.21Lg,\"difference_from_previous_truncation\":",first?"":",",terms,forcing,total);
  if(first)printf("null");else printf("%.21Lg",fabsl(total-previous));printf("}");previous=total;first=false;
 }printf("],\"scope\":\"Long-double compensated finite sine sums. Observed truncation convergence, not a rigorous global truncation bound; continuum comparison includes FE/rounded-quadrature/solver errors.\"}\n");}
