#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
// Independent tensor identities and externally generated LAPACK singular values.
// No source SVD body, GPU output substitution or GPU work in this checker.
extern "C" int svd_check(const float* a,const float* o,const double* sigma,int n,double* canonical,double* maxima){
  for(int k=0;k<5;k++)maxima[k]=0.;
  int good=1;
  for(int t=0;t<n;t++){
    double A[9],U[9],S[3],V[9],norm=0.;
    for(int k=0;k<9;k++){A[k]=a[t+k*n];U[k]=o[t+k*n];V[k]=o[t+(k+12)*n];norm+=A[k]*A[k];good&=std::isfinite(A[k])&&std::isfinite(U[k])&&std::isfinite(V[k]);}
    for(int k=0;k<3;k++){S[k]=o[t+(k+9)*n];good&=std::isfinite(S[k]);}
    double d=std::max(1.,std::sqrt(norm)),rec=0.,ue=0.,ve=0.;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      double r=0.,u=0.,v=0.;for(int k=0;k<3;k++){r+=U[3*i+k]*S[k]*V[3*j+k];u+=U[3*k+i]*U[3*k+j];v+=V[3*k+i]*V[3*k+j];}
      rec+=(r-A[3*i+j])*(r-A[3*i+j]);u-=i==j;v-=i==j;ue+=u*u;ve+=v*v;
    }
    auto det=[](const double* x){return x[0]*(x[4]*x[8]-x[5]*x[7])-x[1]*(x[3]*x[8]-x[5]*x[6])+x[2]*(x[3]*x[7]-x[4]*x[6]);};
    double ss[3]={std::abs(S[0]),std::abs(S[1]),std::abs(S[2])};std::sort(ss,ss+3,std::greater<double>());double se=0.;
    for(int k=0;k<3;k++){se=std::max(se,std::abs(ss[k]-sigma[3*t+k])/d);canonical[3*t+k]=ss[k]/d;}
    double e[5]={std::sqrt(rec)/d,se,std::sqrt(ue),std::sqrt(ve),std::max(std::abs(det(U)-1),std::abs(det(V)-1))};
    for(int k=0;k<5;k++)maxima[k]=std::max(maxima[k],e[k]);
  }
  const double bounds[5]={.002,.002,.0002,.0002,.001};
  for(int k=0;k<5;k++)good&=std::isfinite(maxima[k])&&maxima[k]<=bounds[k];
  return good;
}
