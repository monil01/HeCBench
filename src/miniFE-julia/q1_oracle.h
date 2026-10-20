#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
struct Q1Input{
 int dim=0,n=0,nz=0;std::vector<int>rp,ci;std::vector<double>A,b,x,r,p,Ap;
};
struct Q1MatrixReview{bool pass=true;double max_A_abs=0,max_b_abs=0;long long entries=0;};
static bool q1_hot(int x,int dim){return x==dim;}
static bool q1_boundary(int x,int y,int z,int dim){return x==0||y==0||z==0||x==dim||y==dim||z==dim;}
static long double q1_coefficient(int dx,int dy,int dz,int dim){
 const long double g=(long double)(double)0.577350269;const long double m0=(1+g*g)/2,m1=(1-g*g)/4;
 const long double mx=dx?m1:m0,my=dy?m1:m0,mz=dz?m1:m0,kx=dx?-1:2,ky=dy?-1:2,kz=dz?-1:2;
 return (kx*my*mz+mx*ky*mz+mx*my*kz)/dim;
}
static Q1MatrixReview q1_review_matrix(const Q1Input&v){
 Q1MatrixReview out;const int s=v.dim+1;long long expected_n=(long long)s*s*s;
 if(v.n!=expected_n||v.rp.size()!=(size_t)v.n+1||v.ci.size()!=(size_t)v.nz||v.A.size()!=(size_t)v.nz||v.b.size()!=(size_t)v.n||v.rp[0]!=0||v.rp.back()!=v.nz){out.pass=false;return out;}
 for(int row=0;row<v.n;row++){
  int xx=row%s,yy=(row/s)%s,zz=row/(s*s);bool edge=q1_boundary(xx,yy,zz,v.dim);long double rhs=edge?(q1_hot(xx,v.dim)?1.L:0.L):1.L/((long double)v.dim*v.dim*v.dim);
  int at=v.rp[row];if(at<0||v.rp[row+1]<at||v.rp[row+1]>v.nz){out.pass=false;return out;}
  for(int z=std::max(0,zz-1);z<=std::min(v.dim,zz+1);z++)for(int y=std::max(0,yy-1);y<=std::min(v.dim,yy+1);y++)for(int x=std::max(0,xx-1);x<=std::min(v.dim,xx+1);x++){
   int col=x+s*y+s*s*z;if(at>=v.rp[row+1]||v.ci[at]!=col){out.pass=false;return out;}long double expected;
   if(edge)expected=col==row?1:0;
   else{long double c=q1_coefficient(x-xx,y-yy,z-zz,v.dim);if(q1_hot(x,v.dim))rhs-=c;expected=q1_boundary(x,y,z,v.dim)?0:c;}
   double err=(double)fabsl((long double)v.A[at]-expected);out.max_A_abs=std::max(out.max_A_abs,err);if(!std::isfinite(v.A[at])||err>2e-13)out.pass=false;at++;out.entries++;
  }
  if(at!=v.rp[row+1]){out.pass=false;return out;}double e=(double)fabsl((long double)v.b[row]-rhs);out.max_b_abs=std::max(out.max_b_abs,e);if(!std::isfinite(v.b[row])||e>2e-13)out.pass=false;
 }
 return out;
}
struct Q1SolutionReview{bool pass=true;double relative_true_residual=0,max_boundary_abs=0,min_u=0,max_u=0,center=0;std::vector<double>residual;};
static Q1SolutionReview q1_review_solution(const Q1Input&v,const std::vector<double>&x){
 Q1SolutionReview o;o.residual.resize(v.n);long double rr=0,bb=0;o.min_u=1e300;o.max_u=-1e300;int s=v.dim+1;
 for(int i=0;i<v.n;i++){
  long double ax=0;for(int j=v.rp[i];j<v.rp[i+1];j++)ax+=(long double)v.A[j]*x[v.ci[j]];long double residual=(long double)v.b[i]-ax;o.residual[i]=(double)residual;rr+=residual*residual;bb+=(long double)v.b[i]*v.b[i];
  if(!std::isfinite(x[i]))o.pass=false;o.min_u=std::min(o.min_u,x[i]);o.max_u=std::max(o.max_u,x[i]);int xx=i%s,yy=(i/s)%s,zz=i/(s*s);
  if(q1_boundary(xx,yy,zz,v.dim))o.max_boundary_abs=std::max(o.max_boundary_abs,std::abs(x[i]-(q1_hot(xx,v.dim)?1.:0.)));
 }
 o.relative_true_residual=(double)sqrtl(rr/std::max(1.L,bb));o.center=x[(v.dim/2)*(1+s+s*s)];o.pass&=o.relative_true_residual<=1e-7&&o.max_boundary_abs<=1e-6&&o.min_u>=-1e-6&&o.max_u<=1.125+1e-6;return o;
}
