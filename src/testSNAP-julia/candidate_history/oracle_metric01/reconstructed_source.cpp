// Independent SNAP operator: SU(2) symmetric-power polynomial + dual derivatives,
// full Clebsch-Gordan tensor energy reverse gradient. No native recursive dU/Y body.
#include "study_input.h"
#include <complex>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <omp.h>
using L=long double;using C=std::complex<L>;
struct D{L v;std::array<L,3>d{};D(L x=0):v(x){}D(L x,int k):v(x){d[k]=1;}};
D operator+(D a,D b){D c(a.v+b.v);for(int k=0;k<3;k++)c.d[k]=a.d[k]+b.d[k];return c;}
D operator-(D a,D b){D c(a.v-b.v);for(int k=0;k<3;k++)c.d[k]=a.d[k]-b.d[k];return c;}
D operator*(D a,D b){D c(a.v*b.v);for(int k=0;k<3;k++)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
D operator/(D a,D b){D c(a.v/b.v);for(int k=0;k<3;k++)c.d[k]=(a.d[k]-c.v*b.d[k])/b.v;return c;}
D sqrtd(D a){D c(sqrtl(a.v));for(int k=0;k<3;k++)c.d[k]=a.d[k]/(2*c.v);return c;}
D tand(D a){D c(tanl(a.v));for(int k=0;k<3;k++)c.d[k]=a.d[k]*(1+c.v*c.v);return c;}
D cosd(D a){D c(cosl(a.v));for(int k=0;k<3;k++)c.d[k]=-sinl(a.v)*a.d[k];return c;}
struct Z{C v{};std::array<C,3>d{};Z(C x=C{}):v(x){}Z(D r,D i):v(r.v,i.v){for(int k=0;k<3;k++)d[k]=C(r.d[k],i.d[k]);}};
Z operator+(const Z&a,const Z&b){Z c(a.v+b.v);for(int k=0;k<3;k++)c.d[k]=a.d[k]+b.d[k];return c;}
Z operator*(const Z&a,const Z&b){Z c(a.v*b.v);for(int k=0;k<3;k++)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
Z scale(Z a,L b){a.v*=b;for(auto&x:a.d)x*=b;return a;}
Z cj(Z a){a.v=std::conj(a.v);for(auto&x:a.d)x=std::conj(x);return a;}
Z multiply(D a,Z b){Z c; c.v=a.v*b.v;for(int k=0;k<3;k++)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
uint64_t choose(int n,int k){if(k<0||k>n)return 0;uint64_t a=1;for(int i=1;i<=k;i++)a=a*(n-i+1)/i;return a;}
std::vector<Z> polynomial(const SnapInput&s,int atom,int nbor,L&uniterr){
 int at=atom+s.na*nbor;D x(s.rij[at],0),y(s.rij[at+s.na*s.nb],1),z(s.rij[at+2*s.na*s.nb],2);D r=sqrtd(x*x+y*y+z*z);if(!(r.v>0))throw std::runtime_error("positive pair radius required");
 const L pi=acosl(-1.L);D theta=(r-D(s.scalars[3]))*D(s.scalars[2]*pi)/(D(s.rcut[at])-D(s.scalars[3]));D z0=r/tand(theta),norm=sqrtd(r*r+z0*z0);Z a(z0/norm,(D(0)-z)/norm),b(y/norm,(D(0)-x)/norm);
 std::array<std::vector<Z>,4>pow;std::array<Z,4>base{cj(a),scale(cj(b),-1),b,a};for(int q=0;q<4;q++){pow[q].resize(s.J+1);pow[q][0]=Z(C(1));for(int k=1;k<=s.J;k++)pow[q][k]=pow[q][k-1]*base[q];}
 D sf;if(r.v<=s.scalars[3])sf=D(1);else if(r.v>s.rcut[at])sf=D(0);else sf=D(.5)*(cosd((r-D(s.scalars[3]))*D(pi)/(D(s.rcut[at])-D(s.scalars[3])))+D(1));sf=sf*D(s.w[at]);
 std::vector<Z>u(s.nu);for(int j=0;j<=s.J;j++){std::vector<L>norms(j+1);for(int mb=0;mb<=j;mb++)for(int ma=0;ma<=j;ma++){
  Z value;for(int k=std::max(0,ma-(j-mb));k<=std::min(ma,mb);k++){Z term=pow[0][j-mb-ma+k]*pow[1][ma-k]*pow[2][mb-k]*pow[3][k];value=value+scale(term,L(choose(j-mb,ma-k))*L(choose(mb,k)));}
  value=scale(value,sqrtl(L(choose(j,mb))/L(choose(j,ma))));norms[mb]+=std::norm(value.v);u[s.iu[j]+(j+1)*mb+ma]=multiply(sf,value);
 }for(L v:norms)uniterr=std::max(uniterr,fabsl(v-1));}return u;
}
struct Term{int a,b,c;L w;};
std::vector<Term>energy_terms(const SnapInput&s,L&cgerr){std::vector<Term>terms;int coef=0;for(int j1=0;j1<=s.J;j1++)for(int j2=0;j2<=j1;j2++)for(int j=abs(j1-j2);j<=std::min(s.J,j1+j2);j+=2){
 int block=s.icg[j1+s.jdim*j2+s.jdim*s.jdim*j];std::vector<std::array<int,3>>r;std::vector<L>w;std::vector<L>norm(j+1);
 for(int m1=0;m1<=j1;m1++)for(int m2=0;m2<=j2;m2++){int m=(2*m1-j1+2*m2-j2+j)/2;if(m<0||m>j)continue;L v=s.cg[block+m1*(j2+1)+m2];norm[m]+=v*v;if(v!=0){r.push_back({m1,m2,m});w.push_back(v);}}
 for(L v:norm)cgerr=std::max(cgerr,fabsl(v-1));if(j<j1)continue;L beta=s.coeff[++coef];for(size_t ib=0;ib<r.size();ib++)for(size_t ia=0;ia<r.size();ia++){auto a=r[ia],b=r[ib];terms.push_back({s.iu[j1]+(j1+1)*b[0]+a[0],s.iu[j2]+(j2+1)*b[1]+a[1],s.iu[j]+(j+1)*b[2]+a[2],beta*w[ia]*w[ib]});}}
 if(coef!=s.ncoeff)throw std::runtime_error("coefficient enumeration");return terms;}
int main(int argc,char**argv){try{if(argc!=3)return 2;SnapInput s(argv[1]);L cgerr=0;auto terms=energy_terms(s,cgerr);std::vector<double>dedr(s.na*s.nb*3),force(s.nt*3);L unitmax=0;
#pragma omp parallel for reduction(max:unitmax) schedule(dynamic)
 for(int atom=0;atom<s.na;atom++){
  std::vector<C>total(s.nu),grad(s.nu);for(int j=0;j<=s.J;j++)for(int m=0;m<=j;m++)total[s.iu[j]+(j+1)*m+m]=s.scalars[1];std::vector<std::vector<Z>>u(s.nb);L ue=0;
  for(int n=0;n<s.nb;n++){u[n]=polynomial(s,atom,n,ue);for(int i=0;i<s.nu;i++)total[i]+=u[n][i].v;}
  unitmax=std::max(unitmax,ue);
  for(const auto&t:terms){C a=total[t.a],b=total[t.b],c=total[t.c];grad[t.c]+=t.w*a*b;grad[t.a]+=t.w*c*std::conj(b);grad[t.b]+=t.w*c*std::conj(a);}
  for(int n=0;n<s.nb;n++)for(int k=0;k<3;k++){L e=0;for(int i=0;i<s.nu;i++)e+=std::real(std::conj(grad[i])*u[n][i].d[k]);dedr[atom+s.na*n+s.na*s.nb*k]=double(e);}
 }
 for(int atom=0;atom<s.na;atom++)for(int n=0;n<s.nb;n++)for(int k=0;k<3;k++){int j=s.inside[atom+s.na*n];if(j<0||j>=s.nt)throw std::runtime_error("neighbor index");double d=dedr[atom+s.na*n+s.na*s.nb*k];force[atom+s.nt*k]+=d;force[j+s.nt*k]-=d;}
 double worst=0,budgetratio=0,rms=0;for(size_t i=0;i<force.size();i++){double err=abs(force[i]-s.reference[i]);worst=std::max(worst,err);budgetratio=std::max(budgetratio,err/(1e-7+1e-9*abs(s.reference[i])));rms+=err*err;}
 std::ofstream out(argv[2],std::ios::binary);out.write((char*)dedr.data(),dedr.size()*8);out.write((char*)force.data(),force.size()*8);if(!out)throw std::runtime_error("oracle output");
 printf("SNAP_INDEPENDENT J=%d atoms=%d nbor=%d terms=%zu unitary_max=%.21Lg CG_norm_max=%.21Lg author_force_max_abs=%.17g normalized_budget=%.17g rms_component=%.17g\n",s.J,s.na,s.nb,terms.size(),unitmax,cgerr,worst,budgetratio,sqrt(rms/force.size()));
 return (unitmax>1e-10L||cgerr>1e-11L||budgetratio>1)?3:0;
}catch(const std::exception&e){fprintf(stderr,"SNAP_ORACLE %s\n",e.what());return 2;}}
