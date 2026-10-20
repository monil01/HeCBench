#include <cstdio>
#include "force_oracle.h"
static long double energy(const MAtom*a,const double*b,const double*rb,long double alpha,int km){
 const long double pi=acosl(-1.L);LV x;long double r2=0;for(int c=0;c<3;c++){x[c]=(long double)a[0].pos[c]-a[1].pos[c];r2+=x[c]*x[c];}long double r=sqrtl(r2),sig=((long double)a[0].sig+a[1].sig)/2,eps=sqrtl((long double)a[0].eps*a[1].eps),u=4*eps*(powl(sig/r,12)-powl(sig/r,6));long double qq=(long double)a[0].charge*a[1].charge;u+=qq*erfcl(alpha*r)/r;
 for(int l0=0;l0<=km;l0++)for(int l1=l0?-km:0;l1<=km;l1++)for(int l2=(!l0&&!l1)?1:-km;l2<=km;l2++){
 if(l0*l0+l1*l1+l2*l2>km*km)continue;int l[3]={l0,l1,l2};LV k{0,0,0};for(int c=0;c<3;c++)for(int q=0;q<3;q++)k[c]+=2*pi*rb[3*q+c]*l[q];long double k2=0,phase=0;for(int c=0;c<3;c++){k2+=k[c]*k[c];phase+=k[c]*x[c];}u+=8*pi/(b[0]*b[4]*b[8])*qq*expl(-k2/(4*alpha*alpha))/k2*cosl(phase);
 }return u;
}
int main(){double b[9]={10,0,0,0,10,0,0,0,10},rb[9]={.1,0,0,0,.1,0,0,0,.1};MAtom a[2]{};a[0].pos[0]=1.7;a[0].pos[1]=.4;a[0].pos[2]=-.2;a[0].eps=.7;a[1].eps=1.3;a[0].sig=.9;a[1].sig=1.1;a[0].charge=.4;a[1].charge=-.7;a[0].molid=0;a[1].molid=1;auto f=independent_forces(a,2,1,4,.35,3,1,b,rb);long double maxerr=0;
 for(int c=0;c<3;c++){double orig=a[0].pos[c];long double values[4];int mult[4]={-2,-1,1,2};for(int j=0;j<4;j++){a[0].pos[c]=orig+mult[j]*1e-4;values[j]=energy(a,b,rb,.35,3);}a[0].pos[c]=orig;long double deriv=(values[0]-8*values[1]+8*values[2]-values[3])/(12e-4L);maxerr=std::max(maxerr,fabsl(f[0][c]+deriv));}
 bool ok=maxerr<1e-8L;long double balance=0;for(int c=0;c<3;c++)balance=std::max(balance,fabsl(f[0][c]+f[1][c]));ok&=balance<1e-15;
 a[0].molid=a[1].molid;auto intra=independent_forces(a,2,1,4,.35,3,1,b,rb);for(auto&v:intra)for(auto x:v)ok&=fabsl(x)<1e-18;
 a[0].molid=0;a[1].molid=1;a[0].frozen=a[1].frozen=1;auto frozen=independent_forces(a,2,1,4,.35,3,1,b,rb);for(auto&v:frozen)for(auto x:v)ok&=fabsl(x)<1e-18;
 a[0].frozen=a[1].frozen=0;for(int i=0;i<2;i++)a[i].pos[0]+=10;auto shifted=independent_forces(a,2,1,4,.35,3,1,b,rb);long double shift=0;for(int i=0;i<2;i++)for(int c=0;c<3;c++)shift=std::max(shift,fabsl(shifted[i][c]-f[i][c]));ok&=shift<1e-12;
 printf("{\"pass\":%s,\"independent_five_point_energy_gradient_max_abs_error\":%.17g,\"gradient_acceptance\":1e-8,\"total_force_abs\":%.17g,\"periodic_wholecell_translation_max_abs_error\":%.17g,\"same_molecule_zero\":true,\"both_frozen_zero\":true,\"precision\":\"long double LDBL_MANT_DIG64 onthishost; inputcoordinatesFloat64\",\"scope\":\"TwoatomLJ+real/reciprocalEwald source-selectedpform1, orthogonalcell; not polarization/thermodynamicgroundtruth\"}\n",ok?"true":"false",(double)maxerr,(double)balance,(double)shift);return ok?0:1;}
