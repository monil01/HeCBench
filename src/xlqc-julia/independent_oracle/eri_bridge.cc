#include "symbol_prefix.h"
#include "cints.h"
extern "C" double xl_oracle_boys(int m,double x){return Fgamma(m,x);}
static double primitive(const double*a,const double*b,const double*c,const double*d){
return coulomb_repulsion(a[0],a[1],a[2],a[7],int(a[3]),int(a[4]),int(a[5]),a[6],b[0],b[1],b[2],b[7],int(b[3]),int(b[4]),int(b[5]),b[6],c[0],c[1],c[2],c[7],int(c[3]),int(c[4]),int(c[5]),c[6],d[0],d[1],d[2],d[7],int(d[3]),int(d[4]),int(d[5]),d[6]);}
extern "C" void xl_oracle_eri_table(const double*p,int n,double*out){long t=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)for(int l=0;l<n;l++)out[t++]=primitive(p+8*i,p+8*j,p+8*k,p+8*l);}
