#include "study_support.h"
extern "C" {
int hbc_ncases(){return hload().size();}
int hbc_field(int c,int field){static auto cs=hload();return field==0?cs[c].n:cs[c].m;}
const int*hbc_array(int c,int field){static auto cs=hload();return field==0?cs[c].R.data():field==1?cs[c].C.data():cs[c].F.data();}
int hbc_check_bc(int ci,const float*raw,double*maxabs,double*maxrel){double a=0,b=0;bool ok=check_bc(ci,raw,a,b);*maxabs=a;*maxrel=b;return ok;}
int hbc_check_rows(int ci,int blocks,const int*d,size_t pd,const uint64_t*sigma,size_t ps,const float*delta,size_t pde,const int*Q,size_t pq,const int*Q2,size_t pq2,const int*S,size_t pst,const int*ep,size_t pe,const int*diam,int next,int jia){try{return check_rows(ci,blocks,d,pd,sigma,ps,delta,pde,Q,pq,Q2,pq2,S,pst,ep,pe,diam,next,jia);}catch(...){return 0;}}
}
