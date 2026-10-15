// Validation only: affine planes are reproduced exactly by bilinear sampling.
#include "find_ellipse.h"
#include <cstdio>
#include <cmath>
int main(int argc,char**){
 MAT*m=m_get(3,4);VEC*x=v_get(3),*y=v_get(3);
 for(int j=0;j<3;++j)for(int i=0;i<4;++i)m_set_val(m,j,i,2.+3.*i-4.*j);
 double xs[]={.25,1.5,2.75},ys[]={.25,1.5,.75};
 for(int k=0;k<3;++k){v_set_val(x,k,xs[k]);v_set_val(y,k,ys[k]);}
 if(argc>1)v_set_val(x,0,4.); // Must fail, never clip or fabricate a value.
 MAT*r=linear_interp2(m,x,y);
 for(int k=0;k<3;++k)if(std::abs(m_get_val(r,0,k)-(2.+3.*xs[k]-4.*ys[k]))>1e-12)return 1;
 m_free(r);v_free(x);v_free(y);m_free(m);
 puts("PASS independent affine-plane interpolation oracle; no clipping/tolerance change");
}
