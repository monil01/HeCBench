#include "corrected_native/find_ellipse.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
int main(int argc,char**argv){
 if(argc>1&&std::strcmp(argv[1],"affine")==0){MAT*m=m_get(3,4);VEC*x=v_get(3),*y=v_get(3);double xs[]={.25,1.5,2.75},ys[]={.25,1.5,.75};for(int j=0;j<3;j++)for(int i=0;i<4;i++)m_set_val(m,j,i,2.+3.*i-4.*j);for(int k=0;k<3;k++){v_set_val(x,k,xs[k]);v_set_val(y,k,ys[k]);}MAT*v=linear_interp2(m,x,y);double ma=0;for(int k=0;k<3;k++)ma=std::fmax(ma,std::fabs(m_get_val(v,0,k)-(2+3*xs[k]-4*ys[k])));printf("affine_max_abs_error %.17g\n",ma);return ma<1e-12?0:1;}
 if(argc>1&&std::strcmp(argv[1],"invalid")==0){MAT*m=m_get(3,4);VEC*x=v_get(1),*y=v_get(1);v_set_val(x,0,.25);v_set_val(y,0,2.25);linear_interp2(m,x,y);return 99;}
 MAT*A=TMatrix(9,4);std::vector<double>B(36*9,0);for(int i=0;i<9;i++)for(int j=0;j<4;j++){double s=double(j)/4;int cols[]={(i+8)%9,i,(i+1)%9,(i+2)%9};double val[]={(-s*s*s+3*s*s-3*s+1)/6,(3*s*s*s-6*s*s+4)/6,(-3*s*s*s+3*s*s+3*s+1)/6,s*s*s/6};for(int k=0;k<4;k++)B[(i*4+j)*9+cols[k]]=val[k];}double err=0;for(int i=0;i<9;i++)for(int j=0;j<9;j++){double v=0;for(int k=0;k<36;k++)v+=m_get_val(A,i,k)*B[k*9+j];err=std::fmax(err,std::fabs(v-(i==j?1.:0.)));}if(argc==2){FILE*f=std::fopen(argv[1],"wb");if(!f)return 2;for(int i=0;i<9;i++)for(int j=0;j<36;j++){double x=m_get_val(A,i,j);std::fwrite(&x,8,1,f);}std::fclose(f);}printf("TMatrix_left_inverse_max_abs_residual %.17g\n",err);m_free(A);return err<1e-12?0:1;
}
