#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdint>
struct Case{int rows,cols,cap;};
int main(){FILE*f=fopen("oracle_i32.bin","wb"),*q=fopen("extended_precision_i32.bin","wb");if(!f||!q)return 2;for(Case p:std::vector<Case>{{32,48,0},{32,48,1},{32,48,2},{65,97,100},{1080,1920,100}}){float dr=2.f/p.rows,di=2.f/p.cols;for(int r=0;r<p.rows;r++)for(int col=0;col<p.cols;col++){float cr=fmaf(float(r),dr,-1.5f),ci=fmaf(float(col),di,-1.f);float x=0,y=0;int count=0;while(count<p.cap){float xx=x*x,yy=y*y;float radius=xx+yy;if(radius>=4.f)break;float difference=xx-yy;float nx=difference+cr;float twice=x+x;y=fmaf(twice,y,ci);x=nx;count++;}fwrite(&count,4,1,f);std::complex<long double>z(0,0),c(cr,ci);int high=0;while(high<p.cap&&std::norm(z)<4.L){z=z*z+c;high++;}fwrite(&high,4,1,q);if(p.rows==32&&p.cap<=2&&count!=p.cap)return 3;}}fclose(f);fclose(q);return 0;}
