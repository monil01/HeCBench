// Independent CPU Float64 scalar mathematical optical-flow implementation.
// No CUDA, source solver, texture library, or Julia code is included.
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <iomanip>
using V=std::vector<double>;
struct Image{int w,h;V v;Image(int w_,int h_):w(w_),h(h_),v(size_t(w)*h){}double&operator()(int x,int y){return v[size_t(y)*w+x];}double operator()(int x,int y)const{return v[size_t(y)*w+x];}};
int mirror(int x,int n){x%=2*n;if(x<0)x+=2*n;return x>=n?2*n-1-x:x;}
double sample(const Image&i,double x,double y){int a=int(std::floor(x)),b=int(std::floor(y));double dx=x-a,dy=y-b;double lo=(1-dx)*i(mirror(a,i.w),mirror(b,i.h))+dx*i(mirror(a+1,i.w),mirror(b,i.h));double hi=(1-dx)*i(mirror(a,i.w),mirror(b+1,i.h))+dx*i(mirror(a+1,i.w),mirror(b+1,i.h));return(1-dy)*lo+dy*hi;}
Image down(const Image&i){Image o(i.w/2,i.h/2);for(int y=0;y<o.h;y++)for(int x=0;x<o.w;x++){double cx=(x+.5)/o.w*i.w-.5,cy=(y+.5)/o.h*i.h-.5,dx=.25/o.w*i.w,dy=.25/o.h*i.h;o(x,y)=.25*(sample(i,cx-dx,cy)+sample(i,cx+dx,cy)+sample(i,cx,cy-dy)+sample(i,cx,cy+dy));}return o;}
Image up(const Image&i,int w,int h,double scale){Image o(w,h);for(int y=0;y<h;y++)for(int x=0;x<w;x++)o(x,y)=sample(i,(x+.5)/w*i.w-.5,(y+.5)/h*i.h-.5)*scale;return o;}
Image warp(const Image&i,const Image&u,const Image&v){Image o(i.w,i.h);for(int y=0;y<i.h;y++)for(int x=0;x<i.w;x++)o(x,y)=sample(i,x+u(x,y),y+v(x,y));return o;}
void derivatives(const Image&a,const Image&b,Image&gx,Image&gy,Image&gt){for(int y=0;y<a.h;y++)for(int x=0;x<a.w;x++){auto dx=[&](const Image&i){return(sample(i,x-2,y)-8*sample(i,x-1,y)+8*sample(i,x+1,y)-sample(i,x+2,y))/12.;};auto dy=[&](const Image&i){return(sample(i,x,y-2)-8*sample(i,x,y-1)+8*sample(i,x,y+1)-sample(i,x,y+2))/12.;};gx(x,y)=.5*(dx(a)+dx(b));gy(x,y)=.5*(dy(a)+dy(b));gt(x,y)=b(x,y)-a(x,y);}}
void jacobi(Image&du,Image&dv,const Image&gx,const Image&gy,const Image&gt,double alpha,int n){Image nu(du.w,du.h),nv(du.w,du.h);for(int iteration=0;iteration<n;iteration++){for(int y=0;y<du.h;y++)for(int x=0;x<du.w;x++){int xm=std::max(0,x-1),xp=std::min(du.w-1,x+1),ym=std::max(0,y-1),yp=std::min(du.h-1,y+1);double a=.25*(du(xm,y)+du(xp,y)+du(x,yp)+du(x,ym));double b=.25*(dv(xm,y)+dv(xp,y)+dv(x,yp)+dv(x,ym));double rx=gx(x,y),ry=gy(x,y),rt=gt(x,y);double t=(rx*a+ry*b+rt)/(rx*rx+ry*ry+alpha);nu(x,y)=a-rx*t;nv(x,y)=b-ry*t;}std::swap(du.v,nu.v);std::swap(dv.v,nv.v);}}
struct Config{int w,h,s,levels,warps,solver;double alpha;Image a,b;Config(int w_,int h_):w(w_),h(h_),a(w_,h_),b(w_,h_){} };
Config load(const char*path){std::ifstream f(path,std::ios::binary);uint32_t h[8];float alpha;f.read((char*)h,32);f.read((char*)&alpha,4);if(!f||h[0]!=0x4f463537)throw std::runtime_error("bad fixture");Config c(h[1],h[2]);c.s=h[3];c.levels=h[4];c.warps=h[5];c.solver=h[6];c.alpha=alpha;if(h[7]!=2*c.s*c.h)throw std::runtime_error("bad fixture size");for(auto*i:{&c.a,&c.b})for(int y=0;y<c.h;y++)for(int x=0;x<c.s;x++){float z;f.read((char*)&z,4);if(!f)throw std::runtime_error("short fixture");if(x<c.w)(*i)(x,y)=z;else if(z!=0)throw std::runtime_error("named padding must be zero");}if(f.peek()!=EOF)throw std::runtime_error("trailing fixture");return c;}
std::pair<Image,Image>flow(const Config&c){std::vector<Image>a,b;a.push_back(c.a);b.push_back(c.b);for(int l=1;l<c.levels;l++){a.push_back(down(a.back()));b.push_back(down(b.back()));}Image u(a.back().w,a.back().h),v(u.w,u.h);for(int l=c.levels-1;l>=0;l--){if(u.w!=a[l].w){u=up(u,a[l].w,a[l].h,double(a[l].w)/u.w);v=up(v,a[l].w,a[l].h,double(a[l].h)/v.h);}for(int w=0;w<c.warps;w++){Image moved=warp(b[l],u,v),gx(u.w,u.h),gy(u.w,u.h),gt(u.w,u.h);derivatives(a[l],moved,gx,gy,gt);Image du(u.w,u.h),dv(u.w,u.h);jacobi(du,dv,gx,gy,gt,c.alpha,c.solver);for(size_t i=0;i<u.v.size();i++){u.v[i]+=du.v[i];v.v[i]+=dv.v[i];}}}return{u,v};}
void save(const char*path,const Image&u,const Image&v){std::ofstream f(path,std::ios::binary);uint32_t h[]={uint32_t(u.w),uint32_t(u.h)};f.write((char*)h,8);for(const Image*i:{&u,&v})f.write((char*)i->v.data(),i->v.size()*8);if(!f)throw std::runtime_error("oracle write failed");}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("fixture output");auto c=load(argv[1]);auto result=flow(c);save(argv[2],result.first,result.second);double mx=0,sum=0;for(auto*i:{&result.first,&result.second})for(double z:i->v){if(!std::isfinite(z))throw std::runtime_error("nonfinite reference");mx=std::max(mx,std::abs(z));sum+=z;}std::cout<<std::setprecision(17)<<"{\"w\":"<<c.w<<",\"h\":"<<c.h<<",\"max_abs\":"<<mx<<",\"sum\":"<<sum<<",\"precision\":\"Float64\",\"independent\":true}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
