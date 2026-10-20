// Independent CPU radial formulation, support-neighbor search and fixed j order.
// Does not include/call original CUDA device helpers or execute GPU work.
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <omp.h>
struct V{double x,y,z;};struct F{double density,pressure;V pos,v,half,a;};struct B{V pos,n;};
struct P{double rho,mass,spacing,h,g,dt,alpha,sigma,cs;int np,nf,nb,steps,spf;};
static_assert(sizeof(F)==112&&sizeof(B)==48&&sizeof(P)==96,"source ABI");
V sub(V a,V b){return{a.x-b.x,a.y-b.y,a.z-b.z};}V add(V a,V b){return{a.x+b.x,a.y+b.y,a.z+b.z};}V scale(double t,V a){return{t*a.x,t*a.y,t*a.z};}double dot(V a,V b){return(a.x*b.x+a.y*b.y)+a.z*b.z;}
struct Cell{int x,y,z;bool operator==(const Cell&b)const{return x==b.x&&y==b.y&&z==b.z;}};
struct Hash{size_t operator()(const Cell&a)const{return(size_t(uint32_t(a.x))*73856093u)^(size_t(uint32_t(a.y))*19349663u)^(size_t(uint32_t(a.z))*83492791u);}};
using Grid=std::unordered_map<Cell,std::vector<int>,Hash>;
Cell cell(V p,double size){return{int(std::floor(p.x/size)),int(std::floor(p.y/size)),int(std::floor(p.z/size))};}
std::vector<int>neighbors(const Grid&grid,V p,double size){auto c=cell(p,size);std::vector<int>ids;for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++){auto it=grid.find({c.x+x,c.y+y,c.z+z});if(it!=grid.end())ids.insert(ids.end(),it->second.begin(),it->second.end());}std::sort(ids.begin(),ids.end());return ids;}
struct Radial{double w,doverr;};
Radial radial(double r,double h){double q=r/h;if(q>=2)return{0,0};double C=1/(3.141592653589793238462643383279502884*h*h*h);if(q<1)return{C*(1-1.5*q*q+.75*q*q*q),-C/(h*h)*(3-2.25*q)};double t=2-q;return{C*.25*t*t*t,-C*.75*t*t/(h*r)};}
double gamma(V d,V n,const P&p){double y2=dot({d.x*d.x,d.y*d.y,d.z*d.z},{n.x*n.x,n.y*n.y,n.z*n.z});if(y2>4*p.h*p.h)return 0;double y=std::sqrt(y2),r=std::sqrt(dot(d,d)),x=r-y,q=y/p.h;double xi=(1-x/p.h)!=0?double(x<p.h):0.;double C=xi*.04*p.cs*p.cs/y;double v=0;if(q>0&&q<2./3.)v=2./3.;else if(q>2./3.&&q<1)v=2*q-1.5*q*q;else if(q>1&&q<2)v=.5*(2-q)*(2-q);return v*C;}
template<class T>std::vector<T>load(std::string f,size_t n){std::vector<T>a(n);std::ifstream s(f,std::ios::binary);s.read((char*)a.data(),n*sizeof(T));assert(s);return a;}
void save(std::string file,const std::vector<F>&a){std::ofstream s(file,std::ios::binary);s.write((char*)a.data(),a.size()*sizeof(F));assert(s);}
int main(int argc,char**argv){assert(argc==3);std::string root=argv[1],name=argv[2];P p=load<P>(root+"/"+name+"_params_raw.bin",1)[0];auto f=load<F>(root+"/"+name+"_initial_f64.bin",p.nf);auto b=load<B>(root+"/"+name+"_boundary_f64.bin",p.nb);omp_set_num_threads(8);std::vector<F>trace;std::vector<int>check=name=="default"?std::vector<int>{1,2,10,50,100,250,500}:std::vector<int>{1,2,17};
for(int step=1;step<=p.steps;step++){Grid grid;for(int i=0;i<p.nf;i++)grid[cell(f[i].pos,2*p.h)].push_back(i);std::vector<std::vector<int>>near(p.nf);for(int i=0;i<p.nf;i++)near[i]=neighbors(grid,f[i].pos,2*p.h);
#pragma omp parallel for schedule(static)
for(int i=0;i<p.nf;i++){double delta=0;for(int j:near[i]){V d=sub(f[i].pos,f[j].pos);double r=std::sqrt(dot(d,d));auto k=radial(r,p.h);delta+=p.mass*p.dt*k.doverr*dot(sub(f[i].v,f[j].v),d);}f[i].density+=delta;double ratio=f[i].density/p.rho;f[i].pressure=(p.rho*p.cs*p.cs/7)*(std::pow(ratio,7.)-1);}
#pragma omp parallel for schedule(static)
for(int i=0;i<p.nf;i++){V a{0,0,-9.8};for(int j:near[i])if(j!=i){V d=sub(f[i].pos,f[j].pos);double r2=dot(d,d);auto k=radial(std::sqrt(r2),p.h);double coeff=-p.mass*(f[i].pressure/(f[i].density*f[i].density)+f[j].pressure/(f[j].density*f[j].density))*k.doverr;double vr=dot(sub(f[i].v,f[j].v),d);if(vr<0)coeff+=p.mass*(2*p.alpha*p.h*p.cs/(f[i].density+f[j].density))*vr/(r2+(p.h/10)*p.h*p.h)*k.doverr;coeff+=p.sigma*k.w;a=add(a,scale(coeff,d));}for(int j=0;j<p.nb;j++)a=add(a,scale(gamma(sub(f[i].pos,b[j].pos),b[j].n,p),b[j].n));f[i].a=a;}
#pragma omp parallel for schedule(static)
for(int i=0;i<p.nf;i++){f[i].half=add(f[i].half,scale(p.dt,f[i].a));f[i].v=add(f[i].half,scale(p.dt/2,f[i].a));f[i].pos=add(f[i].pos,scale(p.dt,f[i].half));}
for(auto&x:f){auto*d=(double*)&x;assert(x.density>0);for(int j=0;j<14;j++)assert(std::isfinite(d[j]));}
if(std::find(check.begin(),check.end(),step)!=check.end()){trace.insert(trace.end(),f.begin(),f.end());printf("CPU_CHECKPOINT %s %d\n",name.c_str(),step);fflush(stdout);}}
save(root+"/"+name+"_CPU_final_f64.bin",f);save(root+"/"+name+"_CPU_checkpoints_f64.bin",trace);printf("CPU_COMPLETE %s steps%d n%d\n",name.c_str(),p.steps,p.nf);}
