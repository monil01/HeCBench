// Independent tensor Euler face operator and Float64 pseudo-time integration.
// No CUDA source, vector-flux helper or GPU algebra is reused in the oracle.
#include <array>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <algorithm>
using State=std::array<double,5>;using Vec=std::array<double,3>;using Tensor=std::array<State,3>;
const double gamma_source=double(1.4f);const int steps=2000;
struct Primitive{double rho,pressure,speed,sound;Vec velocity;Tensor F;};
Primitive physical(const State&u){Primitive p;p.rho=u[0];assert(p.rho>0);double vsq=0;for(int d=0;d<3;d++){p.velocity[d]=u[d+1]/p.rho;vsq+=p.velocity[d]*p.velocity[d];}p.pressure=(gamma_source-1)*(u[4]-.5*p.rho*vsq);assert(p.pressure>0&&std::isfinite(p.pressure));p.speed=std::sqrt(vsq);p.sound=std::sqrt(gamma_source*p.pressure/p.rho);for(int d=0;d<3;d++){p.F[d][0]=u[d+1];for(int k=0;k<3;k++)p.F[d][k+1]=u[k+1]*p.velocity[d]+(k==d?p.pressure:0);p.F[d][4]=(u[4]+p.pressure)*p.velocity[d];}return p;}
State interior(const State&ui,const State&uj,const Primitive&pi,const Primitive&pj,const Vec&normal){State r{};double norm=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);double diss=-norm*double(0.2f)*.5*(pi.speed+pj.speed+pi.sound+pj.sound);for(int k=0;k<5;k++){r[k]=diss*(ui[k]-uj[k]);for(int d=0;d<3;d++)r[k]+=.5*normal[d]*(pi.F[d][k]+pj.F[d][k]);}return r;}
struct Mesh{int n;std::vector<double>areas;std::vector<int>nb;std::vector<Vec>normals;Mesh(const char*name){std::ifstream f(name);assert(f);f>>n;assert(n%192==0);areas.resize(n);nb.resize(n*4);normals.resize(n*4);for(int i=0;i<n;i++){float a;f>>a;areas[i]=a;assert(a>0);for(int j=0;j<4;j++){int q;f>>q;nb[i*4+j]=q<0?-2:q-1;assert(nb[i*4+j]<n);for(int d=0;d<3;d++){float x;f>>x;normals[i*4+j][d]=-x;}}}assert(f);std::string extra;assert(!(f>>extra));}};
void serialize_state(std::vector<double>&out,const std::vector<State>&u){for(int k=0;k<5;k++)for(const State&q:u)out.push_back(q[k]);}
std::vector<double>farfield(){float rho=float(1.4),pressure=1.f,sound=std::sqrt(1.4f*pressure/rho),speed=1.2f*sound;float vx=speed,vy=0,vz=0;float mx=rho*vx,my=rho*vy,mz=rho*vz;float energy=rho*(.5f*(speed*speed))+(pressure/float(1.4f-1.f));float xx=vx*mx+pressure,xy=vx*my,xz=vx*mz,yy=vy*my+pressure,yz=vy*mz,zz=vz*mz+pressure;float de=energy+pressure;return {rho,mx,my,mz,energy,xx,xy,xz,xy,yy,yz,xz,yz,zz,vx*de,vy*de,vz*de};}
int main(){std::vector<double>fixture{3},oracle;auto ff=farfield();std::ofstream summary("oracle_summary.json");assert(summary);summary<<"{\"gamma_source_Float32_expanded\":"<<std::scientific<<gamma_source<<",\"full_iterations\":2000,\"cases\":[";
 const char*paths[]={"freestream192.mesh","cropped192.mesh","cropped384.mesh"};for(int ci=0;ci<3;ci++){Mesh m(paths[ci]);int N=m.n;fixture.insert(fixture.end(),{double(N),2000,double(ci)});fixture.insert(fixture.end(),m.areas.begin(),m.areas.end());for(int j=0;j<4;j++)for(int i=0;i<N;i++)fixture.push_back(m.nb[i*4+j]);for(int d=0;d<3;d++)for(int j=0;j<4;j++)for(int i=0;i<N;i++)fixture.push_back(m.normals[i*4+j][d]);fixture.insert(fixture.end(),ff.begin(),ff.end());
 State ufar;std::copy_n(ff.begin(),5,ufar.begin());Tensor boundary{};for(int d=0;d<3;d++){boundary[d][0]=ff[1+d];for(int k=0;k<3;k++)boundary[d][k+1]=ff[5+k*3+d];boundary[d][4]=ff[14+d];}
 std::vector<State>u(N,ufar),old=u,flux=u;std::vector<double>dt(N,0);for(int q=0;q<3;q++)serialize_state(oracle,u);oracle.insert(oracle.end(),dt.begin(),dt.end());double minrho=1e300,minp=1e300,maxsym=0,freestream=0;
 for(int iteration=0;iteration<steps;iteration++){old=u;for(int i=0;i<N;i++){auto p=physical(u[i]);dt[i]=.5/(std::sqrt(m.areas[i])*(p.speed+p.sound));}
  for(int stage=0;stage<3;stage++){std::vector<Primitive>p;p.reserve(N);for(const auto&z:u){auto q=physical(z);minrho=std::min(minrho,q.rho);minp=std::min(minp,q.pressure);p.push_back(q);}for(int i=0;i<N;i++){flux[i]=State{};for(int face=0;face<4;face++){int j=m.nb[i*4+face];auto normal=m.normals[i*4+face];if(j>=0){auto v=interior(u[i],u[j],p[i],p[j],normal);for(int k=0;k<5;k++)flux[i][k]+=v[k];}else if(j==-1){for(int d=0;d<3;d++)flux[i][d+1]+=normal[d]*p[i].pressure;}else{for(int k=0;k<5;k++)for(int d=0;d<3;d++)flux[i][k]+=.5*normal[d]*(p[i].F[d][k]+boundary[d][k]);}}}
   for(int i=0;i<N;i++)for(int k=0;k<5;k++)u[i][k]=old[i][k]+dt[i]/double(4-stage)*flux[i][k];
  }
  if((iteration+1)%100==0)serialize_state(oracle,u);
 }
 for(const auto&z:u){auto q=physical(z);minrho=std::min(minrho,q.rho);minp=std::min(minp,q.pressure);for(int k=0;k<5;k++)freestream=std::max(freestream,std::abs(z[k]-ufar[k]));}
 for(int i=0;i<N;i++)for(int face=0;face<4;face++){int j=m.nb[i*4+face];if(j>=0){auto n=m.normals[i*4+face],back=n;for(double&z:back)z=-z;auto a=interior(u[i],u[j],physical(u[i]),physical(u[j]),n),b=interior(u[j],u[i],physical(u[j]),physical(u[i]),back);for(int k=0;k<5;k++)maxsym=std::max(maxsym,std::abs(a[k]+b[k]));}}
 assert(maxsym<1e-10);if(ci==0)assert(freestream<1e-11);serialize_state(oracle,u);serialize_state(oracle,old);serialize_state(oracle,flux);oracle.insert(oracle.end(),dt.begin(),dt.end());
 char row[1000];snprintf(row,sizeof(row),"%s{\"case\":%d,\"cells\":%d,\"min_density_all_reference_stages\":%.17g,\"min_pressure_all_reference_stages\":%.17g,\"final_pair_flux_antisymmetry_max_abs\":%.17g,\"final_departure_from_uniform_FF\":%.17g}",ci?",":"",ci,N,minrho,minp,maxsym,freestream);summary<<row;
 }summary<<"],\"fixture_words\":"<<fixture.size()<<",\"oracle_words\":"<<oracle.size()<<",\"scope\":\"IndependentFloat64tensoroperator2000localpseudotimesteps,sourceweights1/(4-j);no globalconservedmassclaimwithcell-varyingstepcoefficients,no physicaltime/aerodynamicsvalidation\"}\n";std::ofstream f("fixture.bin",std::ios::binary),o("oracle.bin",std::ios::binary);f.write((char*)fixture.data(),fixture.size()*8);o.write((char*)oracle.data(),oracle.size()*8);assert(f&&o);return 0;}
