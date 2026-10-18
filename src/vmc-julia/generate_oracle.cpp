#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstdio>
#include <algorithm>
using V=std::vector<double>;constexpr int N=57344,B=224,G=64;double gap=1;
uint32_t next(uint32_t&s){return s=(26757677u*s+1u)&0x7fffffff;}
double rnd(uint32_t&s){return double(float(next(s)))/2147483648.;}
struct Chain{double x[6],p;uint32_t s;double stat[4];};
void distances(const double*x,double*r){r[0]=std::hypot(x[0],x[1],x[2]);r[1]=std::hypot(x[3],x[4],x[5]);r[2]=std::hypot(x[0]-x[3],x[1]-x[4],x[2]-x[5]);}
double psi(const double*x){double a[3];distances(x,a);return (1+a[2]/2)*std::exp(-2*(a[0]+a[1]));}
void prop(Chain&c,int steps){for(int st=0;st<steps;st++){double y[6];for(int j=0;j<6;j++)y[j]=c.x[j]+(rnd(c.s)-.5)*2;double p=psi(y),u=rnd(c.s),lhs=p*p,rhs=c.p*c.p*u;gap=std::min(gap,std::abs(lhs-rhs)/std::max(lhs,rhs));if(lhs>rhs){std::copy(y,y+6,c.x);c.p=p;c.stat[3]++;}double a[3];distances(c.x,a);for(int j=0;j<3;j++)c.stat[j]+=a[j];}}
void append(V&out,V a){out.insert(out.end(),G,16384);out.insert(out.end(),a.begin(),a.end());out.insert(out.end(),G,16384);}
void capture(V&out,const std::vector<Chain>&c,bool withstats){for(int j=0;j<7;j++){V a;for(auto&x:c)a.push_back(j<6?x.x[j]:x.p);append(out,a);}if(withstats){V a;for(int j=0;j<4;j++)for(auto&x:c)a.push_back(x.stat[j]);append(out,a);}V seeds;for(auto&x:c)seeds.push_back(x.s);append(out,seeds);}
int main(int argc,char**argv){if(argc!=2)return 1;V pre{2,5551212,2,3,2,5551213,5,7,2},out;std::vector<double>mingaps;for(int ci=0;ci<2;ci++){int seed=5551212+ci,burn=ci?5:2,steps=ci?7:3;std::vector<Chain> c(N);for(int i=0;i<N;i++){auto&x=c[i];x.s=seed^i;next(x.s);for(int j=0;j<6;j++)x.x[j]=(rnd(x.s)-.5)*4;x.p=psi(x.x);std::fill(x.stat,x.stat+4,0);}capture(out,c,false);gap=1;for(auto&x:c)prop(x,burn);capture(out,c,true);for(int sample=0;sample<2;sample++){for(auto&x:c){std::fill(x.stat,x.stat+4,0);prop(x,steps);}capture(out,c,true);V sums(4),partial(B);for(int j=0;j<4;j++){V ps(B);for(int block=0;block<B;block++)for(int lane=0;lane<256;lane++)ps[block]+=c[block*256+lane].stat[j];for(double x:ps)sums[j]+=x;if(j==3)partial=ps;}append(out,sums);append(out,partial);}mingaps.push_back(gap);}
 auto write=[&](const char*name,V&v){std::ofstream f(std::string(argv[1])+"/"+name,std::ios::binary);f.write((char*)v.data(),v.size()*8);if(!f)throw "writefailed";};write("fixture.bin",pre);write("oracle.bin",out);printf("{\"input_prefix_count\":%zu,\"oracle_count\":%zu,\"artifact_count\":%zu,\"walkers\":57344,\"blocks\":224,\"threads\":256,\"samples_per_case\":2,\"burn_steps\":[2,5],\"sample_steps\":[3,7],\"minimum_independent_double_normalized_acceptance_gaps\":[%.17g,%.17g],\"oracle\":\"IndependentFloat64hypotdistances+analyticwavefunction+mod2power31LCG withsourceFloat32randomcast; perwalkerMetropolis+mathematicalgroupedsums, notsourceGPUtree. Namedshortchains notequilibrium/energyaccuracy.\",\"exact_guard\":16384,\"guards_eachside\":64,\"artifact_encoding\":\"Float64exactUInt32/F32representable\"}\n",pre.size(),out.size(),pre.size()+out.size(),mingaps[0],mingaps[1]);}
