#include <vector>
#include <array>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cassert>
struct Case {int n,z,steps,kind;};
int main(){std::vector<float> pre{4};std::vector<double> gold;FILE*j=fopen("fixture_manifest.json","w");assert(j);fprintf(j,"{\"cases\":[");int id=0;
for(Case c:std::array<Case,4>{{{64,2,1,0},{64,3,2,1},{128,5,3,2},{512,8,5,1}}}){int n=c.n,z=c.z,N=n*n*z;float chip=.0005f,dx=.016f/n,dy=.016f/n,dz=chip/z;float cap=.5*1.75e6*chip*dx*dy,rx=dy/(2.0*100*chip*dx),ry=dx/(2.0*100*chip*dy),rz=dz/(100*dx*dy);float slope=3e6/(.5*chip*1.75e6),dt=.001/slope,step=dt/cap,ce=step/rx,cn=step/ry,ct=step/rz,cc=1.0-(2.0*ce+2.0*cn+3.0*ct);float q[]={ce,ce,cn,cn,ct,ct,cc,step};pre.push_back(n);pre.push_back(z);pre.push_back(c.steps);pre.insert(pre.end(),q,q+8);
std::vector<float> initial(N),power(N),unused(N,-321);for(int i=0;i<N;i++){initial[i]=c.kind==0?80.f:c.kind==2?81.f:80.f+float((i*7)%17-8)/8.f;power[i]=c.kind==0?0.f:c.kind==2?2.f:float((i*17)%31)/8.f;}
auto prefix=[&](const std::vector<float>&v){pre.insert(pre.end(),64,-777);pre.insert(pre.end(),v.begin(),v.end());pre.insert(pre.end(),64,-777);};prefix(initial);prefix(power);prefix(unused);
std::vector<double>a(initial.begin(),initial.end()),b(N);for(int it=0;it<c.steps;it++){for(int k=0;k<z;k++)for(int y=0;y<n;y++)for(int x=0;x<n;x++){int i=x+y*n+k*n*n;int w=x?i-1:i,e=x<n-1?i+1:i,nn=y?i-n:i,s=y<n-1?i+n:i,bot=k?i-n*n:i,t=k<z-1?i+n*n:i; b[i]=double(cc)*a[i]+double(ce)*(a[w]+a[e])+double(cn)*(a[nn]+a[s])+double(ct)*(a[bot]+a[t])+double(step)*power[i]+double(ct)*80.;}a.swap(b);}
auto append=[&](auto&v){gold.insert(gold.end(),64,-777);gold.insert(gold.end(),v.begin(),v.end());gold.insert(gold.end(),64,-777);};append(a);append(b);if(c.steps%2)append(a);else append(b);append(power);
fprintf(j,"%s{\"n\":%d,\"layers\":%d,\"steps\":%d,\"kind\":\"%s\",\"elements\":%d,\"selected_time_level\":%d,\"params\":[",id++?",":"",n,z,c.steps,c.kind==0?"ambient":c.kind==2?"uniform_power":"varying",N,c.steps%2?c.steps:c.steps-1);for(int k=0;k<8;k++)fprintf(j,"%s%.17g",k?",":"",q[k]);fprintf(j,"]}");}
fprintf(j,"],\"input_prefix_count\":%zu,\"output_count\":%zu,\"artifact_count\":%zu,\"atol\":0.0002,\"rtol\":0.000002,\"native_behavior_known_scientific_defect\":true,\"defect\":\"Even publicselectedbuffer is previous notfinal; sourceCPUhelperhasmatchingcallerbufferparity. Fullactualfinal/previousalsochecked independently.\",\"scope\":\"64x64x2/3,128x128x5,original512x512x8footprint;1/2/3/5steps not5000default;positiveclampedthermaloperator, source256threadcolumnrollingmapping\",\"tolerance_rationale\":\"ConservativeFloat32short-horizonpositiveweighteddotroundingbudget;sourceFMAandunfusedJulia, independentFloat64oracle\"}\n",pre.size(),gold.size(),pre.size()+gold.size());fclose(j);std::ofstream f("fixture.bin",std::ios::binary);f.write((char*)pre.data(),pre.size()*4);std::ofstream g("oracle.bin",std::ios::binary);g.write((char*)gold.data(),gold.size()*8);printf("fixture%zu oracle%zu\n",pre.size(),gold.size());}
