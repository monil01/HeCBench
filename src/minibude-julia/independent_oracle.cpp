// Independent Float64 rotations/piecewise potentials; no source GPU body included.
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "study_params.h"
struct Atom {float x,y,z;int32_t type;};struct FF {int32_t hb;float radius,hphb,elsc;};
static_assert(sizeof(Atom)==16&&sizeof(FF)==16,"assetABI");
template<class T>std::vector<T>read(std::string f,int n){std::ifstream a(f,std::ios::binary);std::vector<T>v(n);a.read((char*)v.data(),n*sizeof(T));if(!a||a.peek()!=EOF){fprintf(stderr,"short/extra %s\n",f.c_str());exit(2);}return v;}
void save(std::string f,std::vector<double>&v){std::ofstream a(f,std::ios::binary);a.write((char*)v.data(),v.size()*8);if(!a)exit(3);}
void multiply(const double a[9],const double b[9],double c[9]){for(int i=0;i<3;i++)for(int j=0;j<3;j++){c[3*i+j]=0;for(int k=0;k<3;k++)c[3*i+j]+=a[3*i+k]*b[3*k+j];}}
int main(){for(int id=0;id<NCASES;id++){
 std::string root="case"+std::to_string(id);int n=NS[id],nl=NLIG[id],np=NPRO[id],nf=NFT[id];auto p=read<Atom>(root+"_protein.bin",np);auto l=read<Atom>(root+"_ligand.bin",nl);auto f=read<FF>(root+"_forcefield.bin",nf);auto t=read<float>(root+"_poses.bin",6*n);std::vector<double>out(6*n),xyz(3*nl*n);
 for(auto x:p)if(x.type<0||x.type>=nf)exit(4);for(auto x:l)if(x.type<0||x.type>=nf)exit(4);for(auto x:f)if(!(x.radius>0)||!std::isfinite(x.hphb)||!std::isfinite(x.elsc))exit(4);
 #pragma omp parallel for schedule(static)
 for(int pose=0;pose<n;pose++){
  double ax=t[pose],ay=t[pose+n],az=t[pose+2*n],cx=std::cos(ax),sx=std::sin(ax),cy=std::cos(ay),sy=std::sin(ay),cz=std::cos(az),sz=std::sin(az);
  double rx[9]={1,0,0,0,cx,-sx,0,sx,cx},ry[9]={cy,0,sy,0,1,0,-sy,0,cy},rz[9]={cz,-sz,0,sz,cz,0,0,0,1},tmp[9],rot[9];multiply(ry,rx,tmp);multiply(rz,tmp,rot);
  double steric=0,charge=0,solvation=0,abs_sum=0,margin=1e300;
  for(int i=0;i<nl;i++){
   double v[3]={l[i].x,l[i].y,l[i].z},q[3]={};for(int a=0;a<3;a++){q[a]=t[pose+(3+a)*n];for(int b=0;b<3;b++)q[a]+=rot[3*a+b]*v[b];xyz[(pose*nl+i)*3+a]=q[a];}const auto lf=f[l[i].type];
   for(int j=0;j<np;j++){
    const auto pf=f[p[j].type];double x=q[0]-p[j].x,y=q[1]-p[j].y,z=q[2]-p[j].z;double d=std::sqrt(x*x+y*y+z*z),radius=double(lf.radius)+pf.radius,surface=d-radius;
    double s=76.*std::max(0.,1.-d/radius),cutoff=lf.hb==70&&pf.hb==70?4.:2.,cf=surface<0?1.:std::max(0.,1.-surface/cutoff);double e=double(lf.elsc)*pf.elsc*cf;if(lf.hb==69||pf.hb==69)e=-std::abs(e);e*=45.;double h=0.;
    if(pf.hphb!=0&&(lf.hphb<0||pf.hphb<0)){double length=lf.hphb<0&&pf.hphb<0?5.5:1.,hf=surface<0?1.:std::max(0.,1.-surface/length),ph=pf.hphb,lh=lf.hphb;if(ph<0&&lh>0)ph=-ph;if(pf.hphb>0&&lf.hphb<0)lh=-lh;h=(ph+lh)*hf;margin=std::min(margin,std::abs(surface-length));}
    margin=std::min(margin,std::min(std::abs(surface),std::abs(surface-cutoff)));steric+=s;charge+=e;solvation+=h;abs_sum+=std::abs(s)+std::abs(e)+std::abs(h);
   }
  }
  out[6*pose]=(steric+charge+solvation)*.5;out[6*pose+1]=steric*.5;out[6*pose+2]=charge*.5;out[6*pose+3]=solvation*.5;out[6*pose+4]=abs_sum*.5;out[6*pose+5]=margin;
 }
 save(root+"_oracle_channels_f64.bin",out);save(root+"_oracle_transformed_xyz_f64.bin",xyz);printf("case%d poses%d completepairs%lld channels6 alltransformedligandcoords%zu\n",id,n,(long long)n*nl*np,xyz.size());fflush(stdout);
 }return 0;}
