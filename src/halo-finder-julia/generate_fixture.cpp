// Named deterministic source-shaped sphere fixtures, not an alternative tree.
// Source generator: ForceTreeTest.cxx (LANL notices preserved in reference_history).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <fstream>
#include <string>
#include <cassert>
#include <omp.h>
int main(int argc,char**argv){assert(argc==2);std::string root=argv[1];const int counts[4]={10001,2,31,65};
 FILE*meta=fopen((root+"/fixture_metadata.json").c_str(),"w");assert(meta);fprintf(meta,"{\"cases\":[");
 for(int c=0;c<4;c++){
  int n=counts[c];float L=.5f,rSphere=.1f,center[3],target[3];srand48(12);for(int k=0;k<3;k++){target[k]=L*drand48();center[k]=L*drand48();}
  std::array<std::vector<float>,7>a;for(auto&v:a)v.resize(n,0);for(int k=0;k<3;k++)a[k][0]=target[k];for(int i=0;i<n;i++)a[6][i]=1;
  int p=1;while(p<n){float x=2.0*rSphere*(drand48()-.5),y=2.0*rSphere*(drand48()-.5),z=2.0*rSphere*(drand48()-.5);if(x*x+y*y+z*z>rSphere*rSphere)continue;a[0][p]=center[0]+x;a[1][p]=center[1]+y;a[2][p]=center[2]+z;p++;}
  std::ofstream input(root+"/case"+std::to_string(c)+"_input_Float32.bin",std::ios::binary);for(auto&v:a)input.write((char*)v.data(),v.size()*4);input.close();assert(input);
  std::vector<double>oracle(3*n);float soft2=.1f*.1f,cut2=3.116326355f*3.116326355f;
  #pragma omp parallel for num_threads(8)
  for(int i=0;i<n;i++){
   double acc[3]={0,0,0};for(int j=0;j<n;j++){double dx=double(a[0][j])-a[0][i],dy=double(a[1][j])-a[1][i],dz=double(a[2][j])-a[2][i],r2=dx*dx+dy*dy+dz*dz;
    if(r2>0&&r2<double(cut2)){double f=a[6][j]/std::pow(r2+double(soft2),1.5);acc[0]+=f*dx;acc[1]+=f*dy;acc[2]+=f*dz;}}
   for(int k=0;k<3;k++)oracle[k*n+i]=acc[k];
  }
  std::ofstream output(root+"/case"+std::to_string(c)+"_oracle_Float64.bin",std::ios::binary);output.write((char*)oracle.data(),oracle.size()*8);output.close();assert(output);
  double net[3]={0,0,0},torque[3]={0,0,0},scale=0;
  for(int i=0;i<n;i++){double x=a[0][i],y=a[1][i],z=a[2][i],ax=oracle[i],ay=oracle[n+i],az=oracle[2*n+i];net[0]+=ax;net[1]+=ay;net[2]+=az;torque[0]+=y*az-z*ay;torque[1]+=z*ax-x*az;torque[2]+=x*ay-y*ax;scale+=std::abs(ax)+std::abs(ay)+std::abs(az);}
  fprintf(meta,"%s{\"n\":%d,\"seed\":12,\"L\":%.17g,\"sphere_radius\":%.17g,\"theta\":0,\"softening_squared_source_Float32\":%.17g,\"cutoff_squared_source_Float32\":%.17g,\"independent_F64_pairwise_force_net\":[%.17g,%.17g,%.17g],\"independent_F64_torque_net\":[%.17g,%.17g,%.17g],\"sum_absolute_force_components\":%.17g}",c?",":"",n,double(L),double(rSphere),double(soft2),double(cut2),net[0],net[1],net[2],torque[0],torque[1],torque[2],scale);fflush(meta);
 }
 fprintf(meta,"],\"operator\":\"SourceGPUactualsoftenedcutoffgravity,directlimit(theta0),unitmass,fcoeff1; notdeclaredCLI_N_Newton_orS_grid-subtractedforce\",\"independent_oracle\":\"FullFloat64allpairswithsourceFloat32softening/cutoffconstants,distinctfromsourceCUDAmathandtree traversal\"}\n");fclose(meta);}
