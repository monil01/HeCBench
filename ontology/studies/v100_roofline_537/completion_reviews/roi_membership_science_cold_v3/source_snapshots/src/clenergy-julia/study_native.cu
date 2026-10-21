// Named kernel-only validation. Original kernel is included unchanged.
#define main original_hecbench_main
#include "../clenergy-cuda/clenergy.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <stdexcept>
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static constexpr int W=64,H=8,P=W*H;
static constexpr float GUARD=1024.25f;
static constexpr double ATOL=2e-4,RTOL=5e-5;
static const int COUNTS[]={1,2,127,4001,64};
static std::vector<float> atoms(int which){int n=COUNTS[which];std::vector<float>a(4*n);for(int i=0;i<n;i++){
 a[4*i]=float((17*i)%97)/16.f;a[4*i+1]=float((11*i)%29)/32.f;a[4*i+2]=.25f+float(i%13)/32.f;a[4*i+3]=float((i%7)+1)/8.f*(i%2?-1.f:1.f);
 if(which==0)a[4*i+3]=1.f;
 if(which==1){a[4*i]=0;a[4*i+1]=0;a[4*i+2]=.5f;a[4*i+3]=(i==0?1.f:-1.f);}
 if(which==4)a[4*i+3]=0.f;
 }return a;}
static std::vector<double> oracle(const std::vector<float>&a){std::vector<double>o(P);for(int y=0;y<H;y++)for(int x=0;x<W;x++){
 double xx=double(float(x)*.1f),yy=double(float(y)*.1f),sum=0;
 for(size_t i=0;i<a.size();i+=4){double dx=xx-a[i],dy=yy-a[i+1],z=a[i+2];sum+=double(a[i+3])/std::sqrt(dx*dx+dy*dy+z*z);}o[y*W+x]=sum;}return o;}
int main(){try{
 float*raw;float4*di;checked(cudaMalloc(&raw,(P+2)*sizeof(float)));checked(cudaMalloc(&di,MAXATOMS*sizeof(float4)));
 std::vector<float>init(P+2,0.f);init[0]=init[P+1]=GUARD;std::vector<float>actual(P+2);std::vector<std::vector<float>>aa;std::vector<std::vector<double>>ref;
 for(int k=0;k<5;k++){aa.push_back(atoms(k));ref.push_back(oracle(aa.back()));}
 checked(cudaMemcpy(raw,init.data(),init.size()*4,cudaMemcpyHostToDevice));if(copyatoms(aa[0].data(),1,0.f,di))return 3;cenergy<<<dim3(1,1),dim3(8,8)>>>(1,.1f,raw+1,di);checked(cudaDeviceSynchronize());
 bool allok=true;for(int trial=1;trial<=5;trial++){
  double seconds=0,maxabs=0,maxrel=0;bool ok=true;std::vector<float>prefix,outputs;
  for(int k=0;k<5;k++){
   checked(cudaMemcpy(raw,init.data(),init.size()*4,cudaMemcpyHostToDevice));checked(cudaMemcpy(actual.data(),raw,init.size()*4,cudaMemcpyDeviceToHost));
   ok &= std::memcmp(init.data(),actual.data(),init.size()*4)==0;prefix.insert(prefix.end(),actual.begin()+1,actual.end()-1);
   for(int start=0;start<COUNTS[k];start+=MAXATOMS){int count=std::min(MAXATOMS,COUNTS[k]-start);if(copyatoms(aa[k].data()+4*start,count,0.f,di))return 4;
    std::vector<float>pre(4*count);checked(cudaMemcpy(pre.data(),di,pre.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<count;i++){float dz=0.f-aa[k][4*(i+start)+2];float want[4]={aa[k][4*(i+start)],aa[k][4*(i+start)+1],dz*dz,aa[k][4*(i+start)+3]};ok &= std::memcmp(want,pre.data()+4*i,16)==0;}
    prefix.insert(prefix.end(),pre.begin(),pre.end());checked(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();
    cenergy<<<dim3(1,1),dim3(8,8)>>>(count,.1f,raw+1,di);checked(cudaDeviceSynchronize());seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
   }
   checked(cudaMemcpy(actual.data(),raw,init.size()*4,cudaMemcpyDeviceToHost));ok &= actual[0]==GUARD&&actual[P+1]==GUARD;prefix.push_back(actual[0]);prefix.push_back(actual[P+1]);
   for(int j=0;j<P;j++){double got=actual[j+1],err=std::fabs(got-ref[k][j]);maxabs=std::max(maxabs,err);maxrel=std::max(maxrel,err/std::max(std::fabs(ref[k][j]),ATOL));ok &= std::isfinite(got)&&err<=ATOL+RTOL*std::fabs(ref[k][j]);if(k==1||k==4)ok &= got==0.;outputs.push_back(float(got));}
  }
  allok &=ok;std::vector<float>artifact=prefix;artifact.insert(artifact.end(),outputs.begin(),outputs.end());char path[80];std::snprintf(path,sizeof(path),"output_trial%d.bin",trial);FILE*f=std::fopen(path,"wb");if(!f||std::fwrite(artifact.data(),4,artifact.size(),f)!=artifact.size())return 5;std::fclose(f);
  std::printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_six_kernel_launch_completion_fragments_five_fixtures\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",artifact.size(),maxabs,maxrel,path,artifact.size());
 }checked(cudaFree(raw));checked(cudaFree(di));return allok?0:1;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
