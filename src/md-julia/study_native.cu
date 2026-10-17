#define main archived_md_main
#include "../md-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
struct StudyCase{std::vector<float4>p;std::vector<int>neighbors;std::vector<double>gold;std::vector<float>source;float4*dp,*df;int*dn;};
int main(){const int n=129,repeats=20;std::vector<StudyCase>cs;std::vector<float>prefix;
 for(int mode=0;mode<3;mode++){StudyCase c;c.p.resize(n);c.neighbors.resize(n*128);c.gold.resize(3*n);c.source.resize(3*n);
  for(int i=0;i<n;i++){float x=(i%7)*1.25f,y=((i/7)%7)*1.25f,z=(i/49)*1.25f;if(i==1)x=y=z=0;if(i==2){x=3.f;y=1.5f;z=1.5f;}if(mode==1){x+=.5f;y-=.25f;z+=1.f;}if(mode==2){float old=x;x=-y;y=old;}c.p[i]=make_float4(x,y,z,0.f);}
  for(int i=0;i<n;i++){int slot=0;for(int j=0;j<n;j++)if(j!=i)c.neighbors[(slot++)*n+i]=j;}
  for(int axis=0;axis<3;axis++)for(int i=0;i<n;i++)prefix.push_back(axis==0?c.p[i].x:axis==1?c.p[i].y:c.p[i].z);for(int v:c.neighbors)prefix.push_back(float(v));
  // Independent pair enumeration: each unordered pair updates both force vectors.
  for(int i=0;i<n;i++)for(int slot=0;slot<128;slot++){int j=c.neighbors[slot*n+i];float dx=c.p[i].x-c.p[j].x,dy=c.p[i].y-c.p[j].y,dz=c.p[i].z-c.p[j].z,q=dx*dx+dy*dy+dz*dz;if(q>0&&q<13.5f){float inv=1.f/q,inv6=inv*inv*inv,f=inv*inv6*(1.5f*inv6-2.f);c.source[i]+=dx*f;c.source[n+i]+=dy*f;c.source[2*n+i]+=dz*f;}}
  int active=0,coincident=0,exactcut=0;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){double dx=double(c.p[i].x)-c.p[j].x,dy=double(c.p[i].y)-c.p[j].y,dz=double(c.p[i].z)-c.p[j].z,q=dx*dx+dy*dy+dz*dz;if(q==0)coincident++;if(q==13.5)exactcut++;if(q>0&&q<13.5){active++;double f=1.5*pow(q,-7)-2.*pow(q,-4);double v[3]={dx*f,dy*f,dz*f};for(int k=0;k<3;k++){c.gold[k*n+i]+=v[k];c.gold[k*n+j]-=v[k];}}}
  if(!active||!coincident||!exactcut)return 3;printf("MD_INPUT mode=%d unordered_active=%d coincident=%d exact_cutoff=%d\n",mode,active,coincident,exactcut);
  CK(cudaMalloc(&c.dp,n*sizeof(float4)));CK(cudaMalloc(&c.df,n*sizeof(float4)));CK(cudaMalloc(&c.dn,n*128*4));CK(cudaMemcpy(c.dp,c.p.data(),n*sizeof(float4),cudaMemcpyHostToDevice));CK(cudaMemcpy(c.dn,c.neighbors.data(),n*128*4,cudaMemcpyHostToDevice));md<<<1,256>>>(c.dp,c.df,c.dn,n,128,lj1,lj2,cutsq);cs.push_back(c);
 }CK(cudaDeviceSynchronize());bool all=true;
 // Independent force is negative derivative of U=lj1/(12r^12)-lj2/(6r^6).
 auto potential=[](double r){return 1.5/(12*pow(r,12))-2./(6*pow(r,6));};double rr=1.25,h=1e-5,analytic=1.5/pow(rr,13)-2./pow(rr,7),fd=-(potential(rr+h)-potential(rr-h))/(2*h);if(fabs(fd-analytic)>1e-8)return 4;printf("MD_PAIR_GRADIENT error=%.12g\n",fabs(fd-analytic));
 for(int trial=1;trial<=5;trial++){for(auto&c:cs)CK(cudaMemset(c.df,0xff,n*sizeof(float4)));CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<repeats;rep++)for(auto&c:cs)md<<<1,256>>>(c.dp,c.df,c.dn,n,128,lj1,lj2,cutsq);CK(cudaGetLastError());CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;auto out=prefix;double err=0,momentum=0,science_error=0,force_scale=0;bool ok=true;
  for(auto&c:cs){std::vector<float4>force(n);CK(cudaMemcpy(force.data(),c.df,n*sizeof(float4),cudaMemcpyDeviceToHost));checkResults<float,float4,float4>(force.data(),c.p.data(),c.neighbors.data(),n);for(int axis=0;axis<3;axis++){double sum=0,scale=0;for(int i=0;i<n;i++){float v=axis==0?force[i].x:axis==1?force[i].y:force[i].z;double matherr=fabs(double(v)-c.gold[axis*n+i]);science_error=std::max(science_error,matherr);double e=fabs(double(v)-c.source[axis*n+i]);ok=ok&&matherr<=1e-5+1e-6*fabs(c.gold[axis*n+i]);err=std::max(err,e);ok=ok&&std::isfinite(v)&&e<=1e-5&&force[i].w==0.f;out.push_back(v);sum+=v;scale+=fabs(double(v));}momentum=std::max(momentum,fabs(sum));force_scale=std::max(force_scale,scale);}}
  ok=ok&&momentum/force_scale<=1e-6;all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 5;fwrite(out.data(),4,out.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":1161,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"net_force_inf\":%.12g,\"independent_pair_error_inf\":%.12g,\"normalized_net_force\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":20,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",err,momentum,science_error,momentum/force_scale,path.c_str(),out.size());
 }puts(all?"PASS":"FAIL");return all?0:1;
}
