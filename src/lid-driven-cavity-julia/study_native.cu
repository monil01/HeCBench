#define main archived_cavity_main
#include "native_dimension_specialization.cuh"
#undef main
#undef u
#undef v
#undef F
#undef G
#undef pres_red
#undef pres_black
#include <vector>
#include <string>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double>read(const char*path,int n){std::vector<double>v(n);FILE*f=fopen(path,"rb");if(!f||fread(v.data(),8,n,f)!=size_t(n)){fprintf(stderr,"Fixture read failed %s\n",path);exit(3);}fclose(f);return v;}
int main(){auto prefix=read("study_input_prefix.bin",5856),gold=read("study_expected.bin",5980);int lengths[10]={1156,1156,1156,1156,612,612,32,32,32,32};double*dev[10];std::vector<double>initial[10];int pos=8;for(int i=0;i<10;i++){initial[i].resize(lengths[i]);if(i<6){std::copy(prefix.begin()+pos,prefix.begin()+pos+lengths[i],initial[i].begin());pos+=lengths[i];}CK(cudaMalloc(&dev[i],8*lengths[i]));}
 std::vector<double>hu(1156),hv(1156);set_BCs_host(hu.data(),hv.data());if(hu!=initial[0]||hv!=initial[1]){for(int i=0;i<1156;i++)if(hu[i]!=initial[0][i]||hv[i]!=initial[1][i])fprintf(stderr,"BC initial mismatch index%d u%.12g vs%.12g v%.12g vs%.12g\n",i,hu[i],initial[0][i],hv[i],initial[1][i]);return 4;}bool all=true;
 for(int trial=1;trial<=5;trial++){for(int i=0;i<10;i++)CK(cudaMemcpy(dev[i],initial[i].data(),8*lengths[i],cudaMemcpyHostToDevice));double maxu=2,maxv=1e-10,time=0,dt=0,norm=0;int iter=0,steps=0;auto*u=dev[0],*v=dev[1],*F=dev[2],*G=dev[3],*pr=dev[4],*pb=dev[5];std::vector<double>ps(32),res(32),mu(32),mv(32);dim3 pressure(1,32),field(2,32);CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
 while(time<.02){dt=tau*fmin(.5*Re_num/(1/(dx*dx)+1/(dy*dy)),fmin(dx/maxu,dy/maxv));if(time+dt>=.02)dt=.02-time;
 calculate_F<<<field,16>>>(dt,u,v,F);calculate_G<<<field,16>>>(dt,u,v,G);sum_pressure<<<pressure,16>>>(pr,pb,dev[6]);CK(cudaMemcpy(ps.data(),dev[6],256,cudaMemcpyDeviceToHost));double p0=0;for(double x:ps)p0+=x;p0=sqrt(p0/1024);if(p0<.0001)p0=1;
 for(iter=1;iter<=1000000;iter++){set_horz_pres_BCs<<<1,16>>>(pr,pb);set_vert_pres_BCs<<<1,16>>>(pr,pb);red_kernel<<<pressure,16>>>(dt,F,G,pb,pr);black_kernel<<<pressure,16>>>(dt,F,G,pr,pb);calc_residual<<<pressure,16>>>(dt,F,G,pr,pb,dev[7]);CK(cudaMemcpy(res.data(),dev[7],256,cudaMemcpyDeviceToHost));norm=0;for(double x:res)norm+=x;norm=sqrt(norm/1024)/p0;if(norm<.001)break;}if(iter>1000000)return 5;
 calculate_u<<<pressure,16>>>(dt,F,pr,pb,u,dev[8]);CK(cudaMemcpy(mu.data(),dev[8],256,cudaMemcpyDeviceToHost));calculate_v<<<pressure,16>>>(dt,G,pr,pb,v,dev[9]);CK(cudaMemcpy(mv.data(),dev[9],256,cudaMemcpyDeviceToHost));maxu=maxv=1e-10;for(int i=0;i<32;i++){maxu=fmax(maxu,mu[i]);maxv=fmax(maxv,mv[i]);}set_BCs<<<2,16>>>(u,v);time+=dt;steps++;printf("CAVITY_STEP trial=%d step=%d time=%.12g dt=%.12g iterations=%d residual=%.12g\n",trial,steps,time,dt,iter,norm);
 }CK(cudaGetLastError());CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto out=prefix;for(int i=0;i<10;i++){std::vector<double>tmp(lengths[i]);CK(cudaMemcpy(tmp.data(),dev[i],8*lengths[i],cudaMemcpyDeviceToHost));out.insert(out.end(),tmp.begin(),tmp.end());}for(double x:{double(iter),dt,norm,double(steps)})out.push_back(x);bool ok=true;double err=0;for(int i=0;i<5980;i++){double value=out[5856+i],e=fabs(value-gold[i]);err=std::max(err,e);ok=ok&&std::isfinite(value)&&e<=1e-9+1e-10*fabs(gold[i]);}all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 6;fwrite(out.data(),8,out.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":5980,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"coupled_transient_projection_loop_host_checks_and_partial_transfers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",err,path.c_str(),out.size());}puts(all?"PASS":"FAIL");return all?0:1;}
