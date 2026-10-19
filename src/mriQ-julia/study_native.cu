#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include "../mriQ-cuda/computeQ.cu"
static constexpr float GUARD=4096.25f;
static const int KS[]={1,2,257,1024,1026}, XS[]={256,256,512,256,512};
void ckerr(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct G {
 int n;float *p;std::vector<float> expected;
 G(const std::vector<float>& x):n(x.size()),expected(x){std::vector<float> a(n+2,GUARD);std::copy(x.begin(),x.end(),a.begin()+1);ckerr(cudaMalloc(&p,a.size()*4));ckerr(cudaMemcpy(p,a.data(),a.size()*4,cudaMemcpyHostToDevice));}
 ~G(){cudaFree(p);}float* data(){return p+1;}
 std::vector<float> read(){std::vector<float>a(n+2);ckerr(cudaMemcpy(a.data(),p,a.size()*4,cudaMemcpyDeviceToHost));if(a.front()!=GUARD||a.back()!=GUARD)throw std::runtime_error("guard");return std::vector<float>(a.begin()+1,a.end()-1);}
 void appendguards(std::vector<float>&a){std::vector<float>v(n+2);ckerr(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));if(v.front()!=GUARD||v.back()!=GUARD)throw std::runtime_error("guard");a.push_back(v.front());a.push_back(v.back());}
};
void append(std::vector<float>&a,const std::vector<float>&b){a.insert(a.end(),b.begin(),b.end());}
struct Answer {std::vector<float> prefix,values;double seconds=0,ae=0,re=0;};
void compare(Answer&a,float got,double want){double e=std::abs(double(got)-want);a.ae=std::max(a.ae,e);a.re=std::max(a.re,e/std::max(std::abs(want),1e-4));if(!std::isfinite(got)||e>1e-4+2e-5*std::abs(want))throw std::runtime_error("Float64 scientific oracle mismatch");}
Answer execute(){Answer a;
 for(int c=0;c<5;c++){
 int k=KS[c],n=XS[c];std::vector<float>kr(k),ki(k),kz(k),pr(k),pi(k),x(n),y(n),z(n),zerosK(k,0),zerosX(n,0);
 for(int i=0;i<k;i++){kr[i]=float(i%13-6)/32;ki[i]=float((i*3)%11-5)/32;kz[i]=float((i*7)%9-4)/32;pr[i]=float(i%7+1)/16;pi[i]=float((i*5)%7-3)/16;}
 if(c==0){kr[0]=.125f;ki[0]=-.0625f;kz[0]=.03125f;}
 if(c==1){kr[1]=-kr[0];ki[1]=-ki[0];kz[1]=-kz[0];pr[1]=pr[0];pi[1]=pi[0];}
 for(int i=0;i<n;i++){x[i]=float(i%17)/16;y[i]=float((i*3)%19)/16;z[i]=float((i*7)%23)/16;}x[0]=y[0]=z[0]=0;
 G gkr(kr),gki(ki),gkz(kz),gpr(pr),gpi(pi),gx(x),gy(y),gz(z),gm(zerosK),gr(zerosX),gi(zerosX);
 G* inputs[]={&gkr,&gki,&gkz,&gpr,&gpi,&gx,&gy,&gz};G* all[]={&gkr,&gki,&gkz,&gpr,&gpi,&gx,&gy,&gz,&gm,&gr,&gi};
 for(G*g:inputs){auto actual=g->read();if(actual!=g->expected)throw std::runtime_error("input mismatch");append(a.prefix,actual);}append(a.prefix,gm.read());append(a.prefix,gr.read());append(a.prefix,gi.read());for(G*g:all)g->appendguards(a.prefix);
 auto start=std::chrono::steady_clock::now();ComputePhiMag_GPU<<<(k+255)/256,256>>>(gpr.data(),gpi.data(),gm.data(),k);ckerr(cudaDeviceSynchronize());a.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 auto mag=gm.read();std::vector<kValues> kv(k);for(int i=0;i<k;i++){compare(a,mag[i],double(pr[i])*pr[i]+double(pi[i])*pi[i]);kv[i]={kr[i],ki[i],kz[i],mag[i]};}
 for(int base=0;base<k;base+=1024){int count=std::min(1024,k-base);start=std::chrono::steady_clock::now();ckerr(cudaMemcpyToSymbol(ck,kv.data()+base,count*sizeof(kValues)));ComputeQ_GPU<<<n/256,256>>>(k,base,gx.data(),gy.data(),gz.data(),gr.data(),gi.data());ckerr(cudaDeviceSynchronize());a.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<kValues> actual(count);ckerr(cudaMemcpyFromSymbol(actual.data(),ck,count*sizeof(kValues)));for(int i=0;i<count;i++){float*v=(float*)&actual[i];float*w=(float*)&kv[base+i];for(int t=0;t<4;t++){if(v[t]!=w[t])throw std::runtime_error("constant input readback");a.prefix.push_back(v[t]);}}}
 auto qr=gr.read(),qi=gi.read();
 for(int j=0;j<n;j++){double rr=0,ii=0;for(int i=0;i<k;i++){double m=double(pr[i])*pr[i]+double(pi[i])*pi[i];double angle=2*std::acos(-1.0)*(double(kr[i])*x[j]+double(ki[i])*y[j]+double(kz[i])*z[j]);rr+=m*std::cos(angle);ii+=m*std::sin(angle);}compare(a,qr[j],rr);compare(a,qi[j],ii);if(c==0)compare(a,float(std::hypot(double(qr[j]),double(qi[j]))),double(pr[0])*pr[0]+double(pi[0])*pi[0]);if(c==1&&std::abs(qi[j])>1e-6)throw std::runtime_error("paired cancellation");}
 double summ=0;for(float m:mag)summ+=m;if(qr[0]!=summ||qi[0]!=0)throw std::runtime_error("zero spatial sample");append(a.values,mag);append(a.values,qr);append(a.values,qi);
 for(G*g:inputs){auto actual=g->read();if(actual!=g->expected)throw std::runtime_error("input mutation");append(a.values,actual);}for(G*g:all)g->appendguards(a.values);
 }
 if(a.prefix.size()!=32170||a.values.size()!=22930)throw std::runtime_error("artifact count");return a;
}
int main(){try{execute();for(int t=1;t<=5;t++){auto a=execute();append(a.prefix,a.values);std::string name="mriq_trial"+std::to_string(t)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)a.prefix.data(),a.prefix.size()*4);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":55100,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"timing_region\":\"sum11_Phi_and_Q_upload_launch_completion_fragments\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":55100,\"byte_order\":\"little\"}}\n",t,a.seconds,a.ae,a.re,name.c_str());}}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
