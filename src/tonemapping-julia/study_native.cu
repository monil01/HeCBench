#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include "../tonemapping-cuda/kernels.cu"
static constexpr float GUARD=4096.25f;
void ckerr(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct G {
 int n;float *p;std::vector<float> expected;
 G(const std::vector<float>& x):n(x.size()),expected(x){std::vector<float> a(n+2,GUARD);std::copy(x.begin(),x.end(),a.begin()+1);ckerr(cudaMalloc(&p,a.size()*4));ckerr(cudaMemcpy(p,a.data(),a.size()*4,cudaMemcpyHostToDevice));}
 ~G(){cudaFree(p);}float* data(){return p+1;}
 std::vector<float> read(){std::vector<float>a(n+2);ckerr(cudaMemcpy(a.data(),p,a.size()*4,cudaMemcpyDeviceToHost));if(a.front()!=GUARD||a.back()!=GUARD)throw std::runtime_error("guard");return std::vector<float>(a.begin()+1,a.end()-1);}
 void guards(std::vector<float>&a){std::vector<float>v(n+2);ckerr(cudaMemcpy(v.data(),p,v.size()*4,cudaMemcpyDeviceToHost));if(v.front()!=GUARD||v.back()!=GUARD)throw std::runtime_error("guard");a.push_back(v.front());a.push_back(v.back());}
};
__global__ void echo(float*p,float av,float gamma,float c,float delta,uint w,uint h){if(threadIdx.x==0){p[0]=av;p[1]=gamma;p[2]=c;p[3]=delta;p[4]=w;p[5]=4;p[6]=h;}}
void append(std::vector<float>&a,const std::vector<float>&b){a.insert(a.end(),b.begin(),b.end());}
struct Answer{std::vector<float>prefix,outputs;double seconds=0,ae=0,re=0;};
void compare(Answer&a,float got,double want){double e=std::abs(double(got)-want);a.ae=std::max(a.ae,e);a.re=std::max(a.re,e/std::max(std::abs(want),1e-5));if(!std::isfinite(got)||e>1e-5+2e-5*std::abs(want))throw std::runtime_error("Float64 physical oracle mismatch");}
double lum64(const std::vector<float>&in,int j){return double(.2126f)*in[4*j]+double(.7152f)*in[4*j+1]+double(.0722f)*in[4*j+2];}
Answer execute(){Answer a;const int WS[]={16,32,64,256},HS[]={16,16,32,256};
 for(int c=0;c<4;c++){int w=WS[c],h=HS[c],n=w*h*4;std::vector<float>in(n),zero(n,0);
 for(int j=0;j<w*h;j++){for(int channel=0;channel<3;channel++){float value=.5f;if(c==1)value=1.f/64+float((j*(channel+3)+7*channel)%97)/16;if(c==2)value=.125f;if(c==3)value=std::ldexp(1.f,(j*(channel+3)+7*channel)%13-6)+1.f/64;in[4*j+channel]=value;}in[4*j+3]=float(j%17)/16;}
 if(c==2){int j=(h/2)*w+w/2;in[4*j]=64;in[4*j+1]=32;in[4*j+2]=16;}
 float av=0;for(int j=0;j<w*h;j++)av+=(.2126f*in[4*j]+.7152f*in[4*j+1])+.0722f*in[4*j+2];av/=float(w*h);
 float gamma=.4f,cc=.25f,delta=.000002f;std::vector<float>params={av,gamma,cc,delta,float(w),4.f,float(h)};G input(in),output(zero),param(std::vector<float>(7,0));
 echo<<<1,32>>>(param.data(),av,gamma,cc,delta,w,h);ckerr(cudaDeviceSynchronize());auto actual=input.read();if(actual!=in)throw std::runtime_error("input mismatch");append(a.prefix,actual);append(a.prefix,output.read());auto actualparam=param.read();if(actualparam!=params)throw std::runtime_error("scalar input mismatch");append(a.prefix,actualparam);input.guards(a.prefix);output.guards(a.prefix);param.guards(a.prefix);
 auto t=std::chrono::steady_clock::now();toneMapping<<<dim3(w/16,h/16),dim3(16,16)>>>(input.data(),output.data(),av,gamma,cc,delta,w,4,h);ckerr(cudaDeviceSynchronize());a.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();auto out=output.read();
 for(int yy=0;yy<h;yy++)for(int xx=0;xx<w;xx++){int j=yy*w+xx;double l=lum64(in,j),mean=l;if(xx>0&&yy>0&&xx<w-1&&yy<h-1){mean=0;for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)if(dx||dy)mean+=lum64(in,(yy+dy)*w+xx+dx);mean/=8;}double cp=mean*std::log(double(delta)+mean/l)+double(cc)*av;double display=l/(l+cp);for(int channel=0;channel<3;channel++){double want=std::pow(double(in[4*j+channel])/l,double(gamma))*display;compare(a,out[4*j+channel],want);if(!(out[4*j+channel]>=0))throw std::runtime_error("positive output");}if(out[4*j+3]!=in[4*j+3])throw std::runtime_error("alpha identity");if(c==0){double closed=1/(1+std::log(1+double(delta))+double(cc));for(int channel=0;channel<3;channel++)compare(a,out[4*j+channel],closed);}}
 append(a.outputs,out);actual=input.read();if(actual!=in)throw std::runtime_error("input mutation");append(a.outputs,actual);input.guards(a.outputs);output.guards(a.outputs);param.guards(a.outputs);
 }
 if(a.prefix.size()!=546868||a.outputs.size()!=546840)throw std::runtime_error("artifact count");return a;
}
int main(){try{execute();for(int trial=1;trial<=5;trial++){auto a=execute();append(a.prefix,a.outputs);std::string name="tonemapping_trial"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)a.prefix.data(),a.prefix.size()*4);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":1093708,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"timing_region\":\"sum4_source16x16_tonemapping_kernel_launch_completion\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":1093708,\"byte_order\":\"little\"}}\n",trial,a.seconds,a.ae,a.re,name.c_str());}}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
