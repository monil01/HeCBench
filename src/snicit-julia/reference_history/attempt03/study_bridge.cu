// Shared source-shaped CPU controller; frontend callback owns every GPU entrypoint.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include "study_cases.h"
#include "../snicit-cuda/src/general.hpp"
#include "../snicit-cuda/src/kernel.hpp"
using namespace SNICIT_BEY;
using Callback=void(*)(int,void**,int*,void*);
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}
static constexpr int G=16;static constexpr float SENT=12345.25f;
struct Owned{void*p;size_t n;int size;bool isfloat;};
static std::vector<Owned> allocations;
template<class T>T* alloc(int n,bool managed,const T*src=nullptr){T*p=nullptr;if(managed)ck(cudaMallocManaged(&p,(n+G)*sizeof(T)));else ck(cudaMalloc(&p,(n+G)*sizeof(T)));std::vector<T>h(n+G,T(0));if(src)memcpy(h.data(),src,n*sizeof(T));for(int j=n;j<n+G;j++)h[j]=T(91);ck(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));allocations.push_back({p,size_t(n),int(sizeof(T)),std::is_same<T,float>::value});return p;}
static void native_launch(int op,void**p,int*a,void*s){cudaStream_t st=(cudaStream_t)s;dim3 b(a[5],a[6]);int M=a[0],N=a[1],K=a[2],C=a[3];int gr=a[4],sh=a[7];
#define F(i) ((float*)p[i])
#define I(i) ((int*)p[i])
#define B(i) ((bool*)p[i])
 switch(op){case 0:dense_input<<<gr,b,sh,st>>>(F(0),F(1),F(2),M,N,K,F(3));break;
 case 1:sparse_hidden<<<gr,b,sh,st>>>(F(0),I(4),I(5),F(6),F(2),M,N,K,F(3));break;
 case 2:dense_output<<<gr,b,sh,st>>>(F(0),F(1),F(2),M,N,K,F(3));break;
 case 3:check_acc<<<gr,b,sh,st>>>(F(0),K,M,I(11),I(12));break;
 case 4:y_star_gen<<<gr,b,sh,st>>>(F(0),I(10),M,N,128);break;
 case 5:coarse_cluster<<<gr,b,sh,st>>>(F(0),I(10),B(9),C,I(8),N);break;
 case 6:sparse_hidden_post<<<gr,b,sh,st>>>(I(7),F(0),I(4),I(5),F(6),M,N,K,F(3));break;
 case 7:update_post<<<gr,b,sh,st>>>(I(7),I(8),F(0),F(2),N,B(9),F(3));break;
 case 8:recover<<<gr,b,sh,st>>>(F(0),I(8),N);break;}
}
static void write_state(FILE*f,const Case&c,float*h0,float*h1,float*out,int*ys,int*lut,bool*ne,int*rows,int*cnt){
 std::vector<float>v(2*c.M*c.H+c.M*10);ck(cudaMemcpy(v.data(),h0,c.M*c.H*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(v.data()+c.M*c.H,h1,c.M*c.H*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(v.data()+2*c.M*c.H,out,c.M*10*4,cudaMemcpyDeviceToHost));fwrite(v.data(),4,v.size(),f);
 fwrite(ys,4,128,f);fwrite(lut,4,c.M,f);for(int j=0;j<c.M;j++){int x=ne[j];fwrite(&x,4,1,f);}fwrite(rows,4,c.M,f);fwrite(cnt,4,1,f);
}
extern "C" void study_run(Callback cb,int trial,const char*input,const char*outdir,void*externalstream){
 if(!cb)cb=native_launch;std::vector<uint32_t>fixture(4548580);FILE*f=fopen(input,"rb");if(!f||fread(fixture.data(),4,fixture.size(),f)!=fixture.size())abort();fclose(f);
 std::string output=std::string(outdir)+"/trial"+std::to_string(trial)+".bin";FILE*fo=trial?fopen(output.c_str(),"wb"):nullptr;
 if(fo){std::vector<double>prefix;for(auto u:fixture)prefix.push_back(double(u));fwrite(prefix.data(),8,prefix.size(),fo);}
 std::vector<uint32_t>actual_fixture=fixture;double total=0;bool valid=true;long checked=0;
 for(const Case&c:cases){allocations.clear();auto fp=[&](int off){return(float*)(fixture.data()+off);};auto ip=[&](int off){return(int*)(fixture.data()+off);};
 float*x=alloc<float>(c.M*c.N,true,fp(c.x));float*win=c.cifar?nullptr:alloc<float>(c.N*c.H,false,fp(c.win));float*bin=c.cifar?nullptr:alloc<float>(c.H,false,fp(c.bin));
 float*h[2]={alloc<float>(c.M*c.H,true),alloc<float>(c.M*c.H,true)},*out=alloc<float>(c.M*10,true),*wo=alloc<float>(c.H*10,false,fp(c.wout)),*bo=alloc<float>(10,false,fp(c.bout));
 int*ys=alloc<int>(128,true),*lut=alloc<int>(c.M,true),*rows=alloc<int>(c.M,true),*label=alloc<int>(c.M,true,ip(c.label)),*cnt=alloc<int>(1,true);bool*ne=(bool*)alloc<unsigned char>(c.M,true);
 int*rp[18],*col[18];float*val[18],*bias[18];for(int l=0;l<c.L;l++){auto e=layers[c.id][l];rp[l]=alloc<int>(c.H+1,true,ip(e.rp));col[l]=alloc<int>(e.nnz,true,ip(e.col));val[l]=alloc<float>(e.nnz,true,fp(e.val));bias[l]=alloc<float>(c.H,true,fp(e.bias));}
 auto verifyinput=[&](void*ptr,int offset,int count){ck(cudaMemcpy(actual_fixture.data()+offset,ptr,count*4,cudaMemcpyDeviceToHost));};
 verifyinput(x,c.x,c.M*c.N);verifyinput(label,c.label,c.M);if(!c.cifar){verifyinput(win,c.win,c.N*c.H);verifyinput(bin,c.bin,c.H);}verifyinput(wo,c.wout,c.H*10);verifyinput(bo,c.bout,10);
 for(int l=0;l<c.L;l++){auto e=layers[c.id][l];verifyinput(rp[l],e.rp,c.H+1);verifyinput(col[l],e.col,e.nnz);verifyinput(val[l],e.val,e.nnz);verifyinput(bias[l],e.bias,c.H);}
 cudaStream_t stream=(cudaStream_t)externalstream;if(!stream)ck(cudaStreamCreate(&stream));void*p[13]={};int a[8]={c.M,c.H,c.H,0,c.M,1024/c.H,c.H,4*c.H};p[7]=rows;p[8]=lut;p[9]=ne;p[10]=ys;p[11]=label;p[12]=cnt;
 std::string tr=std::string(outdir)+"/trace_case"+std::to_string(c.id)+".bin";FILE*ft=trial?nullptr:fopen(tr.c_str(),"wb");
 auto launch=[&](int op,int layer){if(ft){int hd[5]={op,c.id,layer,a[3],a[4]};fwrite(hd,4,5,ft);write_state(ft,c,h[0],h[1],out,ys,lut,ne,rows,cnt);}cb(op,p,a,(void*)stream);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));if(ft)write_state(ft,c,h[0],h[1],out,ys,lut,ne,rows,cnt);};
 if(c.cifar)ck(cudaMemcpy(h[0],x,c.M*c.H*4,cudaMemcpyDeviceToDevice));else{p[0]=x;p[1]=win;p[2]=bin;p[3]=h[0];a[1]=c.N;a[2]=c.H;a[5]=c.H;a[6]=1024/c.H;launch(0,-1);}
 a[1]=a[2]=c.H;a[5]=1024/c.H;a[6]=c.H;
 auto start=std::chrono::steady_clock::now();
 for(int l=0;l<c.T;l++){p[0]=h[l%2];p[3]=h[(l+1)%2];p[2]=bias[l];p[4]=rp[l];p[5]=col[l];p[6]=val[l];launch(1,l);ck(cudaMemset(h[l%2],0,c.M*c.H*4));}
 p[0]=h[c.T%2];a[4]=1;a[5]=8;a[6]=128;a[7]=4*(c.H+256);launch(4,-1);
 int ns=0;for(int j=0;j<128;j++)if(ys[j]!=-1){lut[ys[j]]=-1;ys[ns++]=ys[j];}if(ns<1||ns>128)abort();
 a[3]=ns;a[4]=c.M;a[5]=1024/ns;a[6]=ns;a[7]=4*(c.H+ns);launch(5,-1);
 int nr=0;for(int j=0;j<c.M;j++)if(ne[j])rows[nr++]=j;
 for(int l=c.T;l<c.L;l++){if(nr<1)abort();p[0]=h[c.T%2];p[3]=h[(c.T+1)%2];p[4]=rp[l];p[5]=col[l];p[6]=val[l];p[2]=bias[l];a[3]=nr;a[4]=nr;a[5]=1024/c.H;a[6]=c.H;a[7]=4*c.H;launch(6,l);
 p[0]=h[(c.T+1)%2];p[3]=h[c.T%2];a[5]=c.H;a[6]=1;a[7]=0;launch(7,l);
 int n=0;for(int j=0;j<nr;j++)if(ne[rows[j]])rows[n++]=rows[j];nr=n;ck(cudaMemset(h[(c.T+1)%2],0,c.M*c.H*4));}
 p[0]=h[c.T%2];a[4]=c.M;a[5]=c.H;a[6]=1;a[7]=4*c.H;launch(8,-1);total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 p[0]=h[c.T%2];p[1]=wo;p[2]=bo;p[3]=out;a[1]=c.H;a[2]=10;a[5]=10;a[6]=102;a[7]=40;launch(2,-1);
 p[0]=out;a[4]=1;a[5]=1024;a[6]=1;a[7]=4;launch(3,-1);
 if(ft)fclose(ft);
 // All surviving state, exact categorical outputs and redzones after ROI.
 if(fo){FILE*tmp=tmpfile();write_state(tmp,c,h[0],h[1],out,ys,lut,ne,rows,cnt);long bytes=ftell(tmp);rewind(tmp);std::vector<uint32_t>s(bytes/4);fread(s.data(),4,s.size(),tmp);fclose(tmp);size_t nf=2*c.M*c.H+c.M*10;for(size_t j=0;j<s.size();j++){double d;if(j<nf){float v;memcpy(&v,&s[j],4);valid &=std::isfinite(v);d=v;}else d=int32_t(s[j]);fwrite(&d,8,1,fo);checked++;}
 for(int row=0;row<c.M;row++){int pred=0;for(int j=1;j<10;j++)if(out[row*10+j]>out[row*10+pred])pred=j;double d=pred;fwrite(&d,8,1,fo);checked++;}
 int actual=0;for(int row=0;row<c.M;row++){int pred=0;for(int j=1;j<10;j++)if(out[row*10+j]>out[row*10+pred])pred=j;actual+=pred==label[row];}valid &=cnt[0]==actual;
 for(auto e:allocations){std::vector<unsigned char>g(G*e.size);ck(cudaMemcpy(g.data(),(char*)e.p+e.n*e.size,g.size(),cudaMemcpyDeviceToHost));for(int j=0;j<G;j++){double d;if(e.size==1)d=g[j]!=0;else if(e.isfloat){float v;memcpy(&v,g.data()+j*4,4);d=v;}else{int v;memcpy(&v,g.data()+j*4,4);d=v;}valid &=d==(e.size==1?1:91);fwrite(&d,8,1,fo);checked++;}}
 }
 if(!externalstream)ck(cudaStreamDestroy(stream));for(auto e:allocations)ck(cudaFree(e.p));
 }
 if(fo){long n=ftell(fo)/8;valid &=actual_fixture==fixture;rewind(fo);for(auto u:actual_fixture){double d=u;fwrite(&d,8,1,fo);}fclose(fo);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%ld,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"ABCD_complete_sourceSparseROI_GPUstages_CPUadaptivecontrol_managedbuffers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%ld,\"byte_order\":\"little\"}}\n",trial,total,valid?"true":"false",checked,output.c_str(),n);fflush(stdout);if(!valid)exit(1);}
}
#ifdef STUDY_MAIN
int main(){auto d=getenv("STUDY_OUTPUT_DIR");study_run(nullptr,0,"input_u32.bin",d,nullptr);for(int t=1;t<=5;t++)study_run(nullptr,t,"input_u32.bin",d,nullptr);}
#endif
