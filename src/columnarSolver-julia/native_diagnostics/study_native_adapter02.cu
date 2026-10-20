// Fixed supplied academic fixture only; full source workload retained.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <sys/stat.h>
#include "source_defines.h"
#include "corrected_observed_kernels.cuh"
#include "corrected_parser.inc"
#ifndef DATA_PATH
#define DATA_PATH "../columnarSolver-cuda/data"
#endif
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAERROR %s\n",cudaGetErrorString(e));abort();}}
template<class Elem>struct Buf{Elem*base;Elem*p;size_t n;Buf(size_t count):n(count){ck(cudaMalloc(&base,n*sizeof(Elem)+512));ck(cudaMemset(base,0xa7,n*sizeof(Elem)+512));p=(Elem*)((char*)base+256);}~Buf(){cudaFree(base);}std::vector<Elem>read(){std::vector<Elem>a(n);ck(cudaMemcpy(a.data(),p,n*sizeof(Elem),cudaMemcpyDeviceToHost));return a;}void set(const std::vector<Elem>&a){if(a.size()!=n)abort();ck(cudaMemcpy(p,a.data(),n*sizeof(Elem),cudaMemcpyHostToDevice));}void guards(std::vector<unsigned char>&all){unsigned char b[512];ck(cudaMemcpy(b,base,256,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b+256,(char*)p+n*sizeof(Elem),256,cudaMemcpyDeviceToHost));for(unsigned char x:b)if(x!=0xa7)abort();all.insert(all.end(),b,b+512);}};
void save(const std::string&p,const void*data,size_t bytes){FILE*f=fopen(p.c_str(),"wb");if(!f){perror(p.c_str());abort();}if(fwrite(data,1,bytes,f)!=bytes)abort();fclose(f);}
struct Check{bool pass;double maxscoreerror;int best;};
Check independent(const std::vector<int>&encrypted,const std::vector<float>&scores,const std::vector<int>&out,const std::vector<unsigned int>&diag){bool pass=true;double maxerr=0;int best=0;float bestscore=CAP;
 for(int tid=0;tid<THREADS;tid++){const unsigned int*d=&diag[63*tid];for(int phase=0;phase<2;phase++){bool seen[30]={};for(int c=0;c<30;c++){int rank=d[phase*30+c];if(rank<0||rank>=30||seen[rank])pass=false;else seen[rank]=true;}}
 int inv[30]={};for(int c=0;c<30;c++){if(d[30+c]>=30) return {false,0,0};inv[d[30+c]]=c;}int offset[30]={};int cur=0;for(int rank=0;rank<30;rank++){offset[rank]=cur;int c=inv[rank];cur+=ENCRYPTEDLEN/30+(c<ENCRYPTEDLEN%30);}if(cur!=ENCRYPTEDLEN)pass=false;
 float fs=0;double ds=0;int hist[26]={},ehist[26]={};for(int j=0;j<ENCRYPTEDLEN;j++){int v=out[tid*ENCRYPTEDLEN+j];int c=j%30;int expected=encrypted[offset[d[30+c]]+j/30];if(v!=expected||v<0||v>=26){pass=false;continue;}hist[v]++;ehist[encrypted[j]]++;if(j){float term=scores[out[tid*ENCRYPTEDLEN+j-1]*26+v];fs+=term;ds+=term;}}for(int j=0;j<26;j++)if(hist[j]!=ehist[j])pass=false;
 float actual;memcpy(&actual,d+60,4);if(memcmp(&actual,&fs,4)!=0||!std::isfinite(actual))pass=false;double err=std::abs(double(actual)-ds);maxerr=std::max(maxerr,err);double gamma=(396*0x1p-24)/(1-396*0x1p-24);if(err>gamma*ds)pass=false;if(actual<bestscore){bestscore=actual;best=tid;}}
 for(int j=0;j<ENCRYPTEDLEN;j++)if(out[best*ENCRYPTEDLEN+j]+'a'!=DECRYPTED_T[j])pass=false;return {pass,maxerr,best};}
int main(int argc,char**argv){bool diagnostic=argc==2; if((argc!=2&&argc!=3)){fprintf(stderr,"usage: fixed_fixture_native OUTPUTDIR [CPU_GOLD]\n");return 2;}std::string folder=argv[1];mkdir(folder.c_str(),0755);std::vector<float>scores(676,0);if(extractBigrams(scores.data(),DATA_PATH))return 3;std::vector<int>cipher(ENCRYPTEDLEN);for(int j=0;j<ENCRYPTEDLEN;j++)cipher[j]=ENCRYPTED_T[j]-'a';
 Buf<float>gs(676);Buf<int>ge(ENCRYPTEDLEN),go(ENCRYPTEDLEN*THREADS);Buf<unsigned int>gr(THREADS),gd(THREADS*63);gs.set(scores);ge.set(cipher);auto actualscores=gs.read();auto actualcipher=ge.read();if(memcmp(actualscores.data(),scores.data(),676*4)||actualcipher!=cipher)abort();std::vector<unsigned char>gold;if(!diagnostic){FILE*f=fopen(argv[2],"rb");if(!f)abort();gold.resize((THREADS+ENCRYPTEDLEN*THREADS+63*THREADS)*4);if(fread(gold.data(),1,gold.size(),f)!=gold.size())abort();fclose(f);}
 int last=diagnostic?0:5;
 for(int t=0;t<=last;t++){ck(cudaMemset(go.p,0xa7,go.n*4));ck(cudaMemset(gr.p,0xa7,gr.n*4));ck(cudaMemset(gd.p,0xa7,gd.n*4));ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();setupKernel<<<B,T>>>(gr.p);decode<<<B,T>>>(gs.p,ge.p,gr.p,go.p,gd.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto seeds=gr.read();auto out=go.read();auto diag=gd.read();auto afterscores=gs.read();auto aftercipher=ge.read();bool stable=memcmp(afterscores.data(),actualscores.data(),676*4)==0&&aftercipher==actualcipher;std::vector<unsigned char>guards;gs.guards(guards);ge.guards(guards);go.guards(guards);gr.guards(guards);gd.guards(guards);
 std::vector<unsigned char>raw;for(auto pair:{std::pair<const void*,size_t>{seeds.data(),seeds.size()*4},{out.data(),out.size()*4},{diag.data(),diag.size()*4}}){auto p=(const unsigned char*)pair.first;raw.insert(raw.end(),p,p+pair.second);}std::string path=folder+"/trial"+std::to_string(t);save(path+"_state.bin",raw.data(),raw.size());save(path+"_guards.bin",guards.data(),guards.size());bool source_match=diagnostic||raw==gold;auto check=independent(actualcipher,actualscores,out,diag);bool correct=source_match&&check.pass&&stable;
 std::vector<double>a;for(float x:actualscores)a.push_back(x);for(int x:actualcipher)a.push_back(x);for(unsigned x:seeds)a.push_back(x);for(int x:out)a.push_back(x);for(unsigned x:diag)a.push_back(x);for(unsigned char x:guards)a.push_back(x);std::string art="output_trial"+std::to_string(t)+".bin";save(art,a.data(),a.size()*8);
 printf("%s{\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"source_setup_plus_full150000_decode_completed\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"raw_state_path\":\"%s_state.bin\",\"guard_path\":\"%s_guards.bin\",\"source_CPU_exact\":%s,\"known_plaintext_winner\":%d,\"reported_abs_error_definition\":\"Float32bigram_sum_vs_independentFloat64_sum\"}\n",t>0?"STUDY_TRIAL ":"DIAGNOSTIC ",t,sec,correct?"true":"false",a.size(),check.maxscoreerror,art.c_str(),a.size(),path.c_str(),path.c_str(),source_match?"true":"false",check.best);fflush(stdout);}
 return 0;}
