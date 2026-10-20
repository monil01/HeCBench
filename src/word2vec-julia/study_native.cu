#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include "../word2vec-cuda/cbow.cu"
// Link compatibility for unused original host orchestration functions.
real *syn0=nullptr; int *table=nullptr;int vocab_size=0,layer1_size=0,layer1_size_aligned=0,negative=0,window=0,table_size=0;int *sen=nullptr;
void ck(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
struct Case{unsigned d,a,v,w,neg,ns,blocks,t,cold;std::vector<float>s0,s1,ex;std::vector<int>sentence,tab;std::vector<unsigned>rng;float *g0,*g1;int *gs,*gt;unsigned *gr;};
template<class T>void rd(std::ifstream&f,std::vector<T>&v,size_t n){v.resize(n);f.read((char*)v.data(),n*sizeof(T));if(!f)throw std::runtime_error("shortfixture");}
template<class T>void alloc(T*&p,size_t n){ck(cudaMalloc((void**)&p,n*sizeof(T)));}
template<class T>void upload(T*p,const std::vector<T>&v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void append(std::vector<double>&out,T*p,size_t n){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));for(auto z:x)out.push_back(double(z));}
void prep(Case&c){upload(c.g0,c.s0);upload(c.g1,c.s1);upload(c.gs,c.sentence);upload(c.gt,c.tab);upload(c.gr,c.rng);ck(cudaMemcpyToSymbol(expTable,c.ex.data(),1000*sizeof(float)));if(c.cold)device_memset<<<(c.v*c.a)/256+1,256>>>(c.g1,c.v*c.a);ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
void capture(Case&c,std::vector<double>&out){append(out,c.g0,c.s0.size());append(out,c.g1,c.s1.size());append(out,c.gs,c.sentence.size());append(out,c.gr,c.rng.size());append(out,c.gt,c.tab.size());float e[1000];ck(cudaMemcpyFromSymbol(e,expTable,sizeof(e)));for(float z:e)out.push_back(z);}
void launch(Case&c){device_cbow<<<c.blocks,128,(128+2*c.a)*sizeof(float)>>>(c.ns,c.d,c.a,c.w,c.neg,c.t,c.v,c.gs,c.gt,c.g0,c.g1,c.gr);ck(cudaGetLastError());}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("fixture oracle required");std::ifstream f(argv[1],std::ios::binary);unsigned n;f.read((char*)&n,4);if(!f||n!=4)throw std::runtime_error("fixturecasecount");std::vector<Case>cs(n);size_t prefix=0;
for(auto &c:cs){unsigned hdr[9];f.read((char*)hdr,sizeof(hdr));c.d=hdr[0];c.a=hdr[1];c.v=hdr[2];c.w=hdr[3];c.neg=hdr[4];c.ns=hdr[5];c.blocks=hdr[6];c.t=hdr[7];c.cold=hdr[8];rd(f,c.s0,c.v*c.a+17);rd(f,c.s1,c.v*c.a+17);rd(f,c.sentence,6150+17);rd(f,c.rng,1024+17);rd(f,c.tab,c.t+17);rd(f,c.ex,1000);alloc(c.g0,c.s0.size());alloc(c.g1,c.s1.size());alloc(c.gs,c.sentence.size());alloc(c.gt,c.tab.size());alloc(c.gr,c.rng.size());prefix+=c.s0.size()+c.s1.size()+c.sentence.size()+c.tab.size()+c.rng.size()+1000;}
std::ifstream go(argv[2],std::ios::binary);std::vector<double>gold(2*prefix);go.read((char*)gold.data(),gold.size()*8);if(!go||go.peek()!=EOF)throw std::runtime_error("oraclelength");
// Original fixed constant symbol is shared: upload each case table before its
// launch. All four fixtures intentionally contain byte-identical table knots.
for(auto&c:cs){prep(c);launch(c);ck(cudaDeviceSynchronize());} // untimed JIT/fullwork warmup
for(int tr=1;tr<=5;tr++){std::vector<double>out;out.reserve(gold.size());for(auto&c:cs){prep(c);capture(c,out);}auto t=std::chrono::steady_clock::now();for(auto&c:cs)launch(c);ck(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();for(auto&c:cs)capture(c,out);if(out.size()!=gold.size())throw std::runtime_error("capturecount");bool ok=true;double ma=0,mr=0;size_t at=prefix;for(size_t i=0;i<prefix;i++)if(out[i]!=gold[i])ok=false;
for(auto&c:cs){size_t lens[]={c.s0.size(),c.s1.size(),c.sentence.size(),c.rng.size(),c.tab.size(),1000};for(int fld=0;fld<6;fld++){for(size_t j=0;j<lens[fld];j++,at++){double e=std::abs(out[at]-gold[at]);ma=std::max(ma,e);if(gold[at]!=0)mr=std::max(mr,e/std::abs(gold[at]));bool exact=fld>=2||j>=lens[fld]-17; // active float padding checked by numerical budget; tail guards exact
if(!std::isfinite(out[at])||(exact?out[at]!=gold[at]:e>1e-6+2e-5*std::abs(gold[at])))ok=false;}}}
std::string fn=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/native_trial"+std::to_string(tr)+".f64";std::ofstream of(fn,std::ios::binary);of.write((char*)out.data(),out.size()*8);of.close();if(!of)throw std::runtime_error("artifactwrite");std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"correct\":"<<(ok?"true":"false")<<",\"time_seconds\":"<<sec<<",\"timing_valid\":true,\"jit_excluded\":true,\"timing_region\":\"completed_four_named_CBOW_kernel_cases\",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"actual_device_inputs_checked\":true,\"tail_guards_checked\":true,\"output_artifact\":{\"path\":\""<<fn<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
for(auto&c:cs){ck(cudaFree(c.g0));ck(cudaFree(c.g1));ck(cudaFree(c.gs));ck(cudaFree(c.gt));ck(cudaFree(c.gr));}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
