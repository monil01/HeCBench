#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include "../minibude-cuda/kernel.cu"
#include "study_params.h"
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(7);}}while(0)
template<class T>std::vector<T>read(std::string f,int n){std::ifstream a(f,std::ios::binary);std::vector<T>v(n);a.read((char*)v.data(),n*sizeof(T));if(!a||a.peek()!=EOF){fprintf(stderr,"assetbytes %s\n",f.c_str());exit(2);}return v;}
template<class T>void save(std::string f,const std::vector<T>&v){std::ofstream a(f,std::ios::binary);a.write((const char*)v.data(),v.size()*sizeof(T));if(!a)exit(3);}
template<class T>struct B{
 T *base,*p;std::string name;std::vector<T>initial,before,after;
 B(std::string label,std::vector<T>a):name(label){initial.resize(a.size()+32);memset(initial.data(),0xa5,initial.size()*sizeof(T));memcpy(initial.data()+16,a.data(),a.size()*sizeof(T));C(cudaMalloc(&base,initial.size()*sizeof(T)));p=base+16;reset();}
 void reset(){C(cudaMemcpy(base,initial.data(),initial.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>capture(){std::vector<T>x(initial.size());C(cudaMemcpy(x.data(),base,x.size()*sizeof(T),cudaMemcpyDeviceToHost));return x;}
 bool equal(const std::vector<T>&x){return x.size()==initial.size()&&memcmp(x.data(),initial.data(),x.size()*sizeof(T))==0;}
 void start(){before=capture();if(!equal(before))exit(4);}
 void finish(){after=capture();}
 void append(std::vector<double>&out,const std::vector<T>&v){for(size_t i=0;i<v.size()*sizeof(T)/4;i++){uint32_t word;memcpy(&word,((const char*)v.data())+4*i,4);out.push_back(double(word));}}
 void dump(std::string path){save(path+"_"+name+"_before.bin",before);save(path+"_"+name+"_after.bin",after);}
};
struct Work{
 int id,n,np,nl,nf;B<Atom>protein,ligand;B<FFParams>ff;std::vector<B<float>*>poses;B<float>out;std::vector<double>oracle;
 Work(int k):id(k),n(NS[k]),np(NPRO[k]),nl(NLIG[k]),nf(NFT[k]),protein("protein",read<Atom>("case"+std::to_string(k)+"_protein.bin",np)),ligand("ligand",read<Atom>("case"+std::to_string(k)+"_ligand.bin",nl)),ff("forcefield",read<FFParams>("case"+std::to_string(k)+"_forcefield.bin",nf)),out("energy",std::vector<float>(n,-777.f)),oracle(read<double>("case"+std::to_string(k)+"_oracle_channels_f64.bin",6*n)){
  auto t=read<float>("case"+std::to_string(k)+"_poses.bin",6*n);for(int a=0;a<6;a++)poses.push_back(new B<float>("pose"+std::to_string(a),std::vector<float>(t.begin()+a*n,t.begin()+(a+1)*n)));
 }
 void reset(){protein.reset();ligand.reset();ff.reset();out.reset();for(auto x:poses)x->reset();protein.start();ligand.start();ff.start();out.start();for(auto x:poses)x->start();}
 double run(){C(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<R;r++)fasten_main<<<(n+255)/256,256,nf*sizeof(FFParams)>>>(nf,n,nl,np,protein.p,ligand.p,poses[0]->p,poses[1]->p,poses[2]->p,poses[3]->p,poses[4]->p,poses[5]->p,ff.p,out.p);C(cudaGetLastError());C(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/R;}
 void capture(std::vector<double>&prefix){protein.finish();ligand.finish();ff.finish();for(auto x:poses)x->finish();out.finish();prefix.insert(prefix.end(),{double(id),double(n),double(np),double(nl),double(nf),double(R),256.,double((n+255)/256),1.});protein.append(prefix,protein.before);ligand.append(prefix,ligand.before);ff.append(prefix,ff.before);for(auto x:poses)x->append(prefix,x->before);out.append(prefix,out.before);protein.append(prefix,protein.after);ligand.append(prefix,ligand.after);ff.append(prefix,ff.after);for(auto x:poses)x->append(prefix,x->after);}
 bool tail(std::vector<double>&record,std::string raw,double&maxabs,double&maxbudget){bool good=protein.equal(protein.after)&&ligand.equal(ligand.after)&&ff.equal(ff.after);for(auto x:poses)good&=x->equal(x->after);for(int i=0;i<n;i++){double actual=out.after[i+16],expected=oracle[6*i],e=std::abs(actual-expected),budget=.05+.0002*std::abs(expected);good&=std::isfinite(actual)&&e<=budget;maxabs=std::max(maxabs,e);maxbudget=std::max(maxbudget,e/budget);record.push_back(actual);}for(int i=0;i<16;i++){uint32_t a,b;memcpy(&a,out.after.data()+i,4);memcpy(&b,out.after.data()+n+16+i,4);good&=a==0xa5a5a5a5u&&b==0xa5a5a5a5u;record.push_back(double(a));}for(int i=0;i<16;i++){uint32_t b;memcpy(&b,out.after.data()+n+16+i,4);record.push_back(double(b));}record.push_back(good?1.:0.);protein.dump(raw);ligand.dump(raw);ff.dump(raw);out.dump(raw);for(auto x:poses)x->dump(raw);return good;}
};
int main(){static_assert(sizeof(Atom)==16&&sizeof(FFParams)==16,"sourceABI");std::vector<Work*>w;for(int k=0;k<NCASES;k++)w.push_back(new Work(k));for(auto x:w)x->run();for(int trial=1;trial<=5;trial++){for(auto x:w)x->reset();double seconds=0;for(auto x:w)seconds+=x->run();std::vector<double>record;record.reserve(COUNT);for(auto x:w)x->capture(record);if(record.size()!=PREFIX)return 5;bool good=true;double maxabs=0,maxbudget=0;std::string raw=std::string(getenv("STUDY_RAW_DIR"))+"/native_pid"+std::to_string((long long)getpid())+"_trial"+std::to_string(trial);for(auto x:w)good&=x->tail(record,raw+"_case"+std::to_string(x->id),maxabs,maxbudget);if(record.size()!=COUNT)return 6;std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/minibude_trial"+std::to_string(trial)+".bin";save(path,record);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"maximum_independent_error_over_budget\":%.17g,\"jit_excluded\":true,\"internal_repeats\":7,\"kernel_launches\":42,\"timing_region\":\"sixcase_sourceR7_regionmean_sum_completedwarmup_contextsync\",\"raw_state_prefix\":\"%s\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,good?"true":"false",record.size(),maxabs,maxbudget,raw.c_str(),path.c_str(),record.size());fflush(stdout);if(!good)return 8;}return 0;}
