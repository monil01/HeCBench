#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include "corrected_kernels.cuh"
#ifndef FIXTURE
#error FIXTURE required
#endif
#ifndef ORACLE
#error ORACLE required
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double>load(const std::string&name,size_t expected){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=(std::streamoff)(8*expected))exit(3);f.seekg(0);std::vector<double>a(expected);f.read((char*)a.data(),expected*8);if(!f)exit(3);return a;}
struct Outcome{bool ok;std::vector<double>art;double seconds,maxerr;};
struct CaseBase{virtual Outcome execute()=0;};
template<class T>struct Case:CaseBase{int mode,iw,ih,ow,oh,C,repeat;size_t ni,no,lead;T guard,*abase,*obase,*a,*o;std::vector<T>input;std::vector<double>gold;float fx,fy;
Case(int m,int x,int y,int u,int v,int c,int rep,std::vector<T>h,std::vector<double>g):mode(m),iw(x),ih(y),ow(u),oh(v),C(c),repeat(rep),ni(h.size()),no(c*u*v),lead(256/sizeof(T)),input(h),gold(g),fx(resize_scale(x,u)),fy(resize_scale(y,v)){guard=(T)(sizeof(T)==1?0xa5u:sizeof(T)==2?0xa55au:0xa55aa55au);CK(cudaMalloc(&abase,(ni+lead+16)*sizeof(T)));CK(cudaMalloc(&obase,(no+lead+16)*sizeof(T)));a=abase+lead;o=obase+lead;if((uintptr_t)a%256||(uintptr_t)o%256)exit(3);std::vector<T>full(ni+lead+16,guard);std::copy(h.begin(),h.end(),full.begin()+lead);CK(cudaMemcpy(abase,full.data(),full.size()*sizeof(T),cudaMemcpyHostToDevice));}
void launch(){if(mode)resize_bilinear<T,8><<<29184,256>>>(o,no,oh,ow,a,ih,iw,fy,fx,true);else resize<T,8><<<29184,256>>>(o,no,oh,ow,a,ih,iw,fy,fx,true,true);}
Outcome execute(){std::vector<T>ob(no+lead+16,guard),ib(ni+lead+16);CK(cudaMemcpy(obase,ob.data(),ob.size()*sizeof(T),cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());std::vector<T>before(ni);CK(cudaMemcpy(before.data(),a,ni*sizeof(T),cudaMemcpyDeviceToHost));Outcome z;z.ok=before==input;z.maxerr=0;auto start=std::chrono::steady_clock::now();for(int rep=0;rep<repeat;++rep)launch();CK(cudaDeviceSynchronize());z.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();CK(cudaMemcpy(ib.data(),abase,ib.size()*sizeof(T),cudaMemcpyDeviceToHost));CK(cudaMemcpy(ob.data(),obase,ob.size()*sizeof(T),cudaMemcpyDeviceToHost));for(size_t i=0;i<no;++i)z.art.push_back((double)ob[lead+i]);for(size_t i=0;i<ni;++i){z.art.push_back((double)ib[lead+i]);z.ok&=ib[lead+i]==input[i];}for(auto b:{&ib,&ob}){for(size_t i=0;i<lead;++i){z.art.push_back((double)(*b)[i]);z.ok&=(*b)[i]==guard;}for(size_t i=b->size()-16;i<b->size();++i){z.art.push_back((double)(*b)[i]);z.ok&=(*b)[i]==guard;}}z.ok&=z.art==gold;if(z.art.size()!=gold.size())exit(3);for(size_t i=0;i<gold.size();++i)z.maxerr=std::max(z.maxerr,fabs(z.art[i]-gold[i]));return z;}};
template<class T>CaseBase*make(int mode,int iw,int ih,int ow,int oh,int C,int rep,const std::vector<double>&pre,size_t&pos,const std::vector<double>&gold,size_t&gp){size_t ni=C*iw*ih,no=C*ow*oh,ng=2*(256/sizeof(T)+16);std::vector<T>h(ni);for(T&v:h)v=(T)pre[pos++];std::vector<double>g(gold.begin()+gp,gold.begin()+gp+ni+no+ng);gp+=ni+no+ng;return new Case<T>(mode,iw,ih,ow,oh,C,rep,h,g);}
int main(){auto pre=load(FIXTURE,28417),gold=load(ORACLE,105568);if(pre[0]!=24)return 3;size_t pos=1,gp=0;std::vector<CaseBase*>cases;for(int path=0;path<24;++path){int bits=pre[pos++],mode=pre[pos++],iw=pre[pos++],ih=pre[pos++],ow=pre[pos++],oh=pre[pos++],C=pre[pos++],rep=pre[pos++];CaseBase*c;if(bits==8)c=make<unsigned char>(mode,iw,ih,ow,oh,C,rep,pre,pos,gold,gp);else if(bits==16)c=make<unsigned short>(mode,iw,ih,ow,oh,C,rep,pre,pos,gold,gp);else if(bits==32)c=make<unsigned int>(mode,iw,ih,ow,oh,C,rep,pre,pos,gold,gp);else return 3;cases.push_back(c);}if(pos!=pre.size()||gp!=gold.size())return 3;for(auto c:cases)c->execute();
 for(int trial=1;trial<=5;++trial){auto art=pre;std::vector<double>times;bool ok=true;double total=0,ma=0;for(auto c:cases){auto x=c->execute();ok&=x.ok;art.insert(art.end(),x.art.begin(),x.art.end());times.push_back(x.seconds);total+=x.seconds;ma=std::max(ma,x.maxerr);}std::string path="output_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)art.data(),art.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum24_typed_nearest_bilinear_corrected_reference_paths_3repeats\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",trial,total,ok?"true":"false",art.size(),ma,path.c_str(),art.size());for(size_t i=0;i<times.size();++i)printf("%s%.17g",i?",":"",times[i]);printf("],\"source_grid_blocks\":29184,\"source_threads_per_block\":256,\"native_reference_repaired\":true}\n");
 }
 return 0;
}
