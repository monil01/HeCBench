#define main legacy_attention_main
#include "../attention-paged-cuda/main.cu"
#undef main
#include <fstream>
#include <vector>
#include <type_traits>
#include <spawn.h>
#include <sys/wait.h>
extern char **environ;
#ifndef FIXTURE_ROOT
#define FIXTURE_ROOT "."
#endif
std::vector<double>readfile(const char*name){std::ifstream file(std::string(FIXTURE_ROOT)+"/"+name,std::ios::binary|std::ios::ate);assert(file);size_t n=file.tellg();assert(n%8==0);std::vector<double>out(n/8);file.seekg(0);file.read((char*)out.data(),n);assert(file);return out;}
template<class T>struct Buf{T*base;std::vector<T>initial;int n;Buf(int N,const std::vector<double>*input=nullptr,size_t offset=0):n(N){initial.assign(N+64,T(-37));for(int i=0;i<N;i++)initial[i+32]=input?T((*input)[offset+i]):T(0);GPU_CHECK(cudaMalloc(&base,initial.size()*sizeof(T)));GPU_CHECK(cudaMemcpy(base,initial.data(),initial.size()*sizeof(T),cudaMemcpyHostToDevice));}~Buf(){GPU_CHECK(cudaFree(base));}T*p(){return base+32;}std::vector<T>read(){std::vector<T>a(initial.size());GPU_CHECK(cudaMemcpy(a.data(),base,a.size()*sizeof(T),cudaMemcpyDeviceToHost));return a;}void append(std::vector<double>&out){auto a=read();out.insert(out.end(),a.begin(),a.end());}bool guards(const std::vector<T>&a){return std::equal(a.begin(),a.begin()+32,initial.begin())&&std::equal(a.end()-32,a.end(),initial.end()-32);}void payload(const std::vector<double>&f,size_t off){std::vector<T>a(n);for(int i=0;i<n;i++)a[i]=T(f[off+i]);GPU_CHECK(cudaMemcpy(p(),a.data(),n*sizeof(T),cudaMemcpyHostToDevice));}};
template<int D>void invoke(float*out,float*q,float*k,float*v,float scale,int*table,int*lens,float*alibi,int S,int H,int M,int shared){
 paged_attention_v1_kernel<float,float,D,16,128,false><<<dim3(H,S),128,shared>>>(out,q,k,v,H,scale,table,lens,M,alibi,H*D,H*D*16,D*16,0,0,0,0,0);
}
int main(){auto fixture=readfile("fixture.bin"),oracle=readfile("oracle.bin");for(int trial=0;trial<=5;trial++){std::vector<double>prefix{3},computed,extra;double secs=0,maxerr=0;double case_times[3]={};bool ok=true;size_t off=1,go=0;
 for(int ci=0;ci<3;ci++){auto head=std::vector<double>(fixture.begin()+off,fixture.begin()+off+8);off+=8;prefix.insert(prefix.end(),head.begin(),head.end());int S=head[0],H=head[1],D=head[2],maxlen=head[3],pages=head[4],Q=S*H*D,KV=pages*H*D*16,M=(maxlen+15)/16,T=S*M;float scale=head[6],kvscale=head[7];size_t lensoff=off;off+=S;size_t tableoff=off;off+=T;size_t baseline=off;off+=Q+2*KV+H;size_t actual=off;off+=Q+2*KV+H;
 Buf<float>q(Q),k(KV),v(KV),alibi(H),out(Q);Buf<int>tables(T,&fixture,tableoff),lens(S,&fixture,lensoff);
 for(auto*b:{&q,&k,&v,&alibi,&out})b->append(prefix);tables.append(prefix);lens.append(prefix);
 uniform_fill_kernel<float><<<(Q+255)/256,256>>>(q.p(),Q,-scale,scale,1234);uniform_fill_kernel<float><<<(KV+255)/256,256>>>(k.p(),KV,-kvscale,kvscale,1234);uniform_fill_kernel<float><<<(KV+255)/256,256>>>(v.p(),KV,-kvscale,kvscale,1234);norm_fill_kernel<float><<<1,256>>>(alibi.p(),H,1234);GPU_CHECK(cudaGetLastError());GPU_CHECK(cudaDeviceSynchronize());
 size_t b=baseline;for(auto*x:{&q,&k,&v,&alibi}){auto a=x->read();ok&=x->guards(a);for(int i=0;i<x->n;i++)ok&=double(a[i+32])==fixture[b+i];b+=x->n;prefix.insert(prefix.end(),a.begin(),a.end());}
 b=actual;for(auto*x:{&q,&k,&v,&alibi}){x->payload(fixture,b);b+=x->n;}
 b=actual;std::vector<std::vector<float>>before;for(auto*x:{&q,&k,&v,&alibi,&out}){auto a=x->read();ok&=x->guards(a);if(x!=&out){for(int i=0;i<x->n;i++)ok&=double(a[i+32])==fixture[b+i];b+=x->n;}before.push_back(a);prefix.insert(prefix.end(),a.begin(),a.end());}tables.append(prefix);lens.append(prefix);
 int shared=std::max(M*16,2*D)*4;GPU_CHECK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
 if(D==32)invoke<32>(out.p(),q.p(),k.p(),v.p(),scale,tables.p(),lens.p(),alibi.p(),S,H,M,shared);else if(D==64)invoke<64>(out.p(),q.p(),k.p(),v.p(),scale,tables.p(),lens.p(),alibi.p(),S,H,M,shared);else if(D==128)invoke<128>(out.p(),q.p(),k.p(),v.p(),scale,tables.p(),lens.p(),alibi.p(),S,H,M,shared);else return 2;
 GPU_CHECK(cudaGetLastError());GPU_CHECK(cudaDeviceSynchronize());case_times[ci]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();secs+=case_times[ci];auto a=out.read();ok&=out.guards(a);for(int i=0;i<Q;i++){double e=std::abs(double(a[i+32])-oracle[go+i]);maxerr=std::max(maxerr,e);ok&=std::isfinite(a[i+32])&&e<=2e-5+2e-5*std::abs(oracle[go+i]);}go+=Q;computed.insert(computed.end(),a.begin(),a.end());int i=0;for(auto*x:{&q,&k,&v,&alibi,&out}){auto z=x->read();ok&=x->guards(z);if(x!=&out)ok&=z==before[i];i++;extra.insert(extra.end(),z.begin(),z.end());}auto ti=tables.read(),li=lens.read();ok&=ti==tables.initial&&li==lens.initial;extra.insert(extra.end(),ti.begin(),ti.end());extra.insert(extra.end(),li.begin(),li.end());
 }
 assert(off==fixture.size()&&go==oracle.size()&&prefix.size()==1016993);if(trial==0){if(!ok){fprintf(stderr,"Pagedwarmupfailed abs=%.17g\n",maxerr);return 3;}continue;}
 prefix.insert(prefix.end(),computed.begin(),computed.end());prefix.insert(prefix.end(),extra.begin(),extra.end());assert(prefix.size()==1357921);const char*dir=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream file(path,std::ios::binary);file.write((char*)prefix.data(),prefix.size()*8);assert(file);file.close();pid_t child;std::string script=std::string(FIXTURE_ROOT)+"/audit_native_outputs.py";char*args[]={const_cast<char*>("python3"),const_cast<char*>(script.c_str()),const_cast<char*>(path.c_str()),nullptr};int spawn=posix_spawnp(&child,"python3",nullptr,nullptr,args,environ);int status=1;if(spawn==0)waitpid(child,&status,0);ok&=spawn==0&&WIFEXITED(status)&&WEXITSTATUS(status)==0;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"case_times_seconds\":[%.17g,%.17g,%.17g],\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum3source_FP32_pagedattention_launch_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_GPU_body_unchanged\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,case_times[0],case_times[1],case_times[2],ok?"true":"false",prefix.size(),maxerr,path.c_str(),prefix.size());fflush(stdout);if(!ok)return 4;
 }return 0;}
