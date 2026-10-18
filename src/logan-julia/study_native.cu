#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include "../logan-cuda/src/seed.cu"
#include "../logan-cuda/src/score.cu"
#include "../logan-cuda/src/logan_functions.cu"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<int>load(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);size_t z=f.tellg();std::vector<int>a(z/4);f.seekg(0);f.read((char*)a.data(),z);return a;}
template<class T>struct Buf{std::vector<T>h;T*raw;T*ptr;T guard;Buf(std::vector<T>v,T g):h(v.size()+8,g),guard(g){std::copy(v.begin(),v.end(),h.begin()+4);ck(cudaMalloc(&raw,h.size()*sizeof(T)));ptr=raw+4;reset();}void reset(){ck(cudaMemcpy(raw,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>a(h.size());ck(cudaMemcpy(a.data(),raw,a.size()*sizeof(T),cudaMemcpyDeviceToHost));return a;}bool guards(std::vector<T>&a){for(int j=0;j<4;j++)if(a[j]!=guard||a[a.size()-1-j]!=guard)return false;return true;}};
int main(){auto f=load(LOGAN_FIXTURE),oracle=load(LOGAN_ORACLE);int n=f[0],pos=1,off=0,drop=100;std::vector<int>seeds,oq,ot;std::vector<char>q,t;
for(int i=0;i<n;i++){int nq=f[pos++],nt=f[pos++];drop=f[pos++];seeds.insert(seeds.end(),f.begin()+pos,f.begin()+pos+10);pos+=10;for(int j=0;j<nq;j++)q.push_back(char(f[pos++]));for(int j=0;j<nt;j++)t.push_back(char(f[pos++]));oq.push_back(q.size());ot.push_back(t.size());off=std::max(off,std::max(nq,nt)+2);}
// Source chars use byte storage; artifact canonical Int32 encoding preserves exact unsigned ASCII.
Buf<char>Q(q,77),T(t,77);Buf<int>OQ(oq,123456789),OT(ot,123456789);Buf<int>S1(seeds,123456789),S2(seeds,123456789),R1(std::vector<int>(n),123456789),R2(std::vector<int>(n),123456789);Buf<short>A1(std::vector<short>(3*off*n),12345),A2(std::vector<short>(3*off*n),12345);cudaStream_t streams[2];for(auto&s:streams)ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
auto run=[&](){extendSeedLGappedXDropOneDirectionGlobal<<<n,128,256,streams[0]>>>((SeedL*)S1.ptr,Q.ptr,T.ptr,EXTEND_LEFTL,drop,R1.ptr,OQ.ptr,OT.ptr,off,A1.ptr,128);extendSeedLGappedXDropOneDirectionGlobal<<<n,128,256,streams[1]>>>((SeedL*)S2.ptr,Q.ptr,T.ptr,EXTEND_RIGHTL,drop,R2.ptr,OQ.ptr,OT.ptr,off,A2.ptr,128);};run();ck(cudaDeviceSynchronize());
for(int trial=1;trial<=5;trial++){S1.reset();S2.reset();R1.reset();R2.reset();A1.reset();A2.reset();ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();run();ck(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto payload=f;bool ok=true;
for(auto*b:{&Q,&T}){auto a=b->get();ok&=b->guards(a)&&a==b->h;for(size_t j=0;j<a.size();j++)payload.push_back(j<4||j>=a.size()-4?123456789:int(a[j]));}
for(auto*b:{&OQ,&OT}){auto a=b->get();ok&=b->guards(a)&&a==b->h;payload.insert(payload.end(),a.begin(),a.end());}
for(int dir=1;dir<=2;dir++){auto&sb=dir==1?S1:S2;auto&rb=dir==1?R1:R2;auto&ab=dir==1?A1:A2;auto s=sb.get(),v=rb.get();auto a=ab.get();ok&=sb.guards(s)&&rb.guards(v)&&ab.guards(a);
for(int i=0;i<n;i++){std::vector<int>e(seeds.begin()+10*i,seeds.begin()+10*i+10);int score=oracle[5*i],col=oracle[5*i+1],row=oracle[5*i+2],lo=oracle[5*i+3],hi=oracle[5*i+4];e[4]=0;e[9]=0;if(dir==1){e[0]-=row;e[1]-=col;e[5]=std::min(e[5],e[7]+lo);e[6]=std::max(e[6],e[7]+hi);}else{e[2]+=row;e[3]+=col;e[6]=std::max(e[6],e[8]-lo);e[5]=std::min(e[5],e[8]-hi);}s[4+10*i+4]=0;for(int j=0;j<10;j++)ok&=s[4+10*i+j]==e[j];ok&=v[4+i]==score;}
payload.insert(payload.end(),s.begin(),s.end());payload.insert(payload.end(),v.begin(),v.end());for(auto x:a)payload.push_back(x);}
std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*4);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"timing_region\":\"two_stream_left_right_kernel_batch_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":2,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",payload.size(),ok?"0":"null",path.c_str(),payload.size());if(!ok)return 3;}
}
