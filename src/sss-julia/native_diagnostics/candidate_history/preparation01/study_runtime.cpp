// Common retained source CPU scientific controller; no device kernel definition.
#include <vector>
#include <map>
#include <array>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#define main original_sss_main
#include "configured/main.cu"
#undef main
#undef SSS
namespace fs=std::filesystem;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
SSSCallback callback;
struct Allocation{void*base;size_t bytes;};std::map<void*,Allocation> allocations;
std::vector<short> host_in,host_out;
struct GraphRecord{int n;std::vector<uint64_t> edges;};std::vector<GraphRecord> graphs;
struct Record{int op,tasks;size_t input,inputlen,output,graph;};std::vector<Record> records;
short *input_arena,*output_arena;size_t used_in=0,used_out=0;
constexpr size_t input_capacity=512ULL*1024*1024,output_capacity=64ULL*1024*1024; // shorts; bounded before capture
cudaEvent_t delete_ready;std::chrono::steady_clock::time_point timer;double seconds;
long checked_gates=0;size_t guards_checked=0;
template<class T>void raw(std::vector<short>&v,const T*p,size_t n){static_assert(sizeof(T)%2==0);size_t k=v.size();v.resize(k+n*sizeof(T)/2);std::memcpy(v.data()+k,p,n*sizeof(T));}
template<class T>void scalar(std::vector<short>&v,T x){raw(v,&x,1);}
void finite(float x){require(std::isfinite(x),"nonfinite source scientific state");}
void graph_state(std::vector<short>&v,Graph*g){int n=g->nVertices;scalar(v,n);for(int i=0;i<n;i++){uint64_t row=0;for(int j=0;j<n;j++){int e=g->Edge[i][j];require(e==0||e==1,"nonbinary graph");require(e==g->Edge[j][i],"asymmetric graph");require(i!=j||!e,"graph diagonal");row|=uint64_t(e)<<j;}scalar(v,row);}}
void state(std::vector<short>&v,State s){scalar(v,s->n);scalar(v,s->p);scalar(v,int(s->L));finite(s->plp);finite(s->alpha);scalar(v,s->plp);scalar(v,s->alpha);raw(v,s->xi,s->n);std::vector<int>counts(s->L);for(int i=0;i<s->n;i++){require(s->xi[i]>=0&&s->xi[i]<s->L,"partition label");counts[s->xi[i]]++;}for(int l=0;l<s->L;l++){finite(s->pll[l]);scalar(v,s->pll[l]);graph_state(v,s->graphlist[l]);}}
void list(std::vector<short>&v,List l){scalar(v,l->n);scalar(v,l->p);scalar(v,int(l->M));raw(v,l->L_list,l->M);raw(v,l->xi_list,l->M*l->n);raw(v,l->edge_list,size_t(l->M)*l->n*l->p*(l->p-1)/2);for(int i=0;i<l->M;i++)finite(l->score_list[i]);raw(v,l->score_list,l->M);scalar(v,int(l->ind_minOfListScores));scalar(v,l->minOfListScores);}
// Independent maximum-cardinality-search chordality criterion on bit adjacency.
bool chordal(std::vector<uint64_t>a,int n){int weight[64]={},order[64]={};uint64_t done=0;for(int k=0;k<n;k++){int v=-1;for(int i=0;i<n;i++)if(!(done>>i&1)&&(v<0||weight[i]>weight[v]))v=i;uint64_t previous=a[v]&done;if(previous){int parent=-1;for(int i=0;i<n;i++)if(previous>>i&1)if(parent<0||order[i]>order[parent])parent=i;require(parent>=0,"MCS parent");if((previous&~(uint64_t(1)<<parent))&~a[parent])return false;}order[v]=k;done|=uint64_t(1)<<v;for(int i=0;i<n;i++)if(a[v]>>i&1)weight[i]++;}return true;}
void validate(const std::vector<short>&inputs,const std::vector<short>&outputs){
 std::map<std::vector<uint64_t>,std::vector<unsigned char>> cache;
 for(const auto&r:records){auto&g=graphs.at(r.graph);const short*p=inputs.data()+r.input;int n=g.n;require(n>0&&n<=63,"graph size scope");auto it=cache.find(g.edges);if(it==cache.end()){require(chordal(g.edges,n),"source graph is nonchordal");std::vector<unsigned char>truth(n*n);for(int a=0;a<n;a++)for(int b=a+1;b<n;b++){auto e=g.edges;e[a]^=uint64_t(1)<<b;e[b]^=uint64_t(1)<<a;truth[a*n+b]=truth[b*n+a]=chordal(e,n);}it=cache.emplace(g.edges,std::move(truth)).first;}
 const short *a,*b;int nc=0;const short*dims=nullptr,*cliques=nullptr;
 if(r.op==0){require(p[0]==n,"delete dimension");nc=p[1];dims=p+2;cliques=dims+nc;const short*t=cliques+nc*n;require(t[0]==r.tasks,"delete task count");a=t+1;b=a+r.tasks;}
 else{const short*t=p+n;int ns=*t++;t+=ns+ns*n;int ne=*t++;t+=2*ne;for(int i=0;i<n;i++)for(int j=0;j<n;j++)require(t[i*n+j]==int(g.edges[i]>>j&1),"actual add device graph differs from source graph");t+=n*n;require(*t++==r.tasks,"add task count");a=t;b=a+r.tasks;}
 for(int k=0;k<r.tasks;k++){require(a[k]>=0&&a[k]<n&&b[k]>=0&&b[k]<n,"proposal endpoint");short observed=outputs[r.output+k];bool expected=it->second[a[k]*n+b[k]];require((r.op==0?observed!=-1:observed!=0)==expected,"independent MCS edge gate disagreement");if(r.op==0){int count=0,index=-1;for(int c=0;c<nc;c++){bool ca=false,cb=false;require(dims[c]>=0&&dims[c]<=n,"clique dimension");for(int j=0;j<dims[c];j++){ca|=cliques[c*n+j]==a[k];cb|=cliques[c*n+j]==b[k];}if(ca&&cb){count++;index=c;}}require(observed==(count==1?index:-1),"delete clique index disagreement");}else require(observed==0||observed==1,"nonboolean add output");checked_gates++;}
 }
}
}
cudaError_t sss_malloc(void**p,size_t n){void*b=nullptr;auto e=cudaMalloc(&b,n+128);if(e!=cudaSuccess)return e;ck(cudaMemset(b,0x5a,n+128));*p=static_cast<char*>(b)+64;ck(cudaMemset(*p,0,n));allocations[*p]={b,n};return cudaSuccess;}
cudaError_t sss_free(void*p){auto a=allocations.at(p);short guards[64];ck(cudaMemcpy(guards,a.base,64,cudaMemcpyDeviceToHost));ck(cudaMemcpy(guards+32,static_cast<char*>(p)+a.bytes,64,cudaMemcpyDeviceToHost));for(short x:guards)require(x==short(0x5a5a),"device redzone changed");raw(host_out,guards,64);guards_checked+=64;allocations.erase(p);return cudaFree(a.base);}
void sss_launch(int op,int tasks,int shmem,cudaStream_t stream,void*del,void*add,void*out,size_t nd,size_t na){
 size_t len=op?na:nd;require(used_in+len<=input_capacity&&used_out+size_t(tasks)<=output_capacity,"capture arena capacity");records.push_back({op,tasks,used_in,len,used_out,graphs.size()-1});
 if(op==1)ck(cudaStreamWaitEvent(stream,delete_ready,0));
 if(len)ck(cudaMemcpyAsync(input_arena+used_in,op?add:del,len*2,cudaMemcpyDeviceToDevice,stream));
 if(op==0)ck(cudaEventRecord(delete_ready,stream));
 if(tasks){ck(cudaMemsetAsync(out,0x80,allocations.at(out).bytes,stream));int status=callback(op,tasks,shmem,stream,del,add,out,nd,na);require(status==0,"GPU callback failed");ck(cudaGetLastError());ck(cudaMemcpyAsync(output_arena+used_out,out,tasks*2,cudaMemcpyDeviceToDevice,stream));}
 used_in+=len;used_out+=tasks;
}
void sss_observe_graph(void*ptr){Graph*g=static_cast<Graph*>(ptr);GraphRecord r{int(g->nVertices),{}};for(int i=0;i<r.n;i++){uint64_t bits=0;for(int j=0;j<r.n;j++){int x=g->Edge[i][j];require(x==0||x==1,"graph input nonbinary");bits|=uint64_t(x)<<j;}r.edges.push_back(bits);}graphs.push_back(std::move(r));}
void sss_observe_results(int,int,const short*){} // Complete ordered device snapshots checked after ROI, no fabricated host substitution.
void sss_observe_data(const float*x,int n,int p){scalar(host_in,n);scalar(host_in,p);for(int i=0;i<n*p;i++)finite(x[i]);raw(host_in,x,n*p);}
void sss_observe_iteration(void*s,long k,float score,float local,float global,int modes,int cases,long total){scalar(host_out,k);finite(score);finite(local);finite(global);scalar(host_out,score);scalar(host_out,local);scalar(host_out,global);scalar(host_out,modes);scalar(host_out,cases);scalar(host_out,total);state(host_out,static_cast<State>(s));}
void sss_observe_final(void*a,void*b,void*c,void*d,void*e,long k,long total){scalar(host_out,k);scalar(host_out,total);state(host_out,static_cast<State>(a));state(host_out,static_cast<State>(b));state(host_out,static_cast<State>(c));list(host_out,static_cast<List>(d));list(host_out,static_cast<List>(e));}
void sss_start_timer(){ck(cudaDeviceSynchronize());timer=std::chrono::steady_clock::now();}
void sss_stop_timer(){ck(cudaDeviceSynchronize());seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-timer).count();}
extern "C" int sss_run(const char*data_path,const char*out_path,SSSCallback cb,int trials){
 try{callback=cb;fs::create_directories("RES");if(!fs::exists("DATA"))fs::create_directory_symlink(fs::absolute(data_path),"DATA");fs::create_directories(out_path);ck(cudaMalloc(&input_arena,input_capacity*2));ck(cudaMalloc(&output_arena,output_capacity*2));ck(cudaEventCreateWithFlags(&delete_ready,cudaEventDisableTiming));
 for(int trial=0;trial<=trials;trial++){used_in=used_out=0;host_in.clear();host_out.clear();graphs.clear();records.clear();checked_gates=0;guards_checked=0;char name[]="f9_n150_p50";char prog[]="sss_study";char*args[]={prog,name,nullptr};require(original_sss_main(2,args)==0,"source controller exit");ck(cudaDeviceSynchronize());std::vector<short>inputs(used_in),outputs(used_out);ck(cudaMemcpy(inputs.data(),input_arena,used_in*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(outputs.data(),output_arena,used_out*2,cudaMemcpyDeviceToHost));validate(inputs,outputs);require(allocations.empty()&&guards_checked==256,"allocation lifecycle/redzones");
 fs::path dir=fs::path(out_path)/("trial"+std::to_string(trial));fs::create_directories(dir);for(auto &f:fs::directory_iterator("RES"))if(f.is_regular_file())fs::copy_file(f.path(),dir/f.path().filename(),fs::copy_options::overwrite_existing);
 // Lossless input prefix consists of source X and every actual device packed input. Outputs remain actual Int16 plus bit-encoded scientific host states.
 fs::path artifact=dir/"capture_i16.bin";std::ofstream f(artifact,std::ios::binary);f.write((char*)host_in.data(),host_in.size()*2);f.write((char*)inputs.data(),inputs.size()*2);f.write((char*)outputs.data(),outputs.size()*2);f.write((char*)host_out.data(),host_out.size()*2);require(bool(f),"capture write failed");f.close();
 std::ofstream layout(dir/"layout.json");layout<<"{\"host_input_shorts\":"<<host_in.size()<<",\"device_input_shorts\":"<<inputs.size()<<",\"device_output_shorts\":"<<outputs.size()<<",\"host_output_shorts\":"<<host_out.size()<<",\"graphs\":"<<graphs.size()<<",\"launches\":"<<records.size()<<",\"checked_gates\":"<<checked_gates<<",\"guards\":"<<guards_checked<<"}";
 std::ofstream meta(dir/"records_i64.bin",std::ios::binary);for(auto&r:records){uint64_t row[]={uint64_t(r.op),uint64_t(r.tasks),r.input,r.inputlen,r.output,r.graph};meta.write((char*)row,sizeof(row));}for(auto&g:graphs){uint64_t n=g.n;meta.write((char*)&n,8);meta.write((char*)g.edges.data(),g.edges.size()*8);}
 if(trial>0)printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"timing_valid\":true,\"jit_excluded\":true,\"region\":\"full_source_search_with_ordered_capture_and_host_state_observation\",\"timing_region\":\"full_source_search_with_ordered_capture_and_host_state_observation\",\"checked_values\":%zu,\"max_abs_error\":0,\"max_rel_error\":null,\"input_prefix_count\":%zu,\"artifact\":{\"path\":\"%s\",\"dtype\":\"int16\",\"count\":%zu,\"byteorder\":\"little\"}}\n",trial,seconds,host_in.size()+inputs.size()+outputs.size()+host_out.size(),host_in.size()+inputs.size(),fs::absolute(artifact).c_str(),host_in.size()+inputs.size()+outputs.size()+host_out.size());else fprintf(stderr,"SSS_WARMUP_CAPTURE %s gates=%ld bytes=%zu\n",fs::absolute(artifact).c_str(),checked_gates,(host_in.size()+inputs.size()+outputs.size()+host_out.size())*2);fflush(stdout);
 }
 ck(cudaEventDestroy(delete_ready));ck(cudaFree(input_arena));ck(cudaFree(output_arena));return 0;
 }catch(const std::exception&e){fprintf(stderr,"SSS_STUDY_FAILURE %s\n",e.what());return 1;}
}
