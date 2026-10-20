// Owned full-host workflow observer; original source files stay intact.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <chrono>
#include <sys/stat.h>
#include "../../word2vec-cuda/cbow.h"
constexpr int SEN_WORDS=6150, TRACE_CAP=2048, TRACE_WORDS=2+SEN_WORDS+2048;
unsigned int* trace_data=nullptr;
__device__ unsigned int* trace_execution_counter;
std::atomic<int> trace_submitted{0};
std::string folder; double excluded_seconds=0; int trial_number=0;
struct Pause { std::chrono::steady_clock::time_point t;Pause():t(std::chrono::steady_clock::now()){}~Pause(){excluded_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}};
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAFAIL %s\n",cudaGetErrorString(e));abort();}}
struct Allocation{void* base;size_t bytes;};std::map<void*,Allocation> allocs;
cudaError_t gm(void** p,size_t n){void*b;auto e=cudaMalloc(&b,n+512);if(e!=cudaSuccess)return e;ck(cudaMemset(b,0xa7,n+512));*p=(char*)b+256;allocs[*p]={b,n};return cudaSuccess;}
void writebytes(const std::string&name,const void*p,size_t n){FILE*f=fopen((folder+"/"+name).c_str(),"wb");if(!f){perror("fopen");abort();}if(fwrite(p,1,n,f)!=n)abort();fclose(f);}
void snapshot(const std::string&name,void*p,size_t n){std::vector<unsigned char>b(n);ck(cudaMemcpy(b.data(),p,n,cudaMemcpyDeviceToHost));writebytes(name,b.data(),n);}
cudaError_t gf(void*p){auto a=allocs.at(p);{Pause pause;unsigned char g[512];ck(cudaMemcpy(g,a.base,256,cudaMemcpyDeviceToHost));ck(cudaMemcpy(g+256,(char*)p+a.bytes,256,cudaMemcpyDeviceToHost));for(auto c:g)if(c!=0xa7){fprintf(stderr,"GUARD_FAIL\n");abort();}FILE*f=fopen((folder+"/guards.bin").c_str(),"ab");fwrite(g,1,512,f);fclose(f);}allocs.erase(p);return cudaFree(a.base);}
void observe_initial();void observe_final();
void broker(int kind,int ns,int slot,int blocks);
#define cudaMalloc gm
#define cudaFree gf
#include "target_gpu.inc"
#undef cudaMalloc
#undef cudaFree
#include "observed_host.inc"
void observe_initial(){Pause pause;mkdir(folder.c_str(),0755);auto n=(size_t)vocab_size*layer1_size_aligned;
 snapshot("initial_syn0.f32",d_syn0,n*4);snapshot("initial_syn1.f32",d_syn1neg,n*4);snapshot("initial_rng.u32",d_random,4096);snapshot("initial_sen.i32",d_sen,SEN_WORDS*4);
 std::vector<int> actual(table_size);ck(cudaMemcpy(actual.data(),d_table,(size_t)table_size*4,cudaMemcpyDeviceToHost));if(memcmp(actual.data(),table,(size_t)table_size*4))abort();
 const char*canonical="/tmp/w2vfull/table.i32";FILE*f=fopen(canonical,"rb");if(!f){f=fopen(canonical,"wb");fwrite(actual.data(),4,actual.size(),f);}else{std::vector<int>b(1<<20);size_t at=0;while(at<actual.size()){size_t k=std::min(b.size(),actual.size()-at);if(fread(b.data(),4,k,f)!=k||memcmp(b.data(),actual.data()+at,k*4))abort();at+=k;}}fclose(f);
 FILE*proof=fopen((folder+"/table_full_byte_comparison.json").c_str(),"w");fprintf(proof,"{\"actual_device_bytes_compared\":400000000,\"canonical_path\":\"%s\",\"equal\":true}\n",canonical);fclose(proof);
 float et[1000];ck(cudaMemcpyFromSymbol(et,expTable,sizeof(et)));writebytes("exptable.f32",et,sizeof(et));
 FILE*v=fopen((folder+"/vocab.tsv").c_str(),"w");for(int i=0;i<vocab_size;i++)fprintf(v,"%s\t%d\n",vocab[i].word,vocab[i].cn);fclose(v);
 // Fresh independent one-position kernel-gradient control, then exact restore.
 std::vector<int> us(SEN_WORDS,0);for(int i=0;i<1024;i++)us[i]=i+1;float ua=.05f;memcpy(us.data()+6144,&ua,4);writebytes("unit_sen.i32",us.data(),SEN_WORDS*4);
 std::vector<unsigned int> ur(1024);ck(cudaMemcpy(ur.data(),d_random,4096,cudaMemcpyDeviceToHost));
 ck(cudaMemcpy(d_sen,us.data(),SEN_WORDS*4,cudaMemcpyHostToDevice));
 broker(1,1,0,1);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
 snapshot("unit_syn0.f32",d_syn0,n*4);snapshot("unit_syn1.f32",d_syn1neg,n*4);snapshot("unit_rng.u32",d_random,4096);
 ck(cudaMemcpy(d_syn0,syn0,n*4,cudaMemcpyHostToDevice));ck(cudaMemset(d_syn1neg,0,n*4));ck(cudaMemset(d_sen,0,SEN_WORDS*4));ck(cudaMemcpy(d_random,ur.data(),4096,cudaMemcpyHostToDevice));unsigned int*countp;ck(cudaMemcpyFromSymbol(&countp,trace_execution_counter,sizeof(countp)));ck(cudaMemset(countp,0,4));

}
void observe_final(){ck(cudaDeviceSynchronize());Pause pause;snapshot("target_exptable.f32",target_exp,4000);size_t n=(size_t)vocab_size*layer1_size_aligned;snapshot("final_syn0.f32",d_syn0,n*4);snapshot("final_syn1.f32",d_syn1neg,n*4);snapshot("final_rng.u32",d_random,4096);int calls=trace_submitted.load();if(calls>TRACE_CAP)abort();snapshot("callbacks.u32",trace_data,(size_t)calls*TRACE_WORDS*4);unsigned int count;unsigned int*counter;ck(cudaMemcpyFromSymbol(&counter,trace_execution_counter,sizeof(counter)));ck(cudaMemcpy(&count,counter,4,cudaMemcpyDeviceToHost));if(count!=(unsigned)calls)abort();FILE*f=fopen((folder+"/trace_meta.json").c_str(),"w");fprintf(f,"{\"callbacks\":%d,\"trace_words\":%d,\"cap\":%d,\"execution_count\":%u}\n",calls,TRACE_WORDS,TRACE_CAP,count);fclose(f);}
void reset(){vocab_max_size=1000;vocab_size=0;train_words=0;word_count_actual=0;file_size=0;min_reduce=1;alpha=.025;layer1_size_aligned=0;}
int source_workflow_run(int argc,char**argv){if(argc!=2){fprintf(stderr,"usage: native OUTPUT_ROOT\n");return 2;}std::string base=argv[1];mkdir(base.c_str(),0755);ck(cudaMalloc((void**)&trace_data,(size_t)TRACE_CAP*TRACE_WORDS*4));unsigned int*counter;ck(cudaMalloc((void**)&counter,4));ck(cudaMemcpyToSymbol(trace_execution_counter,&counter,sizeof(counter)));
 for(int t=0;t<=5;t++){trial_number=t;folder=base+"/trial"+std::to_string(t);mkdir(folder.c_str(),0755);reset();trace_submitted=0;ck(cudaMemset(counter,0,4));excluded_seconds=0;std::string out=folder+"/vectors.bin";if(out.size()>=100)abort();
 std::vector<std::string>args={"word2vec","-train","/tmp/w2vfull/corpus.txt","-output",out,"-size","128","-threads","20","-iter","15","-window","8","-negative","25","-sample","0.0001","-binary","1","-cbow","1","-hs","0","-debug","0"};std::vector<char*>av;for(auto&s:args)av.push_back(s.data());
 auto starttime=std::chrono::steady_clock::now();int rc=source_main(av.size(),av.data());ck(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-starttime).count()-excluded_seconds;if(rc)abort();
 FILE*f=fopen((folder+"/timing.json").c_str(),"w");fprintf(f,"{\"trial\":%d,\"time_seconds\":%.17g,\"excluded_observation_seconds\":%.17g}\n",t,sec,excluded_seconds);fclose(f);printf("WORKFLOW_CAPTURE trial=%d calls=%d seconds=%.9g\n",t,trace_submitted.load(),sec);fflush(stdout);
 }
 ck(cudaFree(trace_data));ck(cudaFree(counter));return 0;}

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
struct Request {int kind,slot,ns,blocks,d,a,w,neg,t,v;void*s0;void*s1;void*sen;void*rng;void*table;void*trace;void*counter;void*ex;};
struct Pending{Request r;bool ack=false;std::condition_variable cv;};
std::mutex qmutex;std::deque<Pending*> queue;std::thread workflow;std::atomic<bool> finished{false};std::atomic<int> workflow_rc{-999};
void broker(int kind,int ns,int slot,int blocks){unsigned int*counter;ck(cudaMemcpyFromSymbol(&counter,trace_execution_counter,sizeof(counter)));Pending pending;pending.r={kind,slot,ns,blocks,layer1_size,layer1_size_aligned,window,negative,table_size,vocab_size,d_syn0,d_syn1neg,d_sen,d_random,d_table,trace_data,counter,target_exp};std::unique_lock<std::mutex>lock(qmutex);queue.push_back(&pending);pending.cv.wait(lock,[&]{return pending.ack;});}
extern "C" int wf_start(const char*base){if(workflow.joinable())return -1;finished=false;std::string path=base;workflow=std::thread([path]{char arg0[]="workflow";std::vector<char>p(path.begin(),path.end());p.push_back(0);char*av[]={arg0,p.data()};workflow_rc=source_workflow_run(2,av);finished=true;});return 0;}
extern "C" int wf_poll(Request*out){std::lock_guard<std::mutex>lock(qmutex);if(!queue.empty()){*out=queue.front()->r;return 1;}return finished?-1:0;}
extern "C" void wf_ack(){std::lock_guard<std::mutex>lock(qmutex);auto*p=queue.front();queue.pop_front();p->ack=true;p->cv.notify_one();}
extern "C" int wf_join(){if(workflow.joinable())workflow.join();return workflow_rc;}
