#include "cpu_bridge.h"
#include "../pad-cuda/kernel.h"
#include <thread>
#include <vector>
#include <cstring>
#include <cassert>
struct PadContext {int m,n,pad,size,nt,cut;float alpha; std::vector<double> data,backup;std::atomic_int*flags;std::thread worker;
PadContext(int mm,int nn,int pp,float aa,const uint64_t*initial):m(mm),n(nn),pad(pp),size(m*(n+pad)),nt(size/8192),cut(int(nt*aa)),alpha(aa),data(size+16,-12345.),backup(size),flags(new std::atomic_int[nt+1]){assert(size%8192==0);memcpy(backup.data(),initial,size*8);static_assert(sizeof(std::atomic_int)==4);assert(flags[0].is_lock_free());}
~PadContext(){delete[]flags;}};
extern "C" void*pad_create(int m,int n,int p,float a,const uint64_t*x){return new PadContext(m,n,p,a,x);}
extern "C" void pad_destroy(void*v){delete (PadContext*)v;}
extern "C" double*pad_data(void*v){return ((PadContext*)v)->data.data()+8;}
extern "C" double*pad_all_data(void*v){return ((PadContext*)v)->data.data();}
extern "C" int*pad_flags(void*v){return reinterpret_cast<int*>(((PadContext*)v)->flags);}
extern "C" void pad_reset(void*v){auto*c=(PadContext*)v;memcpy(c->data.data()+8,c->backup.data(),c->size*8);for(int i=0;i<=c->nt;i++)c->flags[i].store(0);c->flags[0].store(1);c->flags[c->cut].store(1);}
extern "C" void pad_start(void*v){auto*c=(PadContext*)v;c->worker=std::thread([c](){run_cpu_threads(c->data.data()+8,c->data.data()+8,c->flags,c->n,c->m,c->pad,4,256,c->nt,c->alpha);});}
extern "C" void pad_join(void*v){((PadContext*)v)->worker.join();}
