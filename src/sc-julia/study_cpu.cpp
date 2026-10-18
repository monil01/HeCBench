// Source CPU compaction and partitioner reused unchanged as explicit host component in all models.
#include "../sc-cuda/host_sc.cpp"
#include <cstring>
#include <cstdlib>
#include <cstdint>
static_assert(sizeof(std::atomic_int)==sizeof(int),"sourceatomicInt32ABI");
struct SCHost{int padded,tasks;int*base;std::atomic_int*fb;int*data;std::atomic_int*flags;std::thread*thread;};
extern "C" void* sc_host_create(int padded,int tasks){auto*h=new SCHost{padded,tasks,nullptr,nullptr,nullptr,nullptr,nullptr};h->base=new int[padded+2];h->data=h->base+1;h->fb=new std::atomic_int[tasks+3];h->flags=h->fb+1;return h;}
extern "C" int* sc_host_data(void*ptr){return ((SCHost*)ptr)->data;}
extern "C" int* sc_host_flags(void*ptr){return reinterpret_cast<int*>(((SCHost*)ptr)->flags);}
extern "C" void sc_host_reset(void*ptr,const int*input,int n,int cut){auto*h=(SCHost*)ptr;std::fill(h->base,h->base+h->padded+2,0);h->base[0]=h->base[h->padded+1]=-777;std::copy(input,input+n,h->data);for(int i=0;i<h->tasks+3;i++)h->fb[i].store(0);h->fb[0].store(-777);h->fb[h->tasks+2].store(-777);h->flags[0].store(1);h->flags[cut].store(1);}
extern "C" void sc_host_start(void*ptr,int n,float alpha){auto*h=(SCHost*)ptr;h->thread=new std::thread([h,n,alpha]{run_cpu_threads(h->data,h->data,h->flags,n,0,4,256,h->tasks,alpha);});}
extern "C" void sc_host_wait(void*ptr){auto*h=(SCHost*)ptr;h->thread->join();delete h->thread;h->thread=nullptr;}
extern "C" void sc_host_destroy(void*ptr){auto*h=(SCHost*)ptr;if(h->thread)std::abort();delete[]h->base;delete[]h->fb;delete h;}
