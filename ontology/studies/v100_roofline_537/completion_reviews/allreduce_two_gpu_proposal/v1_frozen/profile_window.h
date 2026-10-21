#pragma once
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
static int prof_trial=0,prof_calls=0,prof_candidates=0,prof_enters=0,prof_exits=0;
static bool prof_active=false;
static void prof_need(bool yes,const char*why){if(!yes){fprintf(stderr,"ALLREDUCE_PROFILE_GUARD %s\n",why);fflush(stderr);abort();}}
static void prof_cuda(cudaError_t e){prof_need(e==cudaSuccess,cudaGetErrorString(e));}
static bool prof_same_pci(const char*a,const char*b){unsigned ad,ab,aa,af,bd,bb,ba,bf;return a&&b&&sscanf(a,"%x:%x:%x.%x",&ad,&ab,&aa,&af)==4&&sscanf(b,"%x:%x:%x.%x",&bd,&bb,&ba,&bf)==4&&ad==bd&&ab==bb&&aa==ba&&af==bf;}
extern "C" void profile_set_trial(int trial){prof_need(trial>=1&&trial<=5,"trialrange");prof_trial=trial;}
extern "C" void profile_before_add(int n,int rank,int offset,int count){
 prof_need(rank==0||rank==1,"rankrange");prof_need(!prof_active,"nestedwindow");++prof_calls;
 if(rank==1&&prof_trial==2&&n==1048576){
  ++prof_candidates;
  if(prof_candidates==1){
   prof_need(offset==0&&count==524288,"rank1chunk");prof_need(prof_enters==0&&prof_exits==0,"windowoncebefore");
   int device=-1;char pci[64]={};prof_cuda(cudaGetDevice(&device));prof_need(device==1,"logicaldevice1");prof_cuda(cudaDeviceGetPCIBusId(pci,sizeof(pci),device));prof_need(prof_same_pci(pci,getenv("STUDY_EXPECTED_GPU6_PCI")),"physicalGPU6PCI");
   prof_cuda(cudaDeviceSynchronize());
   printf("ALLREDUCE_PROFILE_BEGIN {\"rank\":1,\"trial\":2,\"n\":1048576,\"repeat\":1,\"offset\":0,\"count\":524288,\"pid\":%d}\n",int(getpid()));fflush(stdout);
   prof_cuda(cudaProfilerStart());prof_active=true;++prof_enters;
  }
 }
}
extern "C" void profile_after_add(){
 if(prof_active){prof_cuda(cudaDeviceSynchronize());prof_active=false;prof_cuda(cudaProfilerStop());++prof_exits;
  prof_need(prof_enters==1&&prof_exits==1,"windowonceafter");printf("ALLREDUCE_PROFILE_END {\"rank\":1,\"trial\":2,\"n\":1048576,\"repeat\":1,\"authored_add_dispatches\":1,\"pid\":%d}\n",int(getpid()));fflush(stdout);
 }
}
extern "C" void profile_finish(int rank){
 prof_need(prof_trial==5&&prof_calls==96&&!prof_active,"fullsixcasefivecheckscontroller");
 prof_need(rank==1?(prof_candidates==3&&prof_enters==1&&prof_exits==1):(prof_candidates==0&&prof_enters==0&&prof_exits==0),"finalmarkerbranchcount");
 printf("ALLREDUCE_PROFILE_DONE {\"rank\":%d,\"ring_add_calls\":96,\"candidates\":%d,\"enters\":%d,\"exits\":%d,\"pid\":%d}\n",rank,prof_candidates,prof_enters,prof_exits,int(getpid()));fflush(stdout);
}
