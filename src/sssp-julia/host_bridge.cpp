// Common source CPU-only frontier work. No CUDA kernels or runtime calls here.
#include "../sssp-cuda/kernel.h"
#include <thread>
extern "C" void sssp_first(Node*n,Edge*e,int*c,int*col,int*q1,int*q2,int*s){
 auto cost=(std::atomic_int*)c;auto color=(std::atomic_int*)col;auto tail=(std::atomic_int*)(s+2);s[0]=1;
 for(int k=0;k<s[0];k++){int pid=q1[k];color[pid].store(BLACK);int cur=cost[pid].load();for(int j=n[pid].x;j<n[pid].x+n[pid].y;j++){int id=e[j].x;cost[id].store(e[j].y+cur);color[id].store(GRAY0);int at=tail->fetch_add(1);q2[at]=id;}}
 s[0]=tail->load();tail->store(0);s[6]++;s[4]=GRAY1;s[5]++;
}
extern "C" void sssp_host_level(Node*n,Edge*e,int*c,int*col,int*q1,int*q2,int*s,int limit){
 int *qin=s[5]%2==0?q1:q2,*qout=s[5]%2==0?q2:q1;
 std::thread outer(run_cpu_threads,n,e,(std::atomic_int*)c,(std::atomic_int*)col,qin,qout,s,(std::atomic_int*)(s+1),(std::atomic_int*)(s+2),(std::atomic_int*)(s+7),(std::atomic_int*)(s+6),(std::atomic_int*)(s+4),(std::atomic_int*)(s+5),1,limit,1);outer.join();
 s[0]=s[2];s[2]=0;s[1]=0;s[4]=s[5]%2==0?GRAY0:GRAY1;
}
