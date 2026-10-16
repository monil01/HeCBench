#define main hecbench_original_main
#include "concurrentKernels-cuda/main.cu"
#undef main
#include <vector>
void run_graph(long*data,int n,long count,cudaStream_t*streams,cudaEvent_t*events){
 for(int slot=0;slot<n;slot++){clock_block<<<1,1,0,streams[slot]>>>(data+slot,count);cudaEventRecord(events[slot],streams[slot]);cudaStreamWaitEvent(streams[n],events[slot],0);}
 sum<<<1,32,0,streams[n]>>>(data,n);cudaStreamSynchronize(streams[n]);
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4;long count=argc>2?atol(argv[2]):120000;if(n<1||n>32)return 2;
 long*data;cudaMalloc(&data,32*sizeof(long));cudaMemset(data,0,32*sizeof(long));cudaStream_t streams[33];cudaEvent_t events[32];
 for(int s=0;s<n+1;s++)cudaStreamCreate(&streams[s]);for(int s=0;s<n;s++)cudaEventCreateWithFlags(&events[s],cudaEventDisableTiming);cudaDeviceSynchronize();
 run_graph(data,n,count,streams,events);long single=count/3*3+(count%3==2?1:0);
 for(int trial=1;trial<=5;trial++){
  cudaMemset(data,0,32*sizeof(long));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();run_graph(data,n,count,streams,events);auto stop=std::chrono::steady_clock::now();double time=std::chrono::duration<double>(stop-start).count();
  std::vector<long>result(n);cudaMemcpy(result.data(),data,n*sizeof(long),cudaMemcpyDeviceToHost);bool correct=true;for(int s=0;s<n;s++)correct &= result[s]==single*(s==0?n:1);
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(result.data(),sizeof(long),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"timing_region\":\"producer_streams_event_join_reduction_fence_no_d2h\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,correct?"true":"false",n,filename,n);if(!correct)return 1;
 }cudaFree(data);for(int s=0;s<n+1;s++)cudaStreamDestroy(streams[s]);for(int s=0;s<n;s++)cudaEventDestroy(events[s]);return 0;
}
