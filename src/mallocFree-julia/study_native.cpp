
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <dlfcn.h>
#include "/home/imo/HeCBench_julia/src/mallocFree-julia/study_runtime.cpp"
int main(){
 char path[1024]={0};int ver=0;int init=study_init(path,1024,&ver);printf("STUDY_RUNTIME {\"loaded_path\":\"%s\",\"runtime_version\":%d,\"init_status\":%d,\"device_kernels\":0}\n",path,ver,init);fflush(stdout);if(init)return 1;
 const int count=500;std::vector<void*>p(count);std::vector<int>ac(count),fr(count);std::vector<unsigned char>nn(count);double times[2];
 for(int type=0;type<3;type++)for(int i=0;i<19;i++){int s=study_batch(type,size_t(1)<<(i+6),1,p.data(),ac.data(),nn.data(),fr.data(),times);if(s)return 1;}
 for(int trial=1;trial<=5;trial++){
  bool ok=true;double total=0;std::string trace="[";std::vector<double>components;
  for(int type=0;type<3;type++)for(int i=0;i<19;i++){
   size_t bytes=size_t(1)<<(i+6);int s=study_batch(type,bytes,count,p.data(),ac.data(),nn.data(),fr.data(),times);ok&=s==0;total+=times[0]+times[1];components.push_back(times[0]);components.push_back(times[1]);
   if(trace.size()>1)trace+=",";trace+="{\"type\":"+std::to_string(type)+",\"bytes\":"+std::to_string(bytes)+",\"batch_status\":"+std::to_string(s)+",\"allocation_return_codes\":[";
   for(int j=0;j<count;j++){if(j)trace+=",";trace+=std::to_string(ac[j]);}trace+="],\"nonnull_allocations\":[";
   for(int j=0;j<count;j++){if(j)trace+=",";trace+=nn[j]?"true":"false";}trace+="],\"free_return_codes\":[";
   for(int j=0;j<count;j++){if(j)trace+=",";trace+=std::to_string(fr[j]);}trace+="]}";
  }trace+="]";
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"execution_checked\":true,\"checked_values\":0,\"max_abs_error\":null,\"max_rel_error\":null,\"source_empty_kernel\":false,\"source_no_device_kernels\":true,\"no_numerical_outputs\":true,\"no_output_reason\":\"runtime_allocation_lifecycle\",\"runtime_api_status_checked\":true,\"nonnull_allocations_checked\":true,\"allocation_lifecycle_checked\":true,\"jit_excluded\":true,\"timing_region\":\"sum114_mean_host_runtime_alloc_free_loop_intervals\",\"internal_repeats\":500,\"allocated_objects_per_trial\":28500,\"freed_objects_per_trial\":28500,\"component_time_seconds\":[",trial,total,ok?"true":"false");for(int i=0;i<114;i++)printf("%s%.17g",i?",":"",components[i]);printf("],\"runtime_api_status_trace\":%s}\n",trace.c_str());fflush(stdout);if(!ok)return 1;
 }
}
