#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#define main preserved_original_main
#include "../pointerchase-cuda/main.cu"
#undef main
#ifndef DIAGDIR
#error DIAGDIR required
#endif
__global__ void diagnostic(LatencyNode* data,uint64_t* out){
 unsigned counts[4]={0,1,17,1000000};unsigned i=threadIdx.x;
 if(i<4){LatencyNode*p=data;for(unsigned j=0;j<counts[i];++j)p=p->next;out[i]=(uint64_t)p;}
}
int main(){
 static_assert(sizeof(LatencyNode)==8,"64-bit pointer layout");
 const uint64_t n=_2MiB/8;LatencyNode* data;uint64_t*out;
 GPU_CHECK(cudaMalloc(&data,_2MiB));GPU_CHECK(cudaMalloc(&out,32));
 initBuffer(data,_2MiB,true);uint64_t base=(uint64_t)data;
 std::vector<uint64_t> h(n),got(n);for(uint64_t i=0;i<n;++i)h[i]=base+8*((i+strideLen)%n);
 cudaDeviceProp prop;GPU_CHECK(cudaGetDeviceProperties(&prop,0));unsigned sm=prop.multiProcessorCount;
 latencyPtrChaseKernel(data,_2MiB,latencyMemAccessCnt,sm);diagnostic<<<1,4>>>(data,out);GPU_CHECK(cudaDeviceSynchronize());
 for(int trial=1;trial<=5;++trial){
  GPU_CHECK(cudaMemcpy(got.data(),data,_2MiB,cudaMemcpyDeviceToHost));bool ok=got==h;
  std::vector<double> ts;double total=0;
  for(unsigned target=0;target<sm;++target){auto start=std::chrono::steady_clock::now();ptrChasingKernel<<<sm,1>>>(data,_2MiB,latencyMemAccessCnt,target);GPU_CHECK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();ts.push_back(dt);total+=dt;}
  GPU_CHECK(cudaMemcpy(got.data(),data,_2MiB,cudaMemcpyDeviceToHost));bool graph=got==h;ok&=graph;
  diagnostic<<<1,4>>>(data,out);GPU_CHECK(cudaDeviceSynchronize());uint64_t cursors[4],counts[4]={0,1,17,1000000};GPU_CHECK(cudaMemcpy(cursors,out,32,cudaMemcpyDeviceToHost));for(int i=0;i<4;++i)ok&=cursors[i]==base+8*((counts[i]*strideLen)%n);
  std::string path=std::string(DIAGDIR)+"/trial"+std::to_string(trial)+"_raw_graph_cursor.bin";std::ofstream f(path,std::ios::binary);f.write((char*)&base,8);f.write((char*)got.data(),_2MiB);f.write((char*)cursors,32);f.close();
  printf("POINTER_DIAGNOSTIC {\"trial\":%d,\"path\":\"%s\",\"bytes\":2097192,\"pointer_base\":%llu,\"cursor_offsets_bytes\":[%llu,%llu,%llu,%llu],\"graph_nodes_checked\":262144,\"timed\":false}\n",trial,path.c_str(),(unsigned long long)base,(unsigned long long)(cursors[0]-base),(unsigned long long)(cursors[1]-base),(unsigned long long)(cursors[2]-base),(unsigned long long)(cursors[3]-base));
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":0,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_all_SMcount_target_block_launches_and_completions\",\"jit_excluded\":true,\"internal_repeats\":1,\"source_empty_kernel\":false,\"source_no_observable_numerical_outputs\":true,\"no_output_reason\":\"source_final_pointer_assertion_only\",\"source_final_pointer_assertion_checked\":true,\"pointer_graph_checked\":%s,\"device_completion_checked\":true,\"nonnull_final_pointer_checked\":true,\"pointer_storage_bits\":64,\"execution_checked\":true,\"numerical_agreement_claim\":false,\"numeric_outputs\":[],\"numerical_outputs\":[],\"target_blocks\":%u,\"accesses_per_selected_block\":1000000,\"buffer_bytes\":2097152,\"host_inclusive_ns_per_access\":%.17g,\"path_times_seconds\":[",trial,total,ok?"true":"false",graph?"true":"false",sm,total*1e9/(latencyMemAccessCnt*sm));for(unsigned i=0;i<sm;++i)printf("%s%.17g",i?",":"",ts[i]);printf("]}\n");
 }
 GPU_CHECK(cudaFree(out));GPU_CHECK(cudaFree(data));return 0;
}
