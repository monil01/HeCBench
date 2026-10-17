
#define main archived_streamum_main
#include "../streamUM-cuda/main.cu"
#undef main
#include <string>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
void reset_task(Task<double>&t){for(unsigned j=0;j<t.size;j++){t.vector[j]=double(int((5*j+3*t.id)%17)-8)/16.;t.result[j]=0.;for(unsigned i=0;i<t.size;i++)t.data[j*t.size+i]=double(int((13*i+7*j+3*t.id)%31)-15)/32.;}}
int main(){omp_set_dynamic(0);omp_set_num_threads(4);int cma;CK(cudaDeviceGetAttribute(&cma,cudaDevAttrConcurrentManagedAccess,0));if(!cma)return 3;
 std::vector<Task<double>>tasks(8);int ns[]={64,99,100,101,257,511,67,127};std::vector<double>prefix;
 cudaStream_t streams[5];cublasHandle_t handles[5];for(int i=0;i<5;i++){CK(cudaStreamCreate(&streams[i]));if(cublasCreate(&handles[i])!=CUBLAS_STATUS_SUCCESS)return 4;}
 for(int t=0;t<8;t++){tasks[t].allocate(ns[t],t);reset_task(tasks[t]);auto &q=tasks[t];prefix.push_back(q.size);prefix.push_back(q.id);prefix.push_back(q.size<100?0:1);prefix.insert(prefix.end(),q.data,q.data+q.size*q.size);prefix.insert(prefix.end(),q.vector,q.vector+q.size);prefix.insert(prefix.end(),q.result,q.result+q.size);}
 std::vector<int>workers(8);
 auto run=[&](){
 #pragma omp parallel for schedule(dynamic)
 for(int t=0;t<8;t++){int w=omp_get_thread_num();workers[t]=w;execute(tasks[t],handles,streams,w);}
 CK(cudaDeviceSynchronize());};
 for(int warm=0;warm<2;warm++){for(auto&t:tasks)reset_task(t);run();}
 for(int trial=1;trial<=5;trial++){for(auto&t:tasks)reset_task(t);CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();run();double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();auto out=prefix;bool ok=true;
 for(auto&t:tasks){for(unsigned i=0;i<t.size;i++){long long v=0;for(unsigned j=0;j<t.size;j++){unsigned k=t.size<100?i*t.size+j:j*t.size+i;v+=llround(t.data[k]*32)*llround(t.vector[j]*16);}ok&=t.result[i]==double(v)/512.;out.push_back(t.result[i]);}}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 5;fwrite(out.data(),8,out.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":1326,\"max_abs_error\":%d,\"max_rel_error\":%d,\"worker_assignments\":[%d,%d,%d,%d,%d,%d,%d,%d],\"timing_region\":\"dynamic_cpu_gpu_task_dispatch_and_device_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,time,ok?"true":"false",ok?0:1,ok?0:1,workers[0],workers[1],workers[2],workers[3],workers[4],workers[5],workers[6],workers[7],path.c_str(),out.size());if(!ok)return 6;
 }
 for(auto&t:tasks)t.deallocate();for(int i=0;i<5;i++){cublasDestroy(handles[i]);CK(cudaStreamDestroy(streams[i]));}puts("PASS");return 0;
}
