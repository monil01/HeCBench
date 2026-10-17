#include <cuda_runtime.h>
#define main original_zerocopy_main
#include "../zerocopy-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cstdint>
const int N=1048579;
void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(x));exit(9);}}
struct Result{std::vector<float> output;bool ok;double kernel,alloc,binding,freed;};
Result onecase(bool generic){size_t bytes=4*N;float*host[3];void*raw[3];auto t=std::chrono::steady_clock::now();
 for(int j=0;j<3;j++){if(generic){raw[j]=malloc(bytes+4096);host[j]=(float*)ALIGN_UP(raw[j],4096);ck(cudaHostRegister(host[j],bytes,cudaHostRegisterMapped));}else ck(cudaHostAlloc((void**)&host[j],bytes,cudaHostAllocMapped));}
 double alloc=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
 for(int i=0;i<N;i++){host[0][i]=float((37*i+11)%1024-512)/4096;host[1][i]=float((53*i+7)%2048-1024)/8192;host[2][i]=-99;}
 t=std::chrono::steady_clock::now();float*dev[3];for(int j=0;j<3;j++){ck(cudaHostGetDevicePointer((void**)&dev[j],host[j],0));cudaPointerAttributes a;ck(cudaPointerGetAttributes(&a,dev[j]));if(a.type!=cudaMemoryTypeHost)exit(10);}
 double binding=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();ck(cudaDeviceSynchronize());t=std::chrono::steady_clock::now();for(int r=0;r<10;r++)vectorAddGPU<<<(N+255)/256,256>>>(dev[0],dev[1],dev[2],N);ck(cudaDeviceSynchronize());double kernel=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/10;std::vector<float>out(host[2],host[2]+N);bool ok=true;for(int i=0;i<N;i++){float want=float(2*((37*i+11)%1024-512)+((53*i+7)%2048-1024))/8192;if(out[i]!=want)ok=false;}
 t=std::chrono::steady_clock::now();for(int j=0;j<3;j++){if(generic){ck(cudaHostUnregister(host[j]));free(raw[j]);}else ck(cudaFreeHost(host[j]));}double freed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return{out,ok,kernel,alloc,binding,freed};}
int main(){onecase(false);onecase(true);for(int trial=1;trial<=5;trial++){auto a=onecase(false),b=onecase(true);bool ok=a.ok&&b.ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/trial_"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(a.output.data(),4,N,f);fwrite(b.output.data(),4,N,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":2097158,\"max_abs_error\":%d,\"timing_region\":\"sum_both_mapped_host_modes_add_kernel_sync_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"mapped_host_verified\":true,\"excluded_allocation_binding_free_seconds\":[[%.17g,%.17g,%.17g],[%.17g,%.17g,%.17g]],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":2097158,\"byte_order\":\"little\"}}\n",trial,a.kernel+b.kernel,ok?"true":"false",ok?0:1,a.alloc,a.binding,a.freed,b.alloc,b.binding,b.freed,path.c_str());if(!ok)return 11;}}
