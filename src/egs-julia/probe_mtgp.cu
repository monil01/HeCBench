// Native-readiness diagnostic only, not a benchmark translation or timing series.
#define main untouched_full_EGS_main
#include "../egs-cuda/main.cu"
#undef main
#include <vector>
#include <cassert>
#include <string>
__global__ void probe_rng(float*values){indices idx=get_indices();MT_read_status(idx);for(int draw=0;draw<12;draw++)values[draw*32+idx.t]=get_rand(idx);MT_write_status(idx);}
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAerror %s\n",cudaGetErrorString(e));exit(2);}}
template<class T>void dumpdata(const char*name,T*device,size_t size){std::vector<T>z(size);check(cudaMemcpy(z.data(),device,size*sizeof(T),cudaMemcpyDeviceToHost));FILE*f=fopen(name,"wb");assert(f);assert(fwrite(z.data(),sizeof(T),size,f)==size);fclose(f);}
int main(int argc,char**argv){assert(argc==2);std::string path=std::string(argv[1])+"/MTGP_3217_0-8191.bin";MT_params_file=path.c_str();init_MTs(1325631759);check(cudaDeviceSynchronize());dumpdata("mtgp_initial_status.bin",d_MT_statuses,MT_NUM_STATUS);dumpdata("mtgp_actual_params.bin",reinterpret_cast<uint*>(d_MT_params),size_t(4));dumpdata("mtgp_actual_tables.bin",reinterpret_cast<uint*>(d_MT_tables),size_t(32));float*values;check(cudaMalloc(&values,12*32*sizeof(float)));probe_rng<<<1,32>>>(values);check(cudaDeviceSynchronize());dumpdata("mtgp_uniforms.bin",values,size_t(12*32));dumpdata("mtgp_final_status.bin",d_MT_statuses,MT_NUM_STATUS);check(cudaFree(values));check(cudaFree(d_MT_tables));check(cudaFree(d_MT_params));check(cudaFree(d_MT_statuses));printf("nativeRNGprobecompleted,12callsx32lanes,full128wordstatus(before/after),actual4wordparams/32wordtables\n");}
