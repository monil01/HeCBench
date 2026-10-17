// Source-algorithm extraction adapter: same Thrust policies/calls, preallocated buffers.
// Native original main is compiled unchanged under an alternate entrypoint.
#define main original_nosync_cli_main
#include "../nosync-cuda/main.cu"
#undef main
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
extern "C" int study_thrust_version(){return THRUST_VERSION;}
extern "C" int study_create_stream(cudaStream_t*stream){return int(cudaStreamCreate(stream));}
extern "C" int study_destroy_stream(cudaStream_t stream){return int(cudaStreamDestroy(stream));}
extern "C" int study_nosync(int*vec,int*res,int n,cudaStream_t stream,int*out){
 try{
 auto begin=thrust::device_pointer_cast(vec);auto end=begin+n;auto dest=thrust::device_pointer_cast(res);
#if THRUST_VERSION < 101700
 auto nosync=thrust::cuda::par.on(stream);
#else
 auto nosync=thrust::cuda::par_nosync.on(stream);
#endif
 thrust::sequence(nosync,begin,end);
 auto sync=thrust::cuda::par.on(stream);auto op=thrust::plus<int>();
 std::future<int>future=std::async(std::launch::async,[=]{return thrust::reduce(begin,end,0,op);});
 thrust::inclusive_scan(sync,begin,end,dest);
 out[0]=future.get();cudaError_t e=cudaMemcpy(out+1,res+n-1,sizeof(int),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return int(e);out[2]=out[0]-out[1];return int(cudaGetLastError());
 }catch(const std::exception&e){std::cerr<<"ThrustBridge: "<<e.what()<<std::endl;return -1;}
}
