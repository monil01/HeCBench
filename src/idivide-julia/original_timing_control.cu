#include <cuda_runtime.h>
#include "../idivide-cuda/fastdiv.h"
#include "../idivide-cuda/kernels.h"
void launch_original(){
 throughput_test<int><<<32768,256>>>(3,5,7,0,nullptr);
 throughput_test<int_fastdiv><<<32768,256>>>(3,5,7,0,nullptr);
 latency_test<int><<<32768,256>>>(1,2,3,4,5,6,7,8,9,10,0,nullptr);
 latency_test<int_fastdiv><<<32768,256>>>(1,2,3,4,5,6,7,8,9,10,0,nullptr);
}
