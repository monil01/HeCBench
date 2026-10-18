#include <cuda_runtime.h>
#include "../lut-gemm-cuda/nQWeight_fp16.h"
#include "../lut-gemm-cuda/mv_fp16.h"
void instantiate(unsigned*W,__half*A,__half*B,__half*X,__half*Y){
 _nqmv<64><<<dim3(1,2),256>>>(W,A,X,Y,34,128,8,2048,64);
 _nqmv_bias<32><<<dim3(1,2),256>>>(W,A,B,X,Y,34,64,8,2048,32);
}
