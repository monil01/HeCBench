#include "utils.h"
#include "kernels.h"
extern "C" __global__ void reshape_float(const float* key,const float* value,uint8_t* key_cache,uint8_t* value_cache,const int64_t* slot_mapping,int key_stride,int value_stride,int num_heads,int head_size,int block_size,int x,float k_scale,float v_scale){
 reshape_and_cache_kernel<float,uint8_t,Fp8KVCacheDataType::kFp8E4M3>(key,value,key_cache,value_cache,slot_mapping,key_stride,value_stride,num_heads,head_size,block_size,x,k_scale,v_scale);
}
extern "C" __global__ void reshape_bf16(const __hip_bfloat16* key,const __hip_bfloat16* value,uint8_t* key_cache,uint8_t* value_cache,const int64_t* slot_mapping,int key_stride,int value_stride,int num_heads,int head_size,int block_size,int x,float k_scale,float v_scale){
 reshape_and_cache_kernel<__hip_bfloat16,uint8_t,Fp8KVCacheDataType::kFp8E4M3>(key,value,key_cache,value_cache,slot_mapping,key_stride,value_stride,num_heads,head_size,block_size,x,k_scale,v_scale);
}
