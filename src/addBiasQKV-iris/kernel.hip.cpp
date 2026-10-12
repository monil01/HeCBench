#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
using fp8_e4m3_t = __hip_fp8_e4m3;
using fp8x4_e4m3_t = __hip_fp8x4_e4m3;
#else
using fp8_e4m3_t = __hip_fp8_e4m3_fnuz;
using fp8x4_e4m3_t = __hip_fp8x4_e4m3_fnuz;
#endif

inline __device__ __hip_bfloat162 to_bfloat2(float value) {
  const __hip_bfloat16 v = __float2bfloat16(value);
  return __hip_bfloat162{v, v};
}
inline __device__ void decode_fp8x4(__hip_bfloat162* first,
                                    __hip_bfloat162* second,
                                    const fp8x4_e4m3_t* input) {
  const char4 bytes = reinterpret_cast<const char4*>(input)[0];
  *first = __hip_bfloat162(
      static_cast<float>(reinterpret_cast<const fp8_e4m3_t*>(&bytes.x)[0]),
      static_cast<float>(reinterpret_cast<const fp8_e4m3_t*>(&bytes.y)[0]));
  *second = __hip_bfloat162(
      static_cast<float>(reinterpret_cast<const fp8_e4m3_t*>(&bytes.z)[0]),
      static_cast<float>(reinterpret_cast<const fp8_e4m3_t*>(&bytes.w)[0]));
}

extern "C" __global__ void FP8TrtAddQKVBiasKernel(
    uint8_t* qkv_tgt, const uint8_t* qkv_src, const uint16_t* qkv_bias,
    const float* input_scale, const float* output_scale, int valid_word_num,
    int head_num, int size_per_head, int hidden_unit) {
  const int word = static_cast<int>(blockIdx.x);
  const int packed_head = static_cast<int>(blockIdx.y);
  const int component = static_cast<int>(blockIdx.z);
  if (word >= valid_word_num || packed_head >= size_per_head / 4 ||
      component >= head_num * 3)
    return;

  const int size_div_4 = size_per_head / 4;
  const int head = component / 3;
  const int qkv = component % 3;
  const int src_id = qkv * (hidden_unit / 4) + head * size_div_4 + packed_head;
  const int bias_id = src_id * 4;
  const fp8x4_e4m3_t* src = reinterpret_cast<const fp8x4_e4m3_t*>(qkv_src);
  fp8x4_e4m3_t* dst = reinterpret_cast<fp8x4_e4m3_t*>(qkv_tgt);
  const fp8x4_e4m3_t* src_word = src + word * 3 * hidden_unit / 4;
  fp8x4_e4m3_t* dst_word = dst + word * 3 * hidden_unit / 4;

  __hip_bfloat162 val1, val2;
  decode_fp8x4(&val1, &val2, &src_word[src_id]);
  const __hip_bfloat162 in_scale = to_bfloat2(__ldg(input_scale));
  const __hip_bfloat162 out_scale = to_bfloat2(__ldg(output_scale));
  const __hip_bfloat162* bias = reinterpret_cast<const __hip_bfloat162*>(qkv_bias);
  val1 = __hmul2(__hadd2(__hmul2(val1, in_scale), bias[bias_id / 2]), out_scale);
  val2 = __hmul2(__hadd2(__hmul2(val2, in_scale), bias[bias_id / 2 + 1]), out_scale);

  const int dst_id = head * 3 * size_div_4 + qkv * size_div_4 + packed_head;
  dst_word[dst_id] = fp8x4_e4m3_t(val2, val1);
}
