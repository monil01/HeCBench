#pragma once
// CUDA-source NVFP4: E2M1, two codes/byte, UE4M3 scales per 16.
constexpr int FORMAT_BITS=4, MANTISSA_BITS=1, EXPONENT_BIAS=1;
constexpr int SCALE_BLOCK=16, SCALE_FILL=0x38, VALUE_FILL=0x2;
constexpr bool SCALE_E8M0=false;
constexpr const char* FORMAT_NAME="NVFP4 E2M1/UE4M3 block16 (CUDA semantics)";
