#pragma once
// CUDA-source MXFP6: E2M3, four 6-bit values per three bytes, UE8M0/32.
constexpr int FORMAT_BITS=6, MANTISSA_BITS=3, EXPONENT_BIAS=1;
constexpr int SCALE_BLOCK=32, SCALE_FILL=127, VALUE_FILL=0x08;
constexpr bool SCALE_E8M0=true;
// Retain CUDA/HIP helper decodeE2M3 exp==0 factor 0.5 exactly. Source uniform
// code0x08 is normal and unaffected; no all-code hardware-format parity claim.
constexpr int SUBNORMAL_SHIFT=-1;
constexpr const char* FORMAT_NAME="MXFP6 E2M3/UE8M0 block32";
