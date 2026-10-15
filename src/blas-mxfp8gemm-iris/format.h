#pragma once
// CUDA-source MXFP8: E4M3 byte values, UE8M0 block scales per32.
constexpr int FORMAT_BITS=8, MANTISSA_BITS=3, EXPONENT_BIAS=7;
constexpr int SCALE_BLOCK=32, SCALE_FILL=127, VALUE_FILL=0x38;
constexpr bool SCALE_E8M0=true;
constexpr const char* FORMAT_NAME="MXFP8 E4M3/UE8M0 block32";
