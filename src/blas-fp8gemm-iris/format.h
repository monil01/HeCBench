#pragma once
// CUDA-source OCP E4M3, scalar FP32 input scales, BF16 output.
constexpr int FORMAT_BITS=8, MANTISSA_BITS=3, EXPONENT_BIAS=7;
constexpr int SCALE_BLOCK=128, VALUE_FILL=0x38;
constexpr float SCALE_FILL=1.f;
constexpr bool SCALE_E8M0=false, PER_TENSOR=true;
using scale_type=float;
constexpr const char* FORMAT_NAME="FP8 OCP E4M3/FP32 per-tensor scales";
