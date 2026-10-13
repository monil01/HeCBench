#pragma once
typedef float r32;
typedef double r64;
typedef int i32;
struct alignas(16) float4 { float x,y,z,w; };
struct float3 { float x,y,z; };
struct alignas(8) int2 { int x,y; };
static_assert(sizeof(float4)==16 && sizeof(float3)==12 && sizeof(int2)==8,"HIP vector ABI");
