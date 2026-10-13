#pragma once
struct alignas(16) float4 { float x,y,z,w; };
struct alignas(16) double4 { double x,y,z,w; };
inline float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
inline double4 make_double4(double x,double y,double z,double w) { return {x,y,z,w}; }
