#include <hip/hip_runtime.h>
#include "kernels.h"
extern "C" __global__ void Add1_float(float* data,int iterations,float a) {Add1<float>(data,iterations,a);}
extern "C" __global__ void Add2_float(float* data,int iterations,float a) {Add2<float>(data,iterations,a);}
extern "C" __global__ void Add4_float(float* data,int iterations,float a) {Add4<float>(data,iterations,a);}
extern "C" __global__ void Add8_float(float* data,int iterations,float a) {Add8<float>(data,iterations,a);}
extern "C" __global__ void Mul1_float(float* data,int iterations,float a) {Mul1<float>(data,iterations,a);}
extern "C" __global__ void Mul2_float(float* data,int iterations,float a) {Mul2<float>(data,iterations,a);}
extern "C" __global__ void Mul4_float(float* data,int iterations,float a) {Mul4<float>(data,iterations,a);}
extern "C" __global__ void Mul8_float(float* data,int iterations,float a) {Mul8<float>(data,iterations,a);}
extern "C" __global__ void MAdd1_float(float* data,int iterations,float a, float b) {MAdd1<float>(data,iterations,a,b);}
extern "C" __global__ void MAdd2_float(float* data,int iterations,float a, float b) {MAdd2<float>(data,iterations,a,b);}
extern "C" __global__ void MAdd4_float(float* data,int iterations,float a, float b) {MAdd4<float>(data,iterations,a,b);}
extern "C" __global__ void MAdd8_float(float* data,int iterations,float a, float b) {MAdd8<float>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd1_float(float* data,int iterations,float a, float b) {MulMAdd1<float>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd2_float(float* data,int iterations,float a, float b) {MulMAdd2<float>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd4_float(float* data,int iterations,float a, float b) {MulMAdd4<float>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd8_float(float* data,int iterations,float a, float b) {MulMAdd8<float>(data,iterations,a,b);}
extern "C" __global__ void Add1_double(double* data,int iterations,double a) {Add1<double>(data,iterations,a);}
extern "C" __global__ void Add2_double(double* data,int iterations,double a) {Add2<double>(data,iterations,a);}
extern "C" __global__ void Add4_double(double* data,int iterations,double a) {Add4<double>(data,iterations,a);}
extern "C" __global__ void Add8_double(double* data,int iterations,double a) {Add8<double>(data,iterations,a);}
extern "C" __global__ void Mul1_double(double* data,int iterations,double a) {Mul1<double>(data,iterations,a);}
extern "C" __global__ void Mul2_double(double* data,int iterations,double a) {Mul2<double>(data,iterations,a);}
extern "C" __global__ void Mul4_double(double* data,int iterations,double a) {Mul4<double>(data,iterations,a);}
extern "C" __global__ void Mul8_double(double* data,int iterations,double a) {Mul8<double>(data,iterations,a);}
extern "C" __global__ void MAdd1_double(double* data,int iterations,double a, double b) {MAdd1<double>(data,iterations,a,b);}
extern "C" __global__ void MAdd2_double(double* data,int iterations,double a, double b) {MAdd2<double>(data,iterations,a,b);}
extern "C" __global__ void MAdd4_double(double* data,int iterations,double a, double b) {MAdd4<double>(data,iterations,a,b);}
extern "C" __global__ void MAdd8_double(double* data,int iterations,double a, double b) {MAdd8<double>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd1_double(double* data,int iterations,double a, double b) {MulMAdd1<double>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd2_double(double* data,int iterations,double a, double b) {MulMAdd2<double>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd4_double(double* data,int iterations,double a, double b) {MulMAdd4<double>(data,iterations,a,b);}
extern "C" __global__ void MulMAdd8_double(double* data,int iterations,double a, double b) {MulMAdd8<double>(data,iterations,a,b);}
