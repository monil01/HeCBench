// CPU-only comparison of original reference.h against independent fixture oracle.
#include <vector>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <complex>
#include "../tsa-cuda/reference.h"
std::vector<double>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<double>a(n/8);f.read((char*)a.data(),n);return a;}
template<class T>void check(std::vector<double>&input,std::vector<double>&expected,size_t&at,size_t&out,int w,int h,int steps){int n=w*h;T a=input[at++],b=input[at++];std::vector<T>r(n),i(n);for(auto&v:r)v=input[at++];for(auto&v:i)v=input[at++];reference(r.data(),i.data(),a,b,w,h,steps);double e=0;for(int k=0;k<n;k++)e=std::max(e,std::max(fabs(r[k]-expected[out+k]),fabs(i[k]-expected[out+n+k])));printf("{\"dtype\":\"%s\",\"w\":%d,\"h\":%d,\"steps\":%d,\"source_cpu_checker_max_abs_vs_independent\":%.17g}\n",sizeof(T)==4?"float32":"float64",w,h,steps,e);out+=4*n;}
int main(){auto x=load("fixture_f64.bin"),e=load("expected_f64.bin");size_t at=0,out=0;check<float>(x,e,at,out,64,96,3);check<float>(x,e,at,out,70,98,4);check<float>(x,e,at,out,34,50,1);check<double>(x,e,at,out,64,96,3);check<double>(x,e,at,out,70,98,4);check<double>(x,e,at,out,34,50,1);
// Original init_p expression, compare actual FP32 generator to independently defined Gaussian fixture.
size_t offset=2;double diff=0;int unequal=0;for(int y=1;y<=96;y++)for(int ix=1;ix<=64;ix++){std::complex<float>z=std::complex<float>(exp(-(pow(ix-180.,2.)+pow(y-300.,2.))/(2*pow(64.,2.))),0.)*exp(std::complex<float>(0.,.4*(ix+y-480.)));int k=(y-1)*64+ix-1;double d=std::max(fabs(double(z.real())-x[offset+k]),fabs(double(z.imag())-x[offset+6144+k]));diff=std::max(diff,d);if(d!=0)unequal++;}printf("{\"gaussian_fixture_formula_same\":true,\"original_FP32_host_intermediate_evaluation_identical\":false,\"unequal_complex_cells\":%d,\"max_component_difference\":%.17g,\"scope\":\"Common independently defined fixture for all four GPU models; source-formula label does not assert original FP32 generator bytes\"}\n",unequal,diff);}
