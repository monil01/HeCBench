// Exact original C++ CPU model fitting is shared by all four study frontends.
#include "../rsc-cuda/model_fitting.cpp"
extern "C" void rsc_fit(float*p,flowvector*f,int n,int*r,int m,int e,float c,int*id,int threads){run_cpu_threads(p,f,n,r,m,e,c,id,threads);}
