#include "../recursiveGaussian-cuda/RecursiveGaussianHost.cpp"
extern "C" void coefficient_values(float s,float*p){GaussParms g;PreProcessGaussParms(s,0,&g);__builtin_memcpy(p,&g,sizeof(g));}
