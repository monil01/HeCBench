#pragma once
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
inline std::vector<int>sizes(){return{1,32,33,256,4099,1048576};}
inline std::vector<float>fixture(int n,int rank){std::vector<float>x(n);for(int i=0;i<n;i++)x[i]=(rank+1)*.5f+float(i%17-8)/32.f;return x;}
extern "C" int study_mpi_start();extern "C" void study_mpi_finish();extern "C" void study_mpi_barrier();extern "C" void study_mpi_guard();extern "C" void study_mpi_lengths(unsigned long);extern "C" void study_mpi_scatter(float*,int,const float*,int);extern "C" void study_mpi_allgather(float*,int,const float*,int);extern "C" void study_mpi_capture(const float*,int,float*);extern "C" double study_mpi_max(double);extern "C" void study_mpi_abort();
inline int prefixcount(){int n=1;for(int s:sizes())n+=1+2*s;return n;}
inline int outcount(){int n=prefixcount();for(int s:sizes())n+=2*s+2*((s+1)/2)+768;return n;}
extern "C" int study_check(const char*,double*);
