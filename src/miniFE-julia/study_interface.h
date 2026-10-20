#pragma once
#include <cstdint>
struct MiniLaunch{
 int32_t op=0,n=0,grid=0,threads=0;
 double alpha=0,beta=0;
 const double*x=nullptr,*y=nullptr;double*w=nullptr;
 const int32_t*row_offsets=nullptr,*cols=nullptr;const double*coefs=nullptr;
};
using MiniCallback=int(*)(const MiniLaunch*);
extern MiniCallback mini_callback;
void mini_dispatch(const MiniLaunch&);
void mini_observe_initial(int,int,const int*,const int*,const double*,const double*,const double*,const double*,const double*,const double*);
void mini_observe_solution(int,const double*,const double*,const double*,const double*,const double*,const double*,int,double);
void mini_observe_reduction(const double*);
void mini_observe_source_heuristic(int);
