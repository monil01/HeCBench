#pragma once
#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include "symbolic.h"
struct SCall{int op,threads,blocks,smem;unsigned n,nnz,currentCol;int levelHead,inLevPos,stream;float pert;int pad;void*pointers[9];};
static_assert(sizeof(SCall)==120,"SLUCABI");using SCallback=int(*)(SCall*);
extern SCallback study_cb;
extern std::array<unsigned long long,6>study_counts;
void study_before(Symbolic_Matrix&,void**,float*,bool,float);
void study_after(Symbolic_Matrix&,void**,float*);
void study_solved(Symbolic_Matrix&,const std::vector<float>&,const char*,bool);
void study_roi_seconds(double);
