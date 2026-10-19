#include <vector>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cassert>
#include "bn_host_pipeline.h"
int main(){std::vector<double>fixture{3},oracle;for(int ci=0;ci<3;ci++){bn_reset(ci);for(double v:{double(NODE_N),double(DATA_N),double(STATE_N),double(sizepernode)})fixture.push_back(v);for(int x:data)fixture.push_back(x);for(int i=0;i<DATA_N+2;i++){assert(std::isfinite(LG[i]));fixture.push_back(LG[i]);}genScore_ref(sizepernode,localscore_ref,data,LG);for(int i=0;i<NODE_N*sizepernode;i++){assert(std::isfinite(localscore_ref[i]));oracle.push_back(localscore_ref[i]);}bn_release();}std::ofstream f("fixture.bin",std::ios::binary);f.write((char*)fixture.data(),fixture.size()*8);std::ofstream o("score_oracle.bin",std::ios::binary);o.write((char*)oracle.data(),oracle.size()*8);printf("BN_FIXTURE %zu ORACLE %zu\n",fixture.size(),oracle.size());}
