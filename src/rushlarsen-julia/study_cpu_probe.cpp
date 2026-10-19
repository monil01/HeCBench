#include <cstdio>
#include <cmath>
#include <vector>
#include "../rushlarsen-cuda/utils.h"
int main(){std::vector<double>s(19),p(53);init_state_values(s.data(),1);init_parameters_values(p.data(),1);forward_rush_larsen(s.data(),0.,2e-5,p.data(),1);for(int f=0;f<19;f++)printf("STATE %d CPU=%.17g CPU_FINITE=%d\n",f,s[f],std::isfinite(s[f]));}
