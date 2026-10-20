#include "/home/imo/HeCBench_julia/src/sparkler-julia/study_capture.h"
#include <iostream>
int main(){for(auto c:study_cases())std::cout<<c.vectors<<" "<<c.fields<<" "<<c.iterations<<" "<<study_global_hash(c)<<"\n";}
