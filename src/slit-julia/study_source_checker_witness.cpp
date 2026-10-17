#include <cstdio>
#include <vector>
#include <algorithm>
#include "../slit-cuda/reference.h"
int main(){int N=32;std::vector<double>a(N*N),out(N*N);a[2*N+1]=1;reference(a.data(),out.data(),N);double lo=*std::min_element(out.begin(),out.end()),hi=*std::max_element(out.begin(),out.end());printf("{\"source_CPU_reference_min_impulse_intensity\":%.17g,\"max\":%.17g,\"mathematical_all_pixels_expected\":1.0,\"disagrees\":%s}\n",lo,hi,lo<.5?"true":"false");return lo<.5?0:2;}
