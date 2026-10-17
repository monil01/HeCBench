#include <cstdio>
#include <cstring>
#include <cmath>
#include "../backprop-cuda/backprop.h"
#include "../backprop-cuda/reference.h"
int main(){float ly[33],w[33*17]={},old[33*17]={},delta[17];for(float&x:ly)x=1;for(float&x:delta)x=1;kernel_adjust_weights_ref(ly,w,delta,old,2,16);float expected=ETA;printf("{\"original_CPU_checker_bias\":%.17g,\"single_step_expected_bias\":%.17g,\"block_count\":2,\"source_checker_disagrees\":%s}\n",w[1],expected,fabsf(w[1]-expected)>.1f?"true":"false");return fabsf(w[1]-expected)>.1f?0:2;}
