#include "reference_history/diagnostic_reference_print_jacobian.h"
#include "../rushlarsen-cuda/utils.cu"
int main(){double s[19],p[53];init_state_values(s,1);init_parameters_values(p,1);forward_rush_larsen(s,0.,2e-5,p,1);}
