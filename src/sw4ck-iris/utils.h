#ifndef __UTILS_H
#define __UTILS_H

#ifndef SW4CK_DEVICE
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
inline void iris_check(int rc) { if(rc != IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc); exit(1); } }
#endif

// Use double precision for accuracy
#define float_sw4 double

#ifndef SW4CK_DEVICE
void curvilinear4sg_ci(
    int ifirst, int ilast,
    int jfirst, int jlast,
    int kfirst, int klast,
    iris::DMem &d_u,
    iris::DMem &d_mu,
    iris::DMem &d_lambda,
    iris::DMem &d_met,
    iris::DMem &d_jac,
    iris::DMem &d_lu,
    int* onesided,
    iris::DMem &d_acof,
    iris::DMem &d_bope,
    iris::DMem &d_ghcof,
    iris::DMem &d_acof_no_gp,
    iris::DMem &d_ghcof_no_gp,
    iris::DMem &d_strx,
    iris::DMem &d_stry,
    int nk, char op);
#endif

#endif
