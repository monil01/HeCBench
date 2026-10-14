#include <hip/hip_runtime.h>
#include "device/gr_base.h"
#include "device/ratt.h"
#include "device/ratt2.h"
#include "device/ratx.h"
#include "device/qssa.h"
#include "device/qssa2.h"
#include "device/rdwdot.h"
extern "C" __global__ void gr_base_float(const float* P, const float* T, const float* Y, float* C, float TCONV, float PCONV){gr_base<float>(P,T,Y,C,TCONV,PCONV);}
extern "C" __global__ void gr_base_double(const double* P, const double* T, const double* Y, double* C, double TCONV, double PCONV){gr_base<double>(P,T,Y,C,TCONV,PCONV);}
extern "C" __global__ void ratt_kernel_float(const float* T, float* RF, float TCONV){ratt_kernel<float>(T,RF,TCONV);}
extern "C" __global__ void ratt_kernel_double(const double* T, double* RF, double TCONV){ratt_kernel<double>(T,RF,TCONV);}
extern "C" __global__ void rdsmh_kernel_float(const float* T, float* EG, float TCONV){rdsmh_kernel<float>(T,EG,TCONV);}
extern "C" __global__ void rdsmh_kernel_double(const double* T, double* EG, double TCONV){rdsmh_kernel<double>(T,EG,TCONV);}
extern "C" __global__ void ratt2_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt2_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt2_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt2_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt3_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt3_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt3_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt3_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt4_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt4_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt4_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt4_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt5_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt5_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt5_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt5_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt6_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt6_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt6_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt6_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt7_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt7_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt7_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt7_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt8_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt8_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt8_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt8_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt9_kernel_float(const float* T, const float* RF, float* RB, const float* EG, float TCONV){ratt9_kernel<float>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt9_kernel_double(const double* T, const double* RF, double* RB, const double* EG, double TCONV){ratt9_kernel<double>(T,RF,RB,EG,TCONV);}
extern "C" __global__ void ratt10_kernel_float(const float* T, float* RKLOW, float TCONV){ratt10_kernel<float>(T,RKLOW,TCONV);}
extern "C" __global__ void ratt10_kernel_double(const double* T, double* RKLOW, double TCONV){ratt10_kernel<double>(T,RKLOW,TCONV);}
extern "C" __global__ void ratx_kernel_float(const float* T, const float* C, float* RF, float* RB, const float* RKLOW, float TCONV){ratx_kernel<float>(T,C,RF,RB,RKLOW,TCONV);}
extern "C" __global__ void ratx_kernel_double(const double* T, const double* C, double* RF, double* RB, const double* RKLOW, double TCONV){ratx_kernel<double>(T,C,RF,RB,RKLOW,TCONV);}
extern "C" __global__ void ratxb_kernel_float(const float* T, const float* C, float* RF, float* RB, const float* RKLOW, float TCONV){ratxb_kernel<float>(T,C,RF,RB,RKLOW,TCONV);}
extern "C" __global__ void ratxb_kernel_double(const double* T, const double* C, double* RF, double* RB, const double* RKLOW, double TCONV){ratxb_kernel<double>(T,C,RF,RB,RKLOW,TCONV);}
extern "C" __global__ void ratx2_kernel_float(const float* C, float* RF, float* RB){ratx2_kernel<float>(C,RF,RB);}
extern "C" __global__ void ratx2_kernel_double(const double* C, double* RF, double* RB){ratx2_kernel<double>(C,RF,RB);}
extern "C" __global__ void ratx4_kernel_float(const float* C, float* RF, float* RB){ratx4_kernel<float>(C,RF,RB);}
extern "C" __global__ void ratx4_kernel_double(const double* C, double* RF, double* RB){ratx4_kernel<double>(C,RF,RB);}
extern "C" __global__ void qssa_kernel_float(float* RF, float* RB, float* A){qssa_kernel<float>(RF,RB,A);}
extern "C" __global__ void qssa_kernel_double(double* RF, double* RB, double* A){qssa_kernel<double>(RF,RB,A);}
extern "C" __global__ void qssab_kernel_float(float* RF, float* RB, float* A){qssab_kernel<float>(RF,RB,A);}
extern "C" __global__ void qssab_kernel_double(double* RF, double* RB, double* A){qssab_kernel<double>(RF,RB,A);}
extern "C" __global__ void qssa2_kernel_float(float* RF, float* RB, const float* A){qssa2_kernel<float>(RF,RB,A);}
extern "C" __global__ void qssa2_kernel_double(double* RF, double* RB, const double* A){qssa2_kernel<double>(RF,RB,A);}
extern "C" __global__ void rdwdot_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot2_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot2_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot2_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot2_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot3_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot3_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot3_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot3_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot6_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot6_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot6_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot6_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot7_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot7_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot7_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot7_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot8_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot8_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot8_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot8_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot9_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot9_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot9_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot9_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot10_kernel_float(const float* RKF, const float* RKR, float* WDOT, float rateconv, const float* molwt){rdwdot10_kernel<float>(RKF,RKR,WDOT,rateconv,molwt);}
extern "C" __global__ void rdwdot10_kernel_double(const double* RKF, const double* RKR, double* WDOT, double rateconv, const double* molwt){rdwdot10_kernel<double>(RKF,RKR,WDOT,rateconv,molwt);}
