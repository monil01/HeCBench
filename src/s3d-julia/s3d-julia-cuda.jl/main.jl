using CUDA
include("../study_common.jl")
function gr_base(P,T,Y,C,TCONV,PCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 gr_base_lane(i,N,P,T,Y,C,TCONV,PCONV);nothing
end
function ratt_kernel(T,RF,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt_kernel_lane(i,N,T,RF,TCONV);nothing
end
function rdsmh_kernel(T,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdsmh_kernel_lane(i,N,T,EG,TCONV);nothing
end
function ratt2_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt2_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt3_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt3_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt4_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt4_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt5_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt5_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt6_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt6_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt7_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt7_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt8_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt8_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt9_kernel(T,RF,RB,EG,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt9_kernel_lane(i,N,T,RF,RB,EG,TCONV);nothing
end
function ratt10_kernel(T,RKLOW,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratt10_kernel_lane(i,N,T,RKLOW,TCONV);nothing
end
function ratx_kernel(T,C,RF,RB,RKLOW,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratx_kernel_lane(i,N,T,C,RF,RB,RKLOW,TCONV);nothing
end
function ratxb_kernel(T,C,RF,RB,RKLOW,TCONV,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratxb_kernel_lane(i,N,T,C,RF,RB,RKLOW,TCONV);nothing
end
function ratx2_kernel(C,RF,RB,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratx2_kernel_lane(i,N,C,RF,RB);nothing
end
function ratx4_kernel(C,RF,RB,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 ratx4_kernel_lane(i,N,C,RF,RB);nothing
end
function qssa_kernel(RF,RB,A,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 qssa_kernel_lane(i,N,RF,RB,A);nothing
end
function qssab_kernel(RF,RB,A,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 qssab_kernel_lane(i,N,RF,RB,A);nothing
end
function qssa2_kernel(RF,RB,A,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 qssa2_kernel_lane(i,N,RF,RB,A);nothing
end
function rdwdot_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot2_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot2_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot3_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot3_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot6_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot6_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot7_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot7_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot8_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot8_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot9_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot9_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot10_kernel(RKF,RKR,WDOT,rateconv,molwt,N)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 rdwdot10_kernel_lane(i,N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function factory(N,gpu_t,gpu_p,gpu_y,gpu_wdot,gpu_molwt,gpu_rf,gpu_rb,gpu_rklow,gpu_c,gpu_a,gpu_eg,tconv,pconv,rateconv)
 ()->begin
 @cuda threads=128 blocks=Int(N)÷128 ratt_kernel(gpu_t,gpu_rf,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 rdsmh_kernel(gpu_t,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 gr_base(gpu_p,gpu_t,gpu_y,gpu_c,tconv,pconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt2_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt3_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt4_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt5_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt6_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt7_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt8_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt9_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratt10_kernel(gpu_t,gpu_rklow,tconv,N)
 @cuda threads=64 blocks=Int(N)÷64 ratx_kernel(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv,N)
 @cuda threads=64 blocks=Int(N)÷64 ratxb_kernel(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv,N)
 @cuda threads=128 blocks=Int(N)÷128 ratx2_kernel(gpu_c,gpu_rf,gpu_rb,N)
 @cuda threads=128 blocks=Int(N)÷128 ratx4_kernel(gpu_c,gpu_rf,gpu_rb,N)
 @cuda threads=128 blocks=Int(N)÷128 qssa_kernel(gpu_rf,gpu_rb,gpu_a,N)
 @cuda threads=64 blocks=Int(N)÷64 qssab_kernel(gpu_rf,gpu_rb,gpu_a,N)
 @cuda threads=128 blocks=Int(N)÷128 qssa2_kernel(gpu_rf,gpu_rb,gpu_a,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot2_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot3_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot6_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot7_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot8_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot9_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 @cuda threads=128 blocks=Int(N)÷128 rdwdot10_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 end
end
main()=s3d_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
