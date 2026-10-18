using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function gr_base(index,P,T,Y,C,TCONV,PCONV,N)
 gr_base_lane(Int32(index),N,P,T,Y,C,TCONV,PCONV);nothing
end
function ratt_kernel(index,T,RF,TCONV,N)
 ratt_kernel_lane(Int32(index),N,T,RF,TCONV);nothing
end
function rdsmh_kernel(index,T,EG,TCONV,N)
 rdsmh_kernel_lane(Int32(index),N,T,EG,TCONV);nothing
end
function ratt2_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt2_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt3_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt3_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt4_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt4_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt5_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt5_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt6_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt6_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt7_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt7_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt8_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt8_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt9_kernel(index,T,RF,RB,EG,TCONV,N)
 ratt9_kernel_lane(Int32(index),N,T,RF,RB,EG,TCONV);nothing
end
function ratt10_kernel(index,T,RKLOW,TCONV,N)
 ratt10_kernel_lane(Int32(index),N,T,RKLOW,TCONV);nothing
end
function ratx_kernel(index,T,C,RF,RB,RKLOW,TCONV,N)
 ratx_kernel_lane(Int32(index),N,T,C,RF,RB,RKLOW,TCONV);nothing
end
function ratxb_kernel(index,T,C,RF,RB,RKLOW,TCONV,N)
 ratxb_kernel_lane(Int32(index),N,T,C,RF,RB,RKLOW,TCONV);nothing
end
function ratx2_kernel(index,C,RF,RB,N)
 ratx2_kernel_lane(Int32(index),N,C,RF,RB);nothing
end
function ratx4_kernel(index,C,RF,RB,N)
 ratx4_kernel_lane(Int32(index),N,C,RF,RB);nothing
end
function qssa_kernel(index,RF,RB,A,N)
 qssa_kernel_lane(Int32(index),N,RF,RB,A);nothing
end
function qssab_kernel(index,RF,RB,A,N)
 qssab_kernel_lane(Int32(index),N,RF,RB,A);nothing
end
function qssa2_kernel(index,RF,RB,A,N)
 qssa2_kernel_lane(Int32(index),N,RF,RB,A);nothing
end
function rdwdot_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot2_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot2_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot3_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot3_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot6_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot6_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot7_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot7_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot8_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot8_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot9_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot9_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function rdwdot10_kernel(index,RKF,RKR,WDOT,rateconv,molwt,N)
 rdwdot10_kernel_lane(Int32(index),N,RKF,RKR,WDOT,rateconv,molwt);nothing
end
function factory(N,gpu_t,gpu_p,gpu_y,gpu_wdot,gpu_molwt,gpu_rf,gpu_rb,gpu_rklow,gpu_c,gpu_a,gpu_eg,tconv,pconv,rateconv)
 spec64=JACC.launch_spec(;threads=64,blocks=Int(N)÷64,shmem_size=0,sync=false)
 spec128=JACC.launch_spec(;threads=128,blocks=Int(N)÷128,shmem_size=0,sync=false)
 ()->begin
 JACC.parallel_for(spec128,Int(N),ratt_kernel,gpu_t,gpu_rf,tconv,N)
 JACC.parallel_for(spec128,Int(N),rdsmh_kernel,gpu_t,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),gr_base,gpu_p,gpu_t,gpu_y,gpu_c,tconv,pconv,N)
 JACC.parallel_for(spec128,Int(N),ratt2_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt3_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt4_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt5_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt6_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt7_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt8_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt9_kernel,gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratt10_kernel,gpu_t,gpu_rklow,tconv,N)
 JACC.parallel_for(spec64,Int(N),ratx_kernel,gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv,N)
 JACC.parallel_for(spec64,Int(N),ratxb_kernel,gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv,N)
 JACC.parallel_for(spec128,Int(N),ratx2_kernel,gpu_c,gpu_rf,gpu_rb,N)
 JACC.parallel_for(spec128,Int(N),ratx4_kernel,gpu_c,gpu_rf,gpu_rb,N)
 JACC.parallel_for(spec128,Int(N),qssa_kernel,gpu_rf,gpu_rb,gpu_a,N)
 JACC.parallel_for(spec64,Int(N),qssab_kernel,gpu_rf,gpu_rb,gpu_a,N)
 JACC.parallel_for(spec128,Int(N),qssa2_kernel,gpu_rf,gpu_rb,gpu_a,N)
 JACC.parallel_for(spec128,Int(N),rdwdot_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot2_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot3_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot6_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot7_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot8_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot9_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 JACC.parallel_for(spec128,Int(N),rdwdot10_kernel,gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt,N)
 end
end
main()=s3d_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
