using CUDA
include("../study_device.jl")
launch(h,b,th,ti,bias,lg,ho,io,ci,co)=(@cuda threads=256 blocks=cld(h*b,256) elementwise_kernel!(Int32(h),Int32(b),th,ti,bias,lg,ho,io,ci,co))
include("../study_common.jl")
run_study(launch)
