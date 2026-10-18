using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function crc_kernel!(sz,cs,d,words,t,it,n)
 tid=Int64(@index(Global,Linear)-1)
 crc_lane!(tid,sz,cs,d,words,t,it,n,Int64(64))
end
function launch(sz,cs,d,words,t,it,n)
 k=crc_kernel!(CUDABackend(),64)
 k(sz,cs,d,words,t,it,n;ndrange=64)
end
crc_study(launch)
