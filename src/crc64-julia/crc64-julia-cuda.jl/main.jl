using CUDA
include("../study_device.jl")
include("../study_common.jl")
function crc_kernel!(sz,cs,d,words,t,it,n)
 tid=Int64((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 crc_lane!(tid,sz,cs,d,words,t,it,n,Int64(64))
 nothing
end
launch(sz,cs,d,words,t,it,n)=CUDA.@cuda(threads=64,blocks=1,crc_kernel!(sz,cs,d,words,t,it,n))
crc_study(launch)
