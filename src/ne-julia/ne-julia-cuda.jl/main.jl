using CUDA
include("../normal_device.jl")
function normal_kernel(points,out,W,H)
 i=Int((blockIdx().x-1)*blockDim().x+threadIdx().x);normal_site!(points,out,i,W,H);nothing
end
launch(c)=(@cuda threads=256 blocks=cld(c.N,256) normal_kernel(c.p,c.out,c.W,c.H);nothing)
include("../study_common.jl")
normal_study(launch)
