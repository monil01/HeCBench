using CUDA
include("../arithmetic_device.jl")
function arithmetic_kernel(data,nIters,v1,v2,op,L)
 i=Int((blockIdx().x-1)*blockDim().x+threadIdx().x);arithmetic_site!(data,i,nIters,v1,v2,op,L);nothing
end
launch(c)=(@cuda threads=256 blocks=c.N÷256 arithmetic_kernel(c.data,c.R,c.v1,c.v2,Val(c.op),Val(c.L));nothing)
include("../study_common.jl")
arithmetic_study(launch)
