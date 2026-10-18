using CUDA
include("../study_device.jl")
include("../study_common.jl")
function contraction!(tensor,adj,value,n,N,C)
 tid=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 contract_lane!(tid,tensor,adj,value,n,N,C)
 nothing
end
factory(t,a,o)=()->CUDA.@cuda(threads=256,blocks=103,contraction!(t,a,o,Int32(26244),Int32(9),Int32(18)))
contract_study(factory)
