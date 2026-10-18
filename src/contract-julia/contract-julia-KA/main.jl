using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function contraction!(tensor,adj,value,n,N,C)
 tid=Int32(@index(Global,Linear)-1)
 contract_lane!(tid,tensor,adj,value,n,N,C)
end
function factory(t,a,o)
 k=contraction!(CUDABackend(),256)
 ()->k(t,a,o,Int32(26244),Int32(9),Int32(18);ndrange=26244)
end
contract_study(factory)
