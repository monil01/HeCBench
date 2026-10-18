using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function contraction!(i,tensor,adj,value,n,N,C)
 contract_lane!(Int32(i-1),tensor,adj,value,n,N,C)
 nothing
end
function factory(t,a,o)
 spec=JACC.launch_spec(;threads=256,blocks=103,shmem_size=0,sync=false)
 ()->JACC.parallel_for(spec,26244,contraction!,t,a,o,Int32(26244),Int32(9),Int32(18))
end
contract_study(factory)
