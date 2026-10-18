include(joinpath(@__DIR__,"..","study_guard.jl"))
using CUDA
import JACC
JACC.@init_backend
@inline function remote_double_jacc!(i,source,target)
 @inbounds target[i+1]=source[i+1]*2f0
 nothing
end
allocate(n)=JACC.zeros(Float32,n)
function launch(n,source,target)
 spec=JACC.launch_spec(;threads=256,blocks=n÷256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,n,remote_double_jacc!,source,target)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(allocate,launch)
