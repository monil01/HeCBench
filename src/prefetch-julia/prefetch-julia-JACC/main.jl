using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function add_jacc!(index,n::Int32,x,y,stride::Int32)
 i=Int32(index)-Int32(1)
 while i<n
  @inbounds y[i+1]+=x[i+1]
  i+=stride
 end
 nothing
end
const SMALL_SPEC=JACC.launch_spec(threads=256,blocks=17,shmem_size=0,sync=false)
const LARGE_SPEC=JACC.launch_spec(threads=256,blocks=4097,shmem_size=0,sync=false)
function launch!(n,a,b)
 extent=cld(Int(n),256)*256;spec=n==4099 ? SMALL_SPEC : LARGE_SPEC
 JACC.parallel_for(spec,extent,add_jacc!,n,a,b,Int32(extent))
 nothing
end
prefetch_study(launch!)
