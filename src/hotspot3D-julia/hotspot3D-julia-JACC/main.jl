using CUDA,JACC,Printf
include("../thermal_device.jl")
@inline function thermal_jacc!(i,j,a,p,b,n,z,q)
 column!(a,p,b,Int32(i-1),Int32(j-1),n,z,q)
 return nothing
end
function factory(n)
 spec=JACC.launch_spec(;threads=(64,4),shmem_size=0,sync=false)
 return (a,p,b,z,q)->JACC.parallel_for(spec,(n,n),thermal_jacc!,a,p,b,Int32(n),Int32(z),q)
end
include("../study_common.jl")
thermal_study(factory)
