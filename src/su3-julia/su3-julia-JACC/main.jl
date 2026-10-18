using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function su3!(index,a,b,c,n)
 su3_body!(a,b,c,n,Int32(index)-Int32(1))
 nothing
end
function launch(a,b,c,n,threads)
 spec=JACC.launch_spec(;threads=threads,blocks=Int(n),shmem_size=0,sync=false)
 JACC.parallel_for(spec,Int(n)*threads,su3!,a,b,c,n)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(JACC.array,launch)
