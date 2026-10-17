using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,pixel,n)
 if i<=n
 @inbounds pixel[i]=transform(pixel[i])
 end
 return
end
run!(p,n)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,kernel!,p,Int32(n));nothing)
study(run!)
