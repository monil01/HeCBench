using CUDA
import JACC
JACC.@init_backend
function convert_kernel!(idx,dst,src,n)
 i=Int32(idx)
 if i<=n;@inbounds dst[i]=cast_value(eltype(dst),src[i]);end
 return
end
function launch(dst,src,n)
 threads=min(Int(n),256);blocks=cld(Int(n),threads)
 spec=JACC.launch_spec(threads=threads,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(spec,blocks*threads,convert_kernel!,dst,src,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
