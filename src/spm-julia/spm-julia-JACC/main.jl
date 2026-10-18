using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_constants.jl"))
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(M,n,g,f,v,a,c,t)
 spec=JACC.launch_spec(threads=128,blocks=256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,128*256,spm_jacc!,M,Int32(n),g,f,Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),Int32(v),a,c,t)
end
study((T,n)->JACC.zeros(T,n),launch)
