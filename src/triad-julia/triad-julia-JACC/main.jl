using CUDA
import JACC
JACC.@init_backend
function triad_jacc!(i,a,b,c,s,n)
    @inbounds c[i]=a[i]+s*b[i]
    return
end
const TRIAD_SPEC=JACC.launch_spec(threads=128,shmem_size=0,sync=false)
function launch_jacc!(a,b,c,s,n)
    JACC.parallel_for(TRIAD_SPEC,n,triad_jacc!,a,b,c,s,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
triad_study(launch_jacc!)
