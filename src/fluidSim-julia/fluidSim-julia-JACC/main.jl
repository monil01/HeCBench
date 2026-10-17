using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function lbm_jacc!(i,j,a,oa,b,ob,c,oc,types,w,omega)
 lbm_body!(Int32(i)-Int32(1),Int32(j)-Int32(1),a,oa,b,ob,c,oc,types,w,omega)
 nothing
end
const SPEC=JACC.launch_spec(threads=(1,HEIGHT),blocks=(256,1),shmem_size=0,sync=false)
function run!(a,oa,b,ob,c,oc,types,w,omega)
 JACC.parallel_for(SPEC,(WIDTH,HEIGHT),lbm_jacc!,a,oa,b,ob,c,oc,types,w,omega)
 nothing
end
study(run!)
