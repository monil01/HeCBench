using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_types.jl"))
include(joinpath(@__DIR__,"..","study_body.jl"))
const SHARED_TEMPLATE=JACC.zeros(Float32,900)
function md_kernel!(xx,meta,rv,q,fv,alpha::Float32,nb::Int32,template)
 i=Int32(xx)-Int32(1);BX=i÷Int32(128);TX=i%Int32(128);sh=JACC.shared(template)
 md_body!(meta,rv,q,fv,alpha,nb,BX,TX,sh,JACC.sync_workgroup)
 return nothing
end
function launch(meta,rv,q,fv,alpha,nb)
 spec=JACC.launch_spec(;threads=128,blocks=nb,shmem_size=3600,sync=false)
 JACC.parallel_for(spec,128*nb,md_kernel!,meta,rv,q,fv,alpha,Int32(nb),SHARED_TEMPLATE)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
