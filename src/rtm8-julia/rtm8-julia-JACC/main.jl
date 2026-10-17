using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(x,y,z,v,cs,formalcr,formalns,nr,image,a)
 body(Int32(x)-Int32(1),Int32(y)-Int32(1),Int32(z)-Int32(1),v,cs,formalcr,formalns,nr,image,a)
 return
end
run!(v,cs,ns,cr,nr,image,a)=(JACC.parallel_for(JACC.launch_spec(threads=(16,16,1),blocks=(cld(NX,16),cld(NY,16),NZ),shmem_size=0,sync=false),(cld(NX,16)*16,cld(NY,16)*16,NZ),kernel!,v,cs,ns,cr,nr,image,a);nothing)
study(run!)
