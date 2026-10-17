using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function study_scale!(i,y,b)
 @inbounds y[i]=b*y[i]
end
@inline function study_scatter!(j,y,idx,x,a)
 @inbounds y[idx[j]+1]+=a*x[j]
end
function study_launch!(y,idx,x,a,b)
 backend=JACC.get_backend(:cuda)
 spec1=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(length(y),256),shmem_size=0,sync=false)
 spec2=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(length(x),256),shmem_size=0,sync=false)
 JACC.parallel_for(study_scale!,spec1,length(y),y,b)
 JACC.parallel_for(study_scatter!,spec2,length(x),y,idx,x,a)
end
study_main(study_launch!)
