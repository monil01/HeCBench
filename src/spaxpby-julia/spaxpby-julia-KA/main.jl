using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function study_scale!(y,b)
 i=@index(Global,Linear)
 if i<=length(y);@inbounds y[i]=b*y[i];end
end
@kernel function study_scatter!(y,idx,x,a)
 j=@index(Global,Linear)
 if j<=length(x);@inbounds y[idx[j]+1]+=a*x[j];end
end
function study_launch!(y,idx,x,a,b)
 backend=KernelAbstractions.get_backend(y)
 study_scale!(backend,256)(y,b;ndrange=length(y))
 study_scatter!(backend,256)(y,idx,x,a;ndrange=length(x))
end
study_main(study_launch!)
