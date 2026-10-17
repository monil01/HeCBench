using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(pixel,n)
 i=@index(Global,Linear)
 if i<=n
 @inbounds pixel[i]=transform(pixel[i])
 end
end
run!(p,n)=(kernel!(get_backend(p),256)(p,Int32(n);ndrange=n);nothing)
study(run!)
