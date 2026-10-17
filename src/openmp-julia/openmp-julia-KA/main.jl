using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(a,b,repeat)
 i=@index(Global,Linear)
 for j in Int32(0):repeat-Int32(1)
  @inbounds a[i]+=mod(j,b)
 end
end
run!(a,b,r)=kernel!(get_backend(a),256)(a,b,r;ndrange=length(a))
study(run!)
