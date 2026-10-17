using CUDA, KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(a,b,c,n)
 add_body(Int32(@index(Global,Linear))-Int32(1),a,b,c,n)
end
run!(a,b,c,n)=(kernel!(get_backend(a),256)(a,b,c,n;ndrange=cld(n,256)*256);nothing)
study(run!)
