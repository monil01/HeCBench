using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(gate,voltage,n)
 i=@index(Global,Linear)
 if i<=n
 @inbounds gate[i]=step_gate(gate[i],voltage[i])
 end
end
run!(g,v,n)=(kernel!(get_backend(g),256)(g,v,Int32(n);ndrange=n);nothing)
study(run!)
