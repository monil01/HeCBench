using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function cast_from_ka!(input,out,n::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds out[i]=cast_from_fp(input[i]);end
end
@kernel function cast_to_ka!(input,out,n::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds out[i]=cast_to_fp(input[i]);end
end
function factory(input,out,n,mode)
 k=mode=="from_fp" ? cast_from_ka! : cast_to_ka!
 kernel=k(get_backend(input),256)
 return ()->kernel(input,out,Int32(n);ndrange=n)
end
cast_study(factory)
