using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function packed_atomic_kernel!(output,n::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n
  bucket=mod(i-Int32(1),Int32(256))+Int32(1)
  packed_half2_add!(pointer(output,bucket))
 end
end
function factory(output,n)
 kernel=packed_atomic_kernel!(get_backend(output),256)
 return ()->kernel(output,Int32(n);ndrange=n)
end
f16atomic_study(factory)
