using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function log_ka!(input,out,n::Int32,row::Int32,p::Int32)
 i=@index(Global,Linear)
 if i<=n
  @inbounds out[1+row*n+i]=binary_log_study(input[i],p)
 end
end
function factory(input,out,n)
 kernel=log_ka!(CUDABackend(),256)
 (row,p)->kernel(input,out,n,row,p;ndrange=n)
end
log_study(factory)
