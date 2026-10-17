using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function add_ka!(n::Int32,x,y,stride::Int32)
 i=Int32(@index(Global,Linear))-Int32(1)
 while i<n
  @inbounds y[i+1]+=x[i+1]
  i+=stride
 end
end
const BACKEND=CUDABackend()
function launch!(n,a,b)
 extent=cld(Int(n),256)*256
 add_ka!(BACKEND,256)(n,a,b,Int32(extent);ndrange=extent)
 nothing
end
prefetch_study(launch!)
