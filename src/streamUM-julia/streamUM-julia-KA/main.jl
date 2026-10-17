using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
# Same genuine CPU helper as inherited CUDA.jl; CPU inputs are row-major.
function gemv_host!(result,data,vector,n)
 @inbounds for i=1:n
  acc=0.0
  for j=1:n;acc+=data[(i-1)*n+j]*vector[j];end
  result[i]=acc
 end
end
@kernel function gemv_ka!(result,data,vector,n::Int32,stride::Int32)
 i=Int32(@index(Global,Linear))
 while i<=n
  acc=0.0
  @inbounds for j=Int32(1):n;acc+=data[(j-Int32(1))*n+i]*vector[j];end
  @inbounds result[i]=acc
  i+=stride
 end
end
const BACKEND=CUDABackend()
function launch!(n,r,a,x)
 extent=cld(Int(n),256)*256
 gemv_ka!(BACKEND,256)(r,a,x,Int32(n),Int32(extent);ndrange=extent)
 nothing
end
streamum_study(launch!)
