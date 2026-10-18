# KA authors only the source-equivalent remap kernel; shared Thrust preprocessing remains vendor interop.
using CUDA, KernelAbstractions
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function remap_device(second,first,out,n,k)
 i=@index(Global,Linear)
 if i<=k
  idx=Int(second[i])+1
  out[Int(first[idx])+1]=Int32(i-1)
  idx+=1
  while idx<=n && (i==k || idx!=Int(second[i+1])+1)
   out[Int(first[idx])+1]=Int32(i-1);idx+=1
  end
 end
end
launch(s,f,o,n,k)=(remap_device(KernelAbstractions.get_backend(o),256)(s,f,o,n,k;ndrange=256cld(k,256));nothing)
study(a->KernelAbstractions.allocate(CUDABackend(),eltype(a),size(a)) |> x->(copyto!(x,a);x),launch)
