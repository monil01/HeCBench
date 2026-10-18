# Source-equivalent interval scatter; Thrust preprocessing is explicit shared vendor interoperation.
using CUDA
CUDA.allowscalar(false)
include(joinpath(@__DIR__,"..","study_common.jl"))
function remap_device(second,first,out,n,k)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=k
  idx=Int(second[i])+1
  out[Int(first[idx])+1]=Int32(i-1)
  idx+=1
  while idx<=n && (i==k || idx!=Int(second[i+1])+1)
   out[Int(first[idx])+1]=Int32(i-1);idx+=1
  end
 end
 return
end
launch(s,f,o,n,k)=(@cuda threads=256 blocks=cld(k,256) remap_device(s,f,o,n,k);nothing)
study(CuArray,launch)
