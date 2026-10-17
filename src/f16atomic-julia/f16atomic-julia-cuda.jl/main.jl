using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function packed_atomic_kernel!(output,n::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n
  bucket=mod(i-Int32(1),Int32(256))+Int32(1)
  packed_half2_add!(pointer(output,bucket))
 end
 return
end
factory(output,n)=()->(@cuda threads=256 blocks=cld(n,256) packed_atomic_kernel!(output,Int32(n)))
if abspath(PROGRAM_FILE)==@__FILE__;f16atomic_study(factory);end
