using CUDA
const MODEL="cuda_jl"
function set_input!(d,nr::UInt64,nc::UInt64,ld::UInt64,base::UInt64)
 i=UInt64(threadIdx().x-1)+UInt64(blockDim().x)*UInt64(blockIdx().x-1)
 if i<nr*nc
  r=i%nr;c=i÷nr;s=UInt64(1)+(r+base)%UInt64(499)
  @inbounds d[Int(r+ld*c+1)]=c%s==0 ? 1f0 : 0f0
 end
 return
end
function launch!(a,base)
 CUDA.@cuda threads=256 blocks=cld(a.rows*a.cols,256) set_input!(a.d,UInt64(a.rows),UInt64(a.cols),UInt64(a.ld),base)
end
include(joinpath(@__DIR__,"../study_common.jl"))
study()
