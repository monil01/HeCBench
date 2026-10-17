include(joinpath(@__DIR__,"..","study_ka_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDABackend()
function run!(a,b,sums,n)
 for shift in UInt32(0):UInt32(4):UInt32(28)
  input,output=iseven(shift÷UInt32(4)) ? (a,b) : (b,a)
  reduce_ka!(BACKEND,256)(input,sums,n,shift;ndrange=16384)
  top_ka!(BACKEND,256)(sums;ndrange=256)
  bottom_ka!(BACKEND,256)(output,input,sums,n,shift;ndrange=16384)
 end
 nothing
end
study(run!)
