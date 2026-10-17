include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run!(a,b,sums,n)
 for shift in UInt32(0):UInt32(4):UInt32(28)
  input,output=iseven(shift÷UInt32(4)) ? (a,b) : (b,a)
  @cuda threads=256 blocks=64 reduce_ref!(input,sums,n,shift)
  @cuda threads=256 blocks=1 top_ref!(sums)
  @cuda threads=256 blocks=64 bottom_ref!(output,input,sums,n,shift)
 end
 nothing
end
study(run!)
