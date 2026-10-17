using CUDA,StaticArrays
include(joinpath(@__DIR__,"..","study_common.jl"))
function init!(table)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 @inbounds table[i]=UInt64(i-1)
 return
end
function update!(table)
 tid=threadIdx().x
 ran=starts(Int64(512)*(tid-1))
 for i in 1:512
  ran=step_rng(ran);index=Int(ran&UInt64(16383))+1
  CUDA.atomic_xor!(pointer(table,index),ran)
 end
 return
end
function run!(table)
 @cuda threads=256 blocks=64 init!(table)
 @cuda threads=128 blocks=1 update!(table)
 return
end
study(run!)
