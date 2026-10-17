using CUDA,KernelAbstractions,StaticArrays
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function init!(table)
 i=@index(Global,Linear)
 @inbounds table[i]=UInt64(i-1)
end
@kernel function update!(table)
 tid=@index(Local,Linear)
 ran=starts(Int64(512)*(tid-1))
 for i in 1:512
  ran=step_rng(ran);index=Int(ran&UInt64(16383))+1
  Atomix.@atomic table[index]⊻=ran
 end
end
function run!(table)
 init!(get_backend(table),256)(table;ndrange=16384)
 update!(get_backend(table),128)(table;ndrange=128)
 return
end
study(run!)
