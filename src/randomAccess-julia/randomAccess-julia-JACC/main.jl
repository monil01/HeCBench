using CUDA,JACC,StaticArrays
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function init!(i,table)
 @inbounds table[i]=UInt64(i-1)
end
function update!(tid,table)
 ran=starts(Int64(512)*(tid-1))
 for i in 1:512
  ran=step_rng(ran);index=Int(ran&UInt64(16383))+1
  Atomix.@atomic table[index]⊻=ran
 end
end
function run!(table)
 JACC.parallel_for(JACC.launch_spec(threads=256,blocks=64,shmem_size=0,sync=false),16384,init!,table)
 JACC.parallel_for(JACC.launch_spec(threads=128,blocks=1,shmem_size=0,sync=false),128,update!,table)
 return
end
study(run!)
