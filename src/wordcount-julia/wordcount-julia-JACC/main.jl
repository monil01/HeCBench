using CUDA
import JACC
JACC.@init_backend
import Atomix
include(joinpath(@__DIR__,"study_kernel.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const COUNTER=JACC.zeros(Int32,1)
const TEMPLATE=JACC.zeros(Int32,128)
function countfn(d,n)
 n==0&&return 0
 fill!(COUNTER,0)
 JACC.parallel_for(JACC.launch_spec(threads=128,blocks=cld(n,128),shmem_size=512,sync=false),cld(n,128)*128,count_jacc!,d,Int32(n),COUNTER,TEMPLATE)
 Int(Array(COUNTER)[1])
end
study(x->JACC.array(x),countfn)
