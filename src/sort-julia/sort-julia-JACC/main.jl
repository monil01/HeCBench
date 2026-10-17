include(joinpath(@__DIR__,"..","study_jacc_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const REDUCE_TEMPLATE=CUDA.zeros(UInt32,256)
const TOP_TEMPLATE=CUDA.zeros(UInt32,513)
const BOTTOM_TEMPLATE=CUDA.zeros(UInt32,544)
const REDUCE_SPEC=JACC.launch_spec(threads=256,blocks=64,shmem_size=1024,sync=false)
const TOP_SPEC=JACC.launch_spec(threads=256,blocks=1,shmem_size=2052,sync=false)
const BOTTOM_SPEC=JACC.launch_spec(threads=256,blocks=64,shmem_size=2176,sync=false)
function run!(a,b,sums,n)
 for shift in UInt32(0):UInt32(4):UInt32(28)
  input,output=iseven(shift÷UInt32(4)) ? (a,b) : (b,a)
  JACC.parallel_for(REDUCE_SPEC,16384,reduce_jacc!,input,sums,n,shift,REDUCE_TEMPLATE)
  JACC.parallel_for(TOP_SPEC,256,top_jacc!,sums,TOP_TEMPLATE)
  JACC.parallel_for(BOTTOM_SPEC,16384,bottom_jacc!,output,input,sums,n,shift,BOTTOM_TEMPLATE)
 end
 nothing
end
study(run!)
