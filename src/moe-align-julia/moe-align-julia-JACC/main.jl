include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const SMALL32=CUDA.zeros(Int32,1089)
const SMALL64=CUDA.zeros(Int32,4225)
const GENERAL32=CUDA.zeros(Int32,1056)
const GENERAL128=CUDA.zeros(Int32,1152)
function run(top,sorted,experts,total,mapping,E,N,cum,maxpad,maxblocks,mapped)
 if N<Int32(1024)&&E<=Int32(64)
  stride=max(E,Int32(32));threads=Int(Int32(256)+stride);template=E==Int32(32) ? SMALL32 : SMALL64
  JACC.parallel_for(JACC.launch_spec(threads=threads,blocks=1,shmem_size=length(template)*4,sync=false),threads,small_jacc!,top,sorted,experts,total,mapping,E,N,maxpad,maxblocks,stride,mapped,template)
 else
  padE=cld(E,Int32(32))*Int32(32);template=padE==Int32(32) ? GENERAL32 : GENERAL128
  JACC.parallel_for(JACC.launch_spec(threads=1024,blocks=2,shmem_size=length(template)*4,sync=false),2048,general_jacc!,top,sorted,experts,total,mapping,E,padE,N,cum,maxpad,maxblocks,mapped,template)
  blocksY=Int(cld(N,Int32(256)));JACC.parallel_for(JACC.launch_spec(threads=(256,1),blocks=(1,blocksY),shmem_size=0,sync=false),(256,blocksY),scatter_jacc!,top,sorted,cum,mapping,E,N,Int32(blocksY),mapped)
 end
 return
end
study(run)
