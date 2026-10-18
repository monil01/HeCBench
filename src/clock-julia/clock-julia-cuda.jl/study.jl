using CUDA
function clock_reduction!(input,out,timer)
 tid=Int32(threadIdx().x)-Int32(1);bid=Int32(blockIdx().x);blocks=Int32(gridDim().x);sh=CuDynamicSharedArray(Float32,512)
 if tid==0;timer[bid]=Int64(CUDA.clock(UInt64));end
 sh[tid+Int32(1)]=input[tid+Int32(1)];sh[tid+Int32(257)]=input[tid+Int32(257)]
 d=Int32(256)
 while d>0
  sync_threads()
  if tid<d
   f0=sh[tid+Int32(1)];f1=sh[tid+d+Int32(1)]
   if f1<f0;sh[tid+Int32(1)]=f1;end
  end
  d÷=Int32(2)
 end
 if tid==0;out[bid]=sh[1];end
 sync_threads()
 if tid==0;timer[bid+blocks]=Int64(CUDA.clock(UInt64));end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
clock_study((a,o,t,b)->(@cuda threads=256 blocks=b shmem=2048 clock_reduction!(a,o,t)),"cuda.jl")
