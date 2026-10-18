using CUDA,KernelAbstractions
@kernel unsafe_indices=true function clock_reduction!(input,out,timer,blocks)
 tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear));sh=@localmem Float32 (512,)
 if tid==0;timer[bid]=Int64(CUDA.clock(UInt64));end
 sh[tid+Int32(1)]=input[tid+Int32(1)];sh[tid+Int32(257)]=input[tid+Int32(257)]
 d=Int32(256)
 while d>0
  @synchronize
  if tid<d
   f0=sh[tid+Int32(1)];f1=sh[tid+d+Int32(1)]
   if f1<f0;sh[tid+Int32(1)]=f1;end
  end
  d÷=Int32(2)
 end
 if tid==0;out[bid]=sh[1];end
 @synchronize
 if tid==0;timer[bid+blocks]=Int64(CUDA.clock(UInt64));end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const CLOCKK=clock_reduction!(CUDABackend(),256)
clock_study((a,o,t,b)->CLOCKK(a,o,t,Int32(b);ndrange=b*256),"ka")
