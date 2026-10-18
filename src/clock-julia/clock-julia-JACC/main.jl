using CUDA,JACC
function clock_reduction!(i,input,out,timer,blocks,template)
 z=Int32(i)-Int32(1);tid=z%Int32(256);bid=z÷Int32(256)+Int32(1);sh=JACC.shared(template)
 if tid==0;timer[bid]=Int64(CUDA.clock(UInt64));end
 sh[tid+Int32(1)]=input[tid+Int32(1)];sh[tid+Int32(257)]=input[tid+Int32(257)]
 d=Int32(256)
 while d>0
  JACC.sync_workgroup()
  if tid<d
   f0=sh[tid+Int32(1)];f1=sh[tid+d+Int32(1)]
   if f1<f0;sh[tid+Int32(1)]=f1;end
  end
  d÷=Int32(2)
 end
 if tid==0;out[bid]=sh[1];end
 JACC.sync_workgroup()
 if tid==0;timer[bid+blocks]=Int64(CUDA.clock(UInt64));end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const CLOCK_TEMPLATE=CUDA.zeros(Float32,512)
const CLOCK_SPECS=Dict(b=>JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=b,shmem_size=2048,sync=false) for b in (1,32,8192))
clock_study((a,o,t,b)->JACC.parallel_for(clock_reduction!,CLOCK_SPECS[b],b*256,a,o,t,Int32(b),CLOCK_TEMPLATE),"jacc")
