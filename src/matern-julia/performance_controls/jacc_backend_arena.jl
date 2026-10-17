# Diagnostic CUDA-backend arena control: removes JACC.shared template initialization + its barrier.
# Primary framework-API source/result is retained unchanged.
using CUDA,JACC
function matern_jacc!(gx,gy,n::Int32,l::Float32,sources,targets,weights,result,template)
 tx=mod(Int32(gx)-Int32(1),Int32(16));ty=Int32(gy)-Int32(1);px=Int32(gx)-Int32(1)
 shared=CUDA.CuDynamicSharedArray(Float32,1048)
 if px>=n||ty>=Int32(50);return;end
 @inbounds begin
  if ty==0
   for i=Int32(0):Int32(2);shared[Int32(800)+tx*Int32(3)+i+Int32(1)]=targets[px*Int32(3)+i+Int32(1)];end
  end
  if tx==0
   for i=Int32(0):Int32(2);shared[Int32(848)+ty*Int32(3)+i+Int32(1)]=sources[ty*Int32(3)+i+Int32(1)];end
   shared[Int32(998)+ty+Int32(1)]=weights[ty+Int32(1)]
  end
 end
 JACC.sync_workgroup();squared=0f0
 @inbounds for i=Int32(0):Int32(2)
  delta=shared[Int32(800)+tx*Int32(3)+i+Int32(1)]-shared[Int32(848)+ty*Int32(3)+i+Int32(1)];squared+=delta*delta
 end
 diff=sqrt(squared)
 @inbounds shared[tx*Int32(50)+ty+Int32(1)]=(1f0+sqrt(5f0)*diff/l+5f0*squared/(3f0*l*l))*exp(-sqrt(5f0)*diff/l)*shared[Int32(998)+ty+Int32(1)]
 JACC.sync_workgroup()
 if ty==0
  res=0f0
  @inbounds for i=Int32(0):Int32(49);res+=shared[tx*Int32(50)+i+Int32(1)];end
  @inbounds result[px+Int32(1)]=res
 end
 return
end
const TEMPLATE=CUDA.zeros(Float32,1048)
const SPEC=JACC.launch_spec(;threads=(16,64),blocks=(308,1),shmem_size=4192,sync=false)
function launch_jacc(n,l,s,t,w,out)
 Int(n)==4913||error("Named study extent")
 JACC.parallel_for(SPEC,(4928,64),matern_jacc!,n,l,s,t,w,out,TEMPLATE)
end
include("../study_common.jl")
matern_study(launch_jacc)
