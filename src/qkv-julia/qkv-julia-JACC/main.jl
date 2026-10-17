using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(bt,oc,out,inp,weight,bias)
 bt-=Int32(1);oc-=Int32(1)
 if bt<Int32(BT)&&oc<Int32(OC)
  @inbounds val=bias[oc+1]
  for k in Int32(0):Int32(CC-1)
   @inbounds val+=inp[bt*Int32(CC)+k+1]*weight[oc*Int32(CC)+k+1]
  end
  @inbounds out[bt*Int32(OC)+oc+1]=val
 end
 return
end
run!(out,inp,w,b)=(JACC.parallel_for(JACC.launch_spec(threads=(16,16),blocks=(cld(BT,16),cld(OC,16)),shmem_size=0,sync=false),(BT,OC),kernel!,out,inp,w,b);nothing)
study(run!)
