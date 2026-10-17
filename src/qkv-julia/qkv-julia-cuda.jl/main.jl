using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(out,inp,weight,bias)
 bt=(blockIdx().x-Int32(1))*Int32(16)+threadIdx().x-Int32(1);oc=(blockIdx().y-Int32(1))*Int32(16)+threadIdx().y-Int32(1)
 if bt<Int32(BT)&&oc<Int32(OC)
  @inbounds val=bias[oc+1]
  for k in Int32(0):Int32(CC-1)
   @inbounds val+=inp[bt*Int32(CC)+k+1]*weight[oc*Int32(CC)+k+1]
  end
  @inbounds out[bt*Int32(OC)+oc+1]=val
 end
 return
end
run!(out,inp,w,b)=(@cuda threads=(16,16) blocks=(cld(BT,16),cld(OC,16)) kernel!(out,inp,w,b);nothing)
study(run!)
