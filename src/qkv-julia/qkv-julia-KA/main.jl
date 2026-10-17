using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(out,inp,weight,bias)
 bt,oc=@index(Global,NTuple);bt-=Int32(1);oc-=Int32(1)
 if bt<Int32(BT)&&oc<Int32(OC)
  @inbounds val=bias[oc+1]
  for k in Int32(0):Int32(CC-1)
   @inbounds val+=inp[bt*Int32(CC)+k+1]*weight[oc*Int32(CC)+k+1]
  end
  @inbounds out[bt*Int32(OC)+oc+1]=val
 end
end
run!(out,inp,w,b)=(kernel!(get_backend(out),(16,16))(out,inp,w,b;ndrange=(BT,OC));nothing)
study(run!)
