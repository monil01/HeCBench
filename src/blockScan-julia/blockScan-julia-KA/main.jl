using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
# Cooperative Blelloch scan; one algorithm compared with all three native CUB strategies.
@kernel unsafe_indices=true function exclusive_scan_ka!(input,output,::Val{BT}) where {BT}
 tid=@index(Local,Linear)
 tile=@localmem Int32 (1024,)
 for j in tid:BT:1024
  @inbounds tile[j]=input[j]
 end
 @synchronize
 stride=2
 while stride<=1024
  for k in tid:BT:(1024÷stride)
   i=k*stride
   @inbounds tile[i]+=tile[i-stride÷2]
  end
  @synchronize
  stride*=2
 end
 if tid==1
  @inbounds output[1025]=tile[1024]
  @inbounds tile[1024]=Int32(0)
 end
 @synchronize
 stride=1024
 while stride>=2
  for k in tid:BT:(1024÷stride)
   right=k*stride;left=right-stride÷2
   @inbounds begin
    temp=tile[left];tile[left]=tile[right];tile[right]+=temp
   end
  end
  @synchronize
  stride÷=2
 end
 for j in tid:BT:1024
  @inbounds output[j]=tile[j]
 end
end
function study_launch!(a,o,::Val{BT},::Val{IPT}) where {BT,IPT}
 exclusive_scan_ka!(CUDABackend(),BT)(a,o,Val(BT);ndrange=BT)
end
study_main(study_launch!)
