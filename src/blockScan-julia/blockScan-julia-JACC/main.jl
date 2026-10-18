using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
const SCAN_BACKEND=JACC.get_backend(:cuda)
const SCAN_TEMPLATE=CUDA.zeros(Int32,1024)
const SCAN_SPECS=Dict(bt=>JACC.LaunchSpec{typeof(SCAN_BACKEND)}(;threads=bt,blocks=1,shmem_size=4096,sync=false) for bt in STUDY_BTS)
# JACC.shared copies the complete template before the explicit input load.
function exclusive_scan_jacc!(i,input,output,::Val{BT},template) where {BT}
 tid=Int32(i);bt=Int32(BT);tile=JACC.shared(template)
 for j in tid:bt:Int32(1024)
  @inbounds tile[j]=input[j]
 end
 JACC.sync_workgroup()
 stride=Int32(2)
 while stride<=Int32(1024)
  for k in tid:bt:(Int32(1024)÷stride)
   right=k*stride
   @inbounds tile[right]+=tile[right-stride÷Int32(2)]
  end
  JACC.sync_workgroup()
  stride*=Int32(2)
 end
 if tid==1
  @inbounds output[1025]=tile[1024]
  @inbounds tile[1024]=Int32(0)
 end
 JACC.sync_workgroup()
 stride=Int32(1024)
 while stride>=Int32(2)
  for k in tid:bt:(Int32(1024)÷stride)
   right=k*stride;left=right-stride÷Int32(2)
   @inbounds begin
    temp=tile[left];tile[left]=tile[right];tile[right]+=temp
   end
  end
  JACC.sync_workgroup()
  stride÷=Int32(2)
 end
 for j in tid:bt:Int32(1024)
  @inbounds output[j]=tile[j]
 end
 nothing
end
function study_launch!(a,o,::Val{BT},::Val{IPT}) where {BT,IPT}
 JACC.parallel_for(exclusive_scan_jacc!,SCAN_SPECS[BT],Int32(BT),a,o,Val(BT),SCAN_TEMPLATE)
end
study_main(study_launch!)
