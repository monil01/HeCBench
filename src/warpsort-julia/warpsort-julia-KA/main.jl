using CUDA, KernelAbstractions
include("../study_device.jl")
@kernel function keys_kernel(inp,out,ix,n)
 lane=@index(Local,Linear)
 sortbody(inp,out,ix,n,Int32(lane-1),Val(false))
end
@kernel function pairs_kernel(inp,out,ix,n)
 lane=@index(Local,Linear)
 sortbody(inp,out,ix,n,Int32(lane-1),Val(true))
end
function launchcase(inp,out,ix,n,mode)
 if mode==0
  keys_kernel(CUDABackend(),32)(inp,out,ix,Int32(n);ndrange=32)
 else
  pairs_kernel(CUDABackend(),32)(inp,out,ix,Int32(n);ndrange=32)
 end
 return
end
include("../study_common.jl")
