using CUDA
include("../study_device.jl")
function keys_kernel(inp,out,ix,n)
 sortbody(inp,out,ix,n,Int32(threadIdx().x-1),Val(false));return
end
function pairs_kernel(inp,out,ix,n)
 sortbody(inp,out,ix,n,Int32(threadIdx().x-1),Val(true));return
end
function launchcase(inp,out,ix,n,mode)
 if mode==0
  CUDA.@cuda threads=32 blocks=1 keys_kernel(inp,out,ix,Int32(n))
 else
  CUDA.@cuda threads=32 blocks=1 pairs_kernel(inp,out,ix,Int32(n))
 end
 return
end
include("../study_common.jl")
