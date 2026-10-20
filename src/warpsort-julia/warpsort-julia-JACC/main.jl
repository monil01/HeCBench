using CUDA, JACC
include("../study_device.jl")
function keys_kernel(i,inp,out,ix,n)
 sortbody(inp,out,ix,n,Int32(i-1),Val(false));return
end
function pairs_kernel(i,inp,out,ix,n)
 sortbody(inp,out,ix,n,Int32(i-1),Val(true));return
end
const SPEC=JACC.launch_spec(threads=32,blocks=1,shmem_size=0,sync=false)
function launchcase(inp,out,ix,n,mode)
 if mode==0
  JACC.parallel_for(SPEC,32,keys_kernel,inp,out,ix,Int32(n))
 else
  JACC.parallel_for(SPEC,32,pairs_kernel,inp,out,ix,Int32(n))
 end
 return
end
include("../study_common.jl")
