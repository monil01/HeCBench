using CUDA,JACC
include("../resize_device.jl")
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=29184,shmem_size=0,sync=false)
function resize_kernel!(i,mode,out,inp,outsize,oh,ow,ih,iw,fy,fx)
 resize_site!(mode,out,inp,Int32(i)-Int32(1),Int32(29184*256),outsize,oh,ow,ih,iw,fy,fx)
 nothing
end
launch(c)=(JACC.parallel_for(resize_kernel!,SPEC,29184*256,c.mode,c.output,c.input,UInt64(c.no),c.oh,c.ow,c.ih,c.iw,c.fy,c.fx);nothing)
include("../study_common.jl")
resize_study(launch)
