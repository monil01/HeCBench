using CUDA,KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel function add_kernel!(table,entries,keys,nkeys,failed,nfailure,maxfailure,::Val{IP})where IP
 sh=@localmem UInt32 (2+2*(32÷(IP+1))+IP*32,)
 lane=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)
 add_body!(table,entries,keys,nkeys,failed,nfailure,maxfailure,sh,block,lane,Val(IP))
end
function launch_add(s,stream,n,::Val{IP})where IP
 add_kernel!(CUDABackend(),32)(s.table.data,Int32(s.entries),s.keys[s.stream_id].data,Int32(n),s.failed.data,s.nf.data,Int32(s.n),Val(IP);ndrange=cld(n,32)*32)
 nothing
end
include("../study_common.jl")
main()
