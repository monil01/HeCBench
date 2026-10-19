using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function add_kernel!(table,entries,keys,nkeys,failed,nfailure,maxfailure,::Val{IP})where IP
 sh=CUDA.CuStaticSharedArray(UInt32,2+2*(32÷(IP+1))+IP*32)
 lane=Int32(CUDA.threadIdx().x)-Int32(1);block=Int32(CUDA.blockIdx().x)-Int32(1)
 add_body!(table,entries,keys,nkeys,failed,nfailure,maxfailure,sh,block,lane,Val(IP))
 return
end
function launch_add(s,stream,n,::Val{IP})where IP
 CUDA.@cuda threads=32 blocks=cld(n,32) stream=stream add_kernel!(s.table.data,Int32(s.entries),s.keys[s.stream_id].data,Int32(n),s.failed.data,s.nf.data,Int32(s.n),Val(IP))
 nothing
end
include("../study_common.jl")
main()
