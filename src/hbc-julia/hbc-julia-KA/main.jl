using CUDA,KernelAbstractions
import Atomix
const MODEL="ka"
include("../study_device.jl")
@inline hsync(::Val{:ka})=KernelAbstractions.@synchronize()
@inline hfetch(::Val{:ka},a,i,v)=first(Atomix.@atomic :monotonic a[Int(i)+1]+v)
@inline hcas(::Val{:ka},a,i,old,new)=(Atomix.@atomicreplace :monotonic a[Int(i)+1] old=>new).old
@kernel cpu=false function hbc_kernel!(d,p,n,m)
 sh=@localmem Int32 (528,)
 tid=Int32(@index(Local,Linear))-Int32(1);block=Int32(@index(Group,Linear))-Int32(1)
 hbc_body!(d,p,n,m,sh,tid,block,Val(:ka))
end
function launch_hbc(s)
 hbc_kernel!(CUDABackend(),1024)(s.device,s.strides,Int32(s.n),Int32(s.m);ndrange=81920)
 nothing
end
include("../study_common.jl")
main()
