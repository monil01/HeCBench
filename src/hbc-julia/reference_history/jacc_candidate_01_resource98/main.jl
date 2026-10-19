using CUDA
import JACC,Atomix
JACC.@init_backend
const MODEL="jacc"
include("../study_device.jl")
@inline hsync(::Val{:jacc})=JACC.sync_workgroup()
@inline hfetch(::Val{:jacc},a,i,v)=first(Atomix.@atomic :monotonic a[Int(i)+1]+v)
@inline hcas(::Val{:jacc},a,i,old,new)=(Atomix.@atomicreplace :monotonic a[Int(i)+1] old=>new).old
function hbc_kernel!(i,d,p,n,m)
 # CUDA-backend static scratch reservation preserves source no-copy shared-memory semantics.
 # This is explicit hybrid device interoperability, not a cross-backend JACC claim.
 sh=CUDA.CuStaticSharedArray(Int32,528)
 index=Int32(i)-Int32(1);tid=index%Int32(1024);block=index÷Int32(1024)
 hbc_body!(d,p,n,m,sh,tid,block,Val(:jacc));nothing
end
function launch_hbc(s)
 spec=JACC.launch_spec(threads=1024,blocks=80,shmem_size=0,sync=false)
 JACC.parallel_for(spec,81920,hbc_kernel!,s.device,s.strides,Int32(s.n),Int32(s.m))
 nothing
end
include("../study_common.jl")
main()
