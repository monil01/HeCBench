using CUDA,Printf,JACC
include("../crypto_device.jl")
include("../study_common.jl")
@inline function schedule_kernel!(i,salt,padding,w)
 index=UInt32(i-1)
 kernel_w_block(salt,padding,w,index÷UInt32(16),UInt32(16),index%UInt32(16),UInt32(1024))
 return nothing
end
@inline function decrypt_kernel!(i,found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w)
 decrypt_vmk_with_mac(UInt32(1),found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w,UInt32(0),UInt32(256),UInt32(i-1),UInt32(1))
 return nothing
end
function schedule!(salt,padding,w)
 spec=JACC.launch_spec(;threads=16,shmem_size=0,sync=false)
 JACC.parallel_for(spec,16384,schedule_kernel!,salt,padding,w)
end
function decrypt!(found,vmk,iv,mac,maciv,computed,v,s,pw,w)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 JACC.parallel_for(spec,256,decrypt_kernel!,found,vmk,iv,mac,maciv,computed,v[1],v[2],v[3],v[4],s[1],s[2],s[3],s[4],pw,w)
end
bitcracker_study(schedule!,decrypt!)
