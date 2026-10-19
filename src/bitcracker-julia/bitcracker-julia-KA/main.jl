using CUDA,Printf,KernelAbstractions
include("../crypto_device.jl")
include("../study_common.jl")
@kernel function schedule_kernel!(salt,padding,w)
 local_id=@index(Local,Linear)
 group=@index(Group,Linear)
 kernel_w_block(salt,padding,w,UInt32(group-1),UInt32(16),UInt32(local_id-1),UInt32(1024))
 nothing
end
@kernel function decrypt_kernel!(found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w)
 local_id=@index(Local,Linear)
 decrypt_vmk_with_mac(UInt32(1),found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w,UInt32(0),UInt32(256),UInt32(local_id-1),UInt32(1))
 nothing
end
const backend=CUDA.CUDABackend()
schedule!(salt,padding,w)=schedule_kernel!(backend,16)(salt,padding,w;ndrange=16384)
function decrypt!(found,vmk,iv,mac,maciv,computed,v,s,pw,w)
 decrypt_kernel!(backend,256)(found,vmk,iv,mac,maciv,computed,v[1],v[2],v[3],v[4],s[1],s[2],s[3],s[4],pw,w;ndrange=256)
end
bitcracker_study(schedule!,decrypt!)
