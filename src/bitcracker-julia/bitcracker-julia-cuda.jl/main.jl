using CUDA,Printf
include("../crypto_device.jl")
include("../study_common.jl")
function schedule_kernel!(salt,padding,w)
 kernel_w_block(salt,padding,w,UInt32(blockIdx().x-1),UInt32(blockDim().x),UInt32(threadIdx().x-1),UInt32(gridDim().x))
 return
end
function decrypt_kernel!(found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w)
 decrypt_vmk_with_mac(UInt32(1),found,vmk,iv,mac,maciv,computed,v0,v1,v2,v3,s0,s1,s2,s3,pw,w,UInt32(blockIdx().x-1),UInt32(blockDim().x),UInt32(threadIdx().x-1),UInt32(gridDim().x))
 return
end
schedule!(salt,padding,w)=(@cuda threads=16 blocks=1024 schedule_kernel!(salt,padding,w))
function decrypt!(found,vmk,iv,mac,maciv,computed,v,s,pw,w)
 @cuda threads=256 blocks=1 decrypt_kernel!(found,vmk,iv,mac,maciv,computed,v[1],v[2],v[3],v[4],s[1],s[2],s[3],s[4],pw,w)
end
bitcracker_study(schedule!,decrypt!)
