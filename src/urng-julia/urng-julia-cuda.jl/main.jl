using CUDA
include("../study_common.jl")
function urng_kernel(input,output,n)
 i=(Int32(blockIdx().x)-1)*Int32(256)+Int32(threadIdx().x);lane=Int32(threadIdx().x);shuffle=CuStaticSharedArray(Int32,4096)
 if i<=n;@inbounds output[i]=urng_pixel(input[i],lane,shuffle);end
 return
end
factory(input,output,n)=()->(@cuda threads=256 blocks=cld(n,Int32(256)) urng_kernel(input,output,n))
urng_study(factory)
