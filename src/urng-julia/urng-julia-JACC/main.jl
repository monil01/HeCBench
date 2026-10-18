using CUDA,JACC
include("../study_portable_common.jl")
@inline function urng_kernel(i,input,output,n,template)
 lane=Int32((i-1)%256+1);shuffle=JACC.shared(template)
 if i<=n;@inbounds output[i]=urng_pixel(input[i],lane,shuffle);end
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(input,output,n)
 total=Int(cld(n,Int32(256)))*256;template=CUDA.zeros(Int32,4096)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=total÷256,shmem_size=16384,sync=false)
 ()->JACC.parallel_for(spec,total,urng_kernel,input,output,n,template)
end
urng_study(factory)
