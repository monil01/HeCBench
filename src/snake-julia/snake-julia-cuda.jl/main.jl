using CUDA,Printf
include("../study_device.jl")
function snake_kernel(A,B,O,N,T)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=N;@inbounds O[i]=snake_filter(A,B,(i-Int32(1))*Int32(8),T);end
 nothing
end
factory(A,B,O,N)=T->(@cuda threads=256 blocks=cld(N,256) snake_kernel(A,B,O,Int32(N),Int32(T)))
include("../study_common.jl")
snake_study(factory)
