using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"study_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=CUDA.CUDABackend()
function launch(k,bs,x,g,o,dx,B,C,H,W)
 dims=(UInt64(B),UInt64(C),UInt64(H),UInt64(W));S=B*C*H*W
 k==1&&return up1!(BACKEND,bs)(x,o,dims...;ndrange=cld(S,bs)*bs)
 k==2&&return up2!(BACKEND,(bs,bs,1))(x,o,dims...;ndrange=(cld(W,bs)*bs,cld(H,bs)*bs,B*C))
 k==3&&return up3!(BACKEND,bs)(g,dx,dims...;ndrange=cld(S,bs)*bs)
 k==4&&return up4!(BACKEND,(bs,bs,1))(g,dx,dims...;ndrange=(cld(W,bs)*bs,cld(H,bs)*bs,B*C))
end
study((T,n)->KernelAbstractions.allocate(BACKEND,T,n),launch)
