using CUDA
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,block,x,g,o,dx,B,C,H,W)
 Bu,Cu,Hu,Wu=UInt64(B),UInt64(C),UInt64(H),UInt64(W);S=UInt64(B*C*H*W)
 if k==1;CUDA.@cuda threads=block blocks=cld(Int(S),block) upsample_forward_kernel!(x,o,Bu,Cu,Hu,Wu,S)
 elseif k==3;CUDA.@cuda threads=block blocks=cld(Int(S),block) upsample_backward_kernel!(g,dx,Bu,Cu,Hu,Wu,S)
 elseif k==2;CUDA.@cuda threads=(block,block,1) blocks=(cld(W,block),cld(H,block),B*C) upsample_forward_kernel2!(x,o,Bu,Cu,Hu,Wu)
 else;CUDA.@cuda threads=(block,block,1) blocks=(cld(W,block),cld(H,block),B*C) upsample_backward_kernel2!(g,dx,Bu,Cu,Hu,Wu)
 end
end
study((T,n)->CUDA.zeros(T,n),launch)
