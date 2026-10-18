using CUDA
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(k,pref,w,gin,a,gout,n,batch)
 ni=Int32(n);bi=Int32(batch);blocks=cld(n,256)
 if k==1;CUDA.@cuda threads=256 blocks=blocks forward_kernel!(pref,a,ni)
 elseif k==2;CUDA.@cuda threads=256 blocks=blocks backward_kernel!(a,gin,gout,ni)
 elseif k==3;CUDA.@cuda threads=256 blocks=blocks update_kernel!(w,gout,ni)
 else;CUDA.@cuda threads=256 blocks=blocks prefetch_kernel!(pref,ni,bi)
 end
end
study((T,n)->CUDA.zeros(T,n),launch)
